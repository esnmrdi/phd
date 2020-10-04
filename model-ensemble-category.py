# %%
# Generalizing Fuel and Emission Models to Categories using Ensemble Learning
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
import pandas as pd
import numpy as np
import csv
import keras
import h5py
import pickle
import resource
import gc
from keras.models import load_model
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler


# %%
# Load settings of best models
def load_best_ensemble_settings(sheet):
    directory = "../../Google Drive/Academia/PhD Thesis/Charts, Tables, Forms, Flowcharts, Spreadsheets, Figures/"
    input_file = "Paper III - Ensemble Lookback Ranking.xlsx"
    input_path = directory + input_file
    settings = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return settings


# %%
# Load data from Excel to a pandas dataframe
def load_test_vehicle_data(sensor, vehicle):
    directory = "../../Google Drive/Academia/PhD Thesis/Field Experiments/{0}/{1}/Processed/RNN/".format(
        sensor, vehicle
    )
    input_file = "{0} - RNN - 05.xlsx".format(vehicle)
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name="Sheet1", header=0)
    return df


# %%
# Save the predicted fields back to Excel file
def save_ensemble_lookback_data(
    df, dependent, criterion, category, test_vehicle, vehicle
):
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/ENSEMBLE_LOOKBACK_PREDICTIONS/"
    )
    output_file = "{0} - {1} - {2} - {3} - {4}.xlsx".format(
        dependent, criterion, category, test_vehicle, vehicle
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print(
        "{0} - {1} - {2} - {3} - {4} -> Lookback Ensemble data saved!".format(
            dependent, criterion, category, test_vehicle, vehicle
        )
    )
    return None


# %%
# Log ensemble model settings and corresponding scores to a file (one by one)
def log_ensemble_settings_and_score(row, output_file):
    directory = "../../Google Drive/Academia/PhD Thesis/Charts, Tables, Forms, Flowcharts, Spreadsheets, Figures/"
    output_path = directory + output_file
    with open(output_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return None


# %%
# Definition of the custom loss function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.array(y_pred) - np.array(y_true))))


# %%
# Generate time-series input for the desired lookback order
def generate_rnn_input(df, features, dependent, lookback):
    dataset = df[features + [dependent]].to_numpy()
    dim = len(features)
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback : i + 1, :dim])
        y.append(dataset[i, dim])
    X, y = np.array(X), np.array(y)
    return X, y


# %%
# Load pre-trained rnn model from .h5 file
def load_from_h5(vehicle, dependent, lookback):
    directory = "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/RNN/"
    input_file = "{0} - RNN - {1} - L{2}.h5".format(vehicle, dependent, lookback)
    input_path = directory + input_file
    model = load_model(input_path, compile=False)
    return model


# %%
# Load ensemble fitted model over separate lookback predictions for each vehicle from .sav file
def load_from_sav(vehicle, dependent, estimator):
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/ENSEMBLE_LOOKBACK_MODELS/"
    )
    input_file = "{0} - {1} - {2}.sav".format(vehicle, dependent, estimator)
    input_path = directory + input_file
    with open(input_path, "rb") as reader:
        model = pickle.load(reader)
    return model


# %%
# Scale the input data
def scale(df, features, dependent):
    df_temp = df.copy()
    scaler_X = StandardScaler().fit(df_temp[features])
    scaler_y = StandardScaler().fit(df_temp[[dependent]])
    df_temp[features] = scaler_X.transform(df_temp[features])
    df_temp[[dependent]] = scaler_y.transform(df_temp[[dependent]])
    return df_temp, scaler_X, scaler_y


# %%
# Generate predictions of component models
# first lookback RNNs and then, their esnemble
# using test vehicle data
def components_predictions(
    sensor, dependent, criterion, category, df_category, test_vehicle, settings
):
    features = settings["FEATURES"]
    batch_size = settings["BATCH_SIZE"]
    df_test = load_test_vehicle_data(sensor, test_vehicle)
    df_output = pd.DataFrame()
    df_output[["DATETIME"] + features + [dependent]] = df_test[
        ["DATETIME"] + features + [dependent]
    ][6:]
    for _, row in df_category.iterrows():
        # vehicle = row["VEHICLE"].value[0]
        vehicle = row["VEHICLE"]
        df_lookbacks = pd.DataFrame()
        df_lookbacks[["DATETIME"] + features + [dependent]] = df_test[
            ["DATETIME"] + features + [dependent]
        ]
        for lookback in range(1, 7):
            df_test_tmp, scaler_X, scaler_y = scale(df_test, features, dependent)
            X, y = generate_rnn_input(df_test_tmp, features, dependent, lookback)
            trim_size = len(X) % batch_size
            trim_length = len(X) - trim_size
            X, y = X[:trim_length], y[:trim_length]
            rnn_model = load_from_h5(vehicle, dependent, lookback)
            y_pred = rnn_model.predict(X, batch_size=batch_size)
            y_pred = scaler_y.inverse_transform(y_pred)
            y_pred = np.insert(y_pred, 0, np.repeat(np.nan, lookback))
            y_pred = np.append(y_pred, np.repeat(np.nan, trim_size))
            df_lookbacks["{0}_PRED_L{1}".format(dependent, lookback)] = y_pred
        df_lookbacks.dropna(inplace=True)
        df_output = df_output[:len(df_lookbacks)]
        ensemble_estimator = row["ESTIMATOR"]
        ensemble_model = load_from_sav(vehicle, dependent, ensemble_estimator)
        lookback_features = [
            "{0}_PRED_L{1}".format(dependent, str(i)) for i in range(1, 7)
        ]
        df_output["{0}_PRED_{1}".format(dependent, vehicle)] = df_lookbacks[
            "{0}_PRED_{1}".format(dependent, ensemble_estimator)
        ] = ensemble_model.predict(df_lookbacks[lookback_features])
        save_ensemble_lookback_data(
            df_lookbacks, dependent, criterion, category, test_vehicle, vehicle
        )
        score_ensemble = rmse(df_lookbacks[dependent], df_output["{0}_PRED_{1}".format(dependent, vehicle)])
        score_components = [
            rmse(df_lookbacks[dependent], df_lookbacks[col])
            for col in df_lookbacks[lookback_features]
        ]
        row = [
            dependent,
            criterion,
            category,
            test_vehicle,
            vehicle,
            ensemble_estimator,
            score_ensemble,
        ] + score_components
        log_ensemble_settings_and_score(
            row,
            "Paper III - Ensemble Lookback Results - Test Vehicle of Category as Input.csv",
        )
    df_output.dropna(inplace=True)
    return df_output


# %%
# General settings
pd.options.mode.chained_assignment = None
SETTINGS = {
    "FEATURES": ["SPD_KH", "ACC_MS2", "ALT_M"],
    "BATCH_SIZE": 64,
    "CATEGORIZATION_CRITERIA": (
        "AGE_RANGE",
        "SEGMENT",
        "ENGINE_TYPE",
        "ENGINE_SIZE_RANGE",
        "TRANSMISSION",
        "TOTAL_WEIGHT_RANGE",
    ),
    "SENSOR_DEPENDENT": {
        "Veepeak": ("FCR_LH",),
        "3DATX parSYNC Plus": ("CO2_KGS", "NO_KGS", "NO2_KGS", "PM_KGS"),
    },
    "ESTIMATORS": (
        LinearRegression(normalize=True),
        Ridge(alpha=0.1, normalize=True),
        Ridge(alpha=1.0, normalize=True),
        SVR(C=1.0),
        SVR(C=10.0),
        DecisionTreeRegressor(splitter="best"),
        DecisionTreeRegressor(splitter="random"),
        GradientBoostingRegressor(n_estimators=10),
        GradientBoostingRegressor(n_estimators=100),
        AdaBoostRegressor(n_estimators=10),
        AdaBoostRegressor(n_estimators=100),
        RandomForestRegressor(n_estimators=10),
        RandomForestRegressor(n_estimators=100),
        MLPRegressor(hidden_layer_sizes=(100,)),
        MLPRegressor(
            hidden_layer_sizes=(
                100,
                100,
            )
        ),
    ),
}

# %%
# Application of ensemble learning methods
# including Linear Regression, Ridge Regression, SVR, Decision Tree,
# Gradient Boosting, Ada Boosting, Random Forest, and MLP
# Batch execution on all the vehicles
rng = check_random_state(0)
sensor_dependent = SETTINGS["SENSOR_DEPENDENT"]
categorization_criteria = SETTINGS["CATEGORIZATION_CRITERIA"]
ensemble_settings = load_best_ensemble_settings("Best Ensemble Settings")
estimators = SETTINGS["ESTIMATORS"]
for sensor, dependents in sensor_dependent.items():
    for dependent in dependents:
        dependent_subset = ensemble_settings.loc[
            (ensemble_settings["DEPENDENT"] == dependent)
        ]
        for criterion in categorization_criteria:
            for category, df_category in dependent_subset.groupby(criterion):
                if len(df_category) > 2:
                    test_row_index = np.random.choice(
                        df_category.index, 1, replace=False
                    )
                    test_row = df_category.loc[test_row_index]
                    test_vehicle = test_row["VEHICLE"].values[0]
                    df_category.drop(test_row_index, inplace=True)
                    print("---------------------")
                    print(
                        "Dependent: {0} | Criterion: {1} | Category: {2}".format(
                            dependent, criterion, category
                        )
                    )
                    print(
                        "Number of vehicles in category (excluding test): {}".format(
                            len(df_category)
                        )
                    )
                    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    print("---------------------")
                    df_input = components_predictions(
                        sensor,
                        dependent,
                        criterion,
                        category,
                        df_category,
                        test_vehicle,
                        SETTINGS,
                    )
                    ensemble_features = [
                        "{0}_PRED_{1}".format(dependent, row["VEHICLE"])
                        for _, row in df_category.iterrows()
                    ]
                    X, y = df_input[ensemble_features], df_input[dependent]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=rng
                    )
                    df_output = pd.DataFrame()
                    df_output[dependent] = y_test
                    df_output[ensemble_features] = X_test
                    for estimator in estimators:
                        model = estimator.fit(X_train, y_train)
                        ensemble = model.predict(X_test)
                        df_output["{0}_PRED_{1}".format(dependent, estimator)] = ensemble
                        score_ensemble = rmse(y_test, ensemble)
                        score_components = [rmse(y_test, X_test[col]) for col in X_test]
                        row = [
                            dependent,
                            criterion,
                            category,
                            len(df_category),
                            estimator,
                            score_ensemble,
                        ] + score_components
                        log_ensemble_settings_and_score(
                            row, "Paper III - Ensemble Category Results.csv"
                        )
                        print("{0}\nScore: {1}".format(estimator, score_ensemble))
# %%

# %%
