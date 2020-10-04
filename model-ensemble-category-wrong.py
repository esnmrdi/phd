# %%
# Generalizing Fuel and Emission Models to Categories using Ensemble Learning
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import (
    BaggingRegressor,
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
def load_data_from_Excel(sensor, vehicle, settings):
    input_type = settings["INPUT_TYPE"]
    input_index = settings["INPUT_INDEX"]
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + sensor
        + "/"
        + vehicle
        + "/Processed/"
        + input_type
        + "/"
    )
    input_file = "{0} - {1} - {2}.xlsx".format(vehicle, input_type, input_index)
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name="Sheet1", header=0)
    return df


# %%
# Log model settings and corresponding scores to a file (one by one)
def log_model_settings_and_score(row, output_file):
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
# General settings
pd.options.mode.chained_assignment = None
SETTINGS = {
    "INPUT_TYPE": "ENSEMBLE",
    "INPUT_INDEX": "06",
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
        # SVR(C=1.0),
        # SVR(C=10.0),
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
categorization_criteria = SETTINGS["CATEGORIZATION_CRITERIA"]
sensor_dependent = SETTINGS["SENSOR_DEPENDENT"]
ensemble_settings = load_best_ensemble_settings("Best Ensemble Settings")
estimators = SETTINGS["ESTIMATORS"]
for criterion in categorization_criteria:
    for sensor, dependents in sensor_dependent.items():
        for dependent in dependents:
            feature = "{0}_PRED_ENSEMBLE_LOOKBACK".format(dependent)
            dependent_subset = ensemble_settings.loc[
                (ensemble_settings["DEPENDENT"] == dependent)
            ]
            for category, df_category in dependent_subset.groupby(criterion):
                print(
                    "\nCriterion: {0} | Dependent: {1} | Category: {2}".format(
                        criterion, dependent, category
                    )
                )
                print("Number of vehicles in category: {}".format(len(df_category)))
                print("---------------------")
                df_input = pd.DataFrame()
                score_components = []
                for index, row in df_category.iterrows():
                    vehicle = row["VEHICLE"]
                    lookback_estimator = row["ESTIMATOR"]
                    rmse_lookback_ensemble = row["RMSE_ENSEMBLE"]
                    score_components.append(rmse_lookback_ensemble)
                    df_vehicle = load_data_from_Excel(sensor, vehicle, SETTINGS)
                    df_vehicle.rename(
                        columns={
                            "{0}_PRED_{1}".format(
                                dependent, lookback_estimator
                            ): feature
                        },
                        inplace=True,
                    )
                    df_temp = df_vehicle[[dependent, feature]]
                    df_input = df_input.append(df_temp, ignore_index=True)

                x, y = df_input[feature], df_input[dependent]
                x, y = np.array(x).reshape(-1, 1), np.array(y)
                for estimator in estimators:
                    ensemble = (
                        BaggingRegressor(
                            base_estimator=estimator, n_estimators=10, random_state=rng
                        )
                        .fit(x, y)
                        .predict(x)
                    )
                    score_ensemble = rmse(y, ensemble)
                    print("{0}\nScore: {1}".format(estimator, score_ensemble))
                    row = [
                        criterion,
                        dependent,
                        category,
                        len(df_category),
                        estimator,
                        score_ensemble,
                    ] + score_components
                    log_model_settings_and_score(
                        row, "Paper III - Ensemble Category Results.csv"
                    )

# %%

# %%
