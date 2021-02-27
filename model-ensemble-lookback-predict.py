# %%
# Ensemble Forecasting of RNN Models Trained for Lookbacks Ranging from 1 to 6
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
import pandas as pd
import numpy as np
import pickle
from scipy.stats.mstats import trimmed_mean, winsorize
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
def load_data_from_Excel(sensor, vehicle):
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + sensor
        + "/"
        + vehicle
        + "/Processed/RNN/"
    )
    input_file = "{0} - RNN - 05.xlsx".format(vehicle)
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name="Sheet1", header=0)
    return df


# %%
# Save back the predictions to Excel
def save_data_to_Excel(df, sensor, vehicle):
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + sensor
        + "/"
        + vehicle
        + "/Processed/ENSEMBLE/"
    )
    output_file = "{0} - ENSEMBLE - 06.xlsx".format(vehicle)
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print("{0} -> Output is successfully saved to Excel.".format(vehicle))
    return None


# %%
# Save fitted model to .sav file
def save_to_sav(model, vehicle, dependent, estimator):
    directory = "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/ENSEMBLE_LOOKBACK_MODELS/"
    output_file = "{0} - {1} - {2}.sav".format(vehicle, dependent, estimator)
    output_path = directory + output_file
    with open(output_path, "wb") as writer:
        pickle.dump(model, writer)


# %%
# General settings
pd.options.mode.chained_assignment = None

# %%
# Batch execution for the best ensemble models dedicated to each vehicle-dependent pair
best_ensemble_settings = load_best_ensemble_settings("Best Ensemble Settings")
old_vehicle = ""
rng = check_random_state(0)
for index, ensemble_setting in best_ensemble_settings.iterrows():
    vehicle = ensemble_setting["VEHICLE"]
    dependent = ensemble_setting["DEPENDENT"]
    features = ["{0}_PRED_L{1}".format(dependent, str(i + 1)) for i in range(6)]
    sensor = "Veepeak" if dependent == "FCR_LH" else "3DATX parSYNC Plus"
    if vehicle != old_vehicle:
        if old_vehicle != "":
            save_data_to_Excel(df, old_sensor, old_vehicle)
        df = load_data_from_Excel(sensor, vehicle)
    estimator = eval(ensemble_setting["ESTIMATOR"])
    print(vehicle, dependent, estimator)
    df.dropna(inplace=True)
    X, y = df[features], df[dependent]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=rng)
    ensemble = estimator.fit(X_train, y_train)
    save_to_sav(ensemble, vehicle, dependent, estimator)
    df["{0}_PRED_{1}".format(dependent, estimator)] = ensemble.predict(X)
    old_vehicle, old_sensor = vehicle, sensor

# %%
