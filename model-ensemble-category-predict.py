# %%
# Ensemble Forecasting of Previously Trained Ensemble Lookback Models
# Over Categories Defined using Different Criteria
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
import pandas as pd
import numpy as np
import pickle
import os
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
    input_file = "Paper III - Ensemble Category Ranking.xlsx"
    input_path = directory + input_file
    settings = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return settings


# %%
# Load data from Excel to a pandas dataframe
def load_data_from_Excel(input_path):
    df = pd.read_excel(input_path, sheet_name="Sheet1", header=0)
    return df


# %%
# Save back the predictions to Excel
def save_data_to_Excel(df, dependent, criterion, category):
    directory = "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/ENSEMBLE_CATEGORY_PREDICTIONS/"
    output_file = "{0} - {1} - {2}.xlsx".format(dependent, criterion, category)
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print(
        "{0} - {1} - {2} -> Output is successfully saved to Excel.".format(
            dependent, criterion, category
        )
    )
    return None


# %%
# Save fitted model to .sav file
def save_to_sav(model, dependent, criterion, category, estimator):
    directory = "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/ENSEMBLE_CATEGORY_MODELS/"
    output_file = "{0} - {1} - {2} - {3}.sav".format(
        dependent, criterion, category, estimator
    )
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
    criterion = ensemble_setting["CRITERION"]
    category = ensemble_setting["CATEGORY"]
    dependent = ensemble_setting["DEPENDENT"]
    estimator = eval(ensemble_setting["ESTIMATOR"])
    print(criterion, category, dependent, estimator)
    df = pd.DataFrame()
    directory = "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/ENSEMBLE_LOOKBACK_PREDICTIONS"
    vehicles = []
    for file in os.listdir(directory):
        if file.startswith("{0} - {1} - {2}".format(dependent, criterion, category)):
            input_path = os.path.join(directory, file)
            vehicle = file.split(" - ")[-1][:-5]
            df_vehicle = load_data_from_Excel(input_path)
            target_col = df_vehicle.columns[-1]
            df["{0}_PRED_{1}".format(dependent, vehicle)] = df_vehicle[target_col]
            vehicles.append(vehicle)
            test_vehicle = file.split(" - ")[-2]
    df[["DATETIME", "SPD_KH", "ACC_MS2", "ALT_M", dependent]] = df_vehicle[
        ["DATETIME", "SPD_KH", "ACC_MS2", "ALT_M", dependent]
    ]
    df.dropna(inplace=True)
    features = ["{0}_PRED_{1}".format(dependent, vehicle) for vehicle in vehicles]
    X, y = df[features], df[dependent]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=rng)
    ensemble = estimator.fit(X_train, y_train)
    save_to_sav(ensemble, dependent, criterion, category, estimator)
    df["{0}_PRED_{1}".format(dependent, estimator)] = ensemble.predict(X)
    save_data_to_Excel(df, dependent, criterion, category)

# %%
