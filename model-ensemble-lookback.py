# %%
# Ensemble Forecasting of RNN Models Trained for Lookbacks Ranging from 1 to 6
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
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
# Load data from Excel to a pandas dataframe
def load_data_from_Excel(vehicle, settings):
    sensor = settings["SENSOR"]
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
# Log model attributes and corresponding scores to a file (one by one)
def log_attributes_and_score(row, output_file):
    directory = "../../Google Drive/Academia/PhD Thesis/Charts, Tables, Forms, Flowcharts, Spreadsheets, Figures/"
    output_path = directory + output_file
    with open(output_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return None


# %%
# Save back the predictions to
def save_data_to_Excel(df, vehicle, settings):
    sensor = settings["SENSOR"]
    output_type = settings["OUTPUT_TYPE"]
    output_index = settings["OUTPUT_INDEX"]
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + sensor
        + "/"
        + vehicle
        + "/Processed/"
        + output_type
        + "/"
    )
    output_file = "{0} - {1} - {2}.xlsx".format(vehicle, output_type, output_index)
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print("{0} -> Output is successfully saved to Excel.".format(vehicle))
    return None


# %%
# Definition of the custom loss function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.array(y_pred) - np.array(y_true))))


# %%
# Experiments to include in modeling
# The boolean points to whether the experiment type is obd_only or pems_included.
EXPERIMENTS = (
    ("009 Renault Logan 2014 (1.6L Manual)", True),
    ("010 JAC J5 2015 (1.8L Auto)", True),
    ("011 JAC S5 2017 (2.0L TC Auto)", True),
    ("012 IKCO Dena 2016 (1.65L Manual)", True),
    ("013 Geely Emgrand7 2014 (1.8L Auto)", True),
    ("014 Kia Cerato 2016 (2.0L Auto)", True),
    ("015 VW Jetta 2016 (1.4L TC Auto)", False),
    ("016 Hyundai Sonata Sport 2019 (2.4L Auto)", True),
    ("017 Chevrolet Trax 2019 (1.4L TC Auto)", True),
    ("018 Hyundai Azera 2006 (3.8L Auto)", True),
    ("019 Hyundai Elantra GT 2019 (2.0L Auto)", True),
    ("020 Honda Civic 2014 (1.8L Auto)", False),
    ("021 Chevrolet N300 2014 (1.2L Manual)", True),
    ("022 Chevrolet Spark GT 2012 (1.2L Manual)", True),
    ("023 Mazda 2 2012 (1.4L Auto)", True),
    ("024 Renault Logan 2010 (1.4L Manual)", True),
    ("025 Chevrolet Captiva 2010 (2.4L Auto)", True),
    ("026 Nissan Versa 2013 (1.6L Auto)", True),
    ("027 Chevrolet Cruze 2011 (1.8L Manual)", True),
    ("028 Nissan Sentra 2019 (1.8L Auto)", True),
    ("029 Ford Escape 2006 (3.0L Auto)", False),
    ("030 Ford Focus 2012 (2.0L Auto)", False),
    ("031 Mazda 3 2016 (2.0L Auto)", False),
    ("032 Toyota RAV4 2016 (2.5L Auto)", False),
    ("033 Toyota Corolla 2019 (1.8L Auto)", False),
    ("034 Toyota Yaris 2015 (1.5L Auto)", False),
    ("035 Kia Rio 2013 (1.6L Auto)", False),
    ("036 Jeep Patriot 2010 (2.4L Auto)", False),
    ("037 Chevrolet Malibu 2019 (1.5L TC Auto)", False),
    ("038 Kia Optima 2012 (2.4L Auto)", False),
    ("039 Honda Fit 2009 (1.5L Auto)", False),
    ("040 Mazda 6 2009 (2.5L Auto)", False),
    ("041 Nissan Micra 2019 (1.6L Auto)", False),
    ("042 Nissan Rouge 2020 (2.5L Auto)", False),
    ("043 Mazda CX-3 2019 (2.0L Auto)", False),
)
EXPERIMENTS = (("010 JAC J5 2015 (1.8L Auto)", True),)

# %%
# General settings
pd.options.mode.chained_assignment = None
plt.style.use("bmh")

# %%
# Trimmed mean ensemble
# Batch execution on all the vehicles
SETTINGS = {
    "SENSOR": "Veepeak",
    "DEPENDENT": "FCR_LH",
    "INPUT_TYPE": "RNN",
    "INPUT_INDEX": "05",
    "OUTPUT_TYPE": "ENSEMBLE",
    "OUTPUT_INDEX": "06",
    "TRIM_LIMITS": [0.0, 0.2, 0.4],
}
rng = check_random_state(0)
dependent = SETTINGS["DEPENDENT"]
features = ["{0}_PRED_L{1}".format(dependent, str(i + 1)) for i in range(6)]
vehicles = (
    (item[0] for item in EXPERIMENTS)
    if SETTINGS["SENSOR"] == "Veepeak"
    else (item[0] for item in EXPERIMENTS if not item[1])
)
for vehicle in vehicles:
    df_input = load_data_from_Excel(vehicle, SETTINGS)
    df_input.dropna(inplace=True)
    X, y = df_input[features], df_input[dependent]
    df_output = pd.DataFrame()
    df_output[dependent] = y
    df_output[features] = X
    for trim_limit in SETTINGS["TRIM_LIMITS"]:
        ensemble = trimmed_mean(X, limits=trim_limit, inclusive=(True, True), axis=1)
        df_output["{0}_PRED_TRIMMED_{1}".format(dependent, str(trim_limit))] = ensemble
        score = round(rmse(y, ensemble), 3)
        row = (dependent, vehicle, trim_limit, score)
        log_attributes_and_score(row, "Paper III - Ensemble - Trimmed Mean Scores.csv")
        print("{0}\nTrimmed mean {1}\nscore {2}.".format(vehicle, trim_limit, score))
    save_data_to_Excel(df_output, vehicle, SETTINGS)

# %%
# Winsorized mean ensemble
# Batch execution on all the vehicles
SETTINGS = {
    "SENSOR": "Veepeak",
    "DEPENDENT": "FCR_LH",
    "INPUT_TYPE": "RNN",
    "INPUT_INDEX": "05",
    "OUTPUT_TYPE": "ENSEMBLE",
    "OUTPUT_INDEX": "07",
    "WINSORIZE_LIMITS": [0.2, 0.4],
}
rng = check_random_state(0)
dependent = SETTINGS["DEPENDENT"]
features = ["{0}_PRED_L{1}".format(dependent, str(i + 1)) for i in range(6)]
vehicles = (
    (item[0] for item in EXPERIMENTS)
    if SETTINGS["SENSOR"] == "Veepeak"
    else (item[0] for item in EXPERIMENTS if not item[1])
)
for vehicle in vehicles:
    df_input = load_data_from_Excel(vehicle, SETTINGS)
    df_input.dropna(inplace=True)
    X, y = df_input[features], df_input[dependent]
    df_output = pd.DataFrame()
    df_output[dependent] = y
    df_output[features] = X
    for winsorize_limit in SETTINGS["WINSORIZE_LIMITS"]:
        ensemble = np.mean(
            winsorize(X, limits=winsorize_limit, inclusive=(True, True), axis=1),
            axis=1,
        )
        df_output[
            "{0}_PRED_WINSORIZED_{1}".format(dependent, str(winsorize_limit))
        ] = ensemble
        score = round(rmse(y, ensemble), 3)
        row = (dependent, vehicle, winsorize_limit, score)
        log_attributes_and_score(
            row, "Paper III - Ensemble - Winsorized Mean Scores.csv"
        )
        print(
            "{0}\nWinsorized mean {1}\nscore {2}.".format(
                vehicle, winsorize_limit, score
            )
        )
    save_data_to_Excel(df_output, vehicle, SETTINGS)

# %%
# Ensemble learning using sophisticated methods
# including Linear Regression, Ridge Regression, SVR, Decision Tree, Gradient Boosting, Ada Boosting, Random Forest, and MLP
# Batch execution on all the vehicles
SETTINGS = {
    "SENSOR": "Veepeak",
    "DEPENDENT": "FCR_LH",
    "INPUT_TYPE": "RNN",
    "INPUT_INDEX": "05",
    "OUTPUT_TYPE": "ENSEMBLE",
    "OUTPUT_INDEX": "08",
    "ESTIMATORS": (
        LinearRegression(normalize=True),
        Ridge(alpha=1.0, normalize=True),
        SVR(kernel="rbf", degree=3, C=1.0, epsilon=0.1),
        DecisionTreeRegressor(splitter="best"),
        GradientBoostingRegressor(
            loss="ls",
            learning_rate=0.1,
            n_estimators=100,
        ),
        AdaBoostRegressor(
            loss="linear",
            learning_rate=0.1,
            n_estimators=100,
        ),
        RandomForestRegressor(n_estimators=100),
        MLPRegressor(
            hidden_layer_sizes=(
                10,
                10,
            )
        ),
    ),
}
rng = check_random_state(0)
dependent = SETTINGS["DEPENDENT"]
features = ["{0}_PRED_L{1}".format(dependent, str(i + 1)) for i in range(6)]
vehicles = (
    (item[0] for item in EXPERIMENTS)
    if SETTINGS["SENSOR"] == "Veepeak"
    else (item[0] for item in EXPERIMENTS if not item[1])
)
for vehicle in vehicles:
    df_input = load_data_from_Excel(vehicle, SETTINGS)
    df_input.dropna(inplace=True)
    X, y = df_input[features], df_input[dependent]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=rng
    )
    df_output = pd.DataFrame()
    df_output[dependent] = y_test
    df_output[features] = X_test
    for estimator in SETTINGS["ESTIMATORS"]:
        ensemble = estimator.fit(X_train, y_train).predict(X_test)
        df_output["{0}_PRED_{1}".format(dependent, estimator)] = ensemble
        score = round(rmse(y_test, ensemble), 3)
        row = (
            dependent,
            vehicle,
            estimator,
            score,
        )
        log_attributes_and_score(
            row, "Paper III - Ensemble - Sophisticated Ensemble Scores.csv"
        )
        print("{0}\nSophisticated Ensemble {1}\nscore {2}.".format(vehicle, estimator, score))
    save_data_to_Excel(df_output, vehicle, SETTINGS)

# %%
