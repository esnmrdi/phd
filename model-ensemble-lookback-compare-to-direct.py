# %%
# Direct application of meta-regressors on data to compare with metamodels' output
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

# %%
# Load data from Excel to a pandas dataframe
def load_data_from_excel(sensor, vehicle):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + sensor
        + "/"
        + vehicle
        + "/Processed/ENSEMBLE/"
    )
    input_file = f"{vehicle} - ENSEMBLE - 06.xlsx"
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name="Sheet1", header=0, engine="openpyxl")
    return df


# %%
# Log model settings and corresponding scores to a file (one by one)
def log_model_settings_and_score(row):
    directory = "../../../Google Drive/Academia/PhD Thesis/Charts, Tables, Forms, Flowcharts, Spreadsheets, Figures/"
    output_file = "paper-iii-ensemble-lookback-compare-to-direct.csv"
    output_path = directory + output_file
    with open(output_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return None

# %%
# Save back the predictions to Excel
def save_data_to_excel(df, vehicle, dependent, estimator):
    directory = "../../../Google Drive/Academia/PhD Thesis/Modeling Outputs/DIRECT/"
    output_file = f"{vehicle} - {dependent} - {estimator}.xlsx"
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print(f"{vehicle}, {dependent} -> Output is successfully saved to Excel.")
    return None


# %%
# Definition of the rmse function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.array(y_pred) - np.array(y_true))))


# %%
# Definition of the r2 function
def r2(y_true, y_pred):
    return 1 - (np.sum(np.square(np.array(y_pred) - np.array(y_true)))) / (
        np.sum(np.square(np.array(y_true) - np.mean(np.array(y_true))))
    )


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

# %%
# General settings
pd.options.mode.chained_assignment = None
SETTINGS = {
    "SENSOR_DEPENDENT": {
        "Veepeak": ("FCR_LH",),
        "3DATX parSYNC Plus": ("CO2_KGS", "NO_KGS", "NO2_KGS", "PM_KGS"),
    },
    "ESTIMATORS": {
        "FCR_LH": RandomForestRegressor(n_estimators=100),
        "CO2_KGS": RandomForestRegressor(n_estimators=100),
        "NO_KGS": LinearRegression(normalize=True),
        "NO2_KGS": LinearRegression(normalize=True),
        "PM_KGS": LinearRegression(normalize=True),
    },
    "FEATURES": ["SPD_KH", "ACC_MS2", "ALT_M"],
}

# %%
# Training the direct models
rng = check_random_state(0)
features = SETTINGS["FEATURES"]
estimators = SETTINGS["ESTIMATORS"]
for sensor, dependents in SETTINGS["SENSOR_DEPENDENT"].items():
    for dependent in dependents:
        vehicles = (
            (item[0] for item in EXPERIMENTS)
            if sensor == "Veepeak"
            else (item[0] for item in EXPERIMENTS if not item[1])
        )
        for vehicle in vehicles:
            # print(vehicle, dependent)
            df_input = load_data_from_excel(sensor, vehicle)
            df_input.dropna(inplace=True)
            X, y, dt = df_input[features], df_input[dependent], df_input["DATETIME"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=rng
            )
            df_output = pd.DataFrame()
            df_output["DATETIME"] = dt
            df_output[features] = X
            df_output[dependent] = y
            estimator = estimators[dependent]
            model = estimator.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            df_output[f"{dependent}_PRED_{estimator}_DIRECT"] = model.predict(X)
            save_data_to_excel(df_output, vehicle, dependent, estimator)
            rmse_score = rmse(y_test, y_pred)
            r2_score = r2(y_test, y_pred)
            row = [dependent, vehicle, estimator, rmse_score, r2_score]
            log_model_settings_and_score(row)
            print(
                f"Dependent: {dependent}\nVehicle: {vehicle}\nEstimator: {estimator}\nRMSE: {rmse_score}\nR2: {r2_score}"
            )
# %%