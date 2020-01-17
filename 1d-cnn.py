#%% [markdown]
# ## 1D Convolutional Neural Network for Fuel Consumption and Emissions Rate Estimation
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import preprocessing

#%% [markdown]
# ### Load data from Excel to a pandas dataframe
def load_from_Excel(vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/3DATX parSYNC Plus/"
        + vehicle
        + "/Processed/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["INPUT_TYPE"], settings["INPUT_INDEX"]
    )
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
    df = pd.DataFrame()
    for _, sheet in sheets_dict.items():
        df = df.append(sheet)
    return df


#%% [markdown]
# ### Scale the features
def scale(df, settings):
    df_temp = df.copy()
    variables = settings["FEATURES"].append(settings["DEPENDENT"])
    scaler = preprocessing.StandardScaler().fit(df_temp[variables])
    df_temp[variables] = scaler.transform(df_temp[variables])
    return df_temp, scaler


#%% [markdown]
# ### Reverse-scale the features
def reverse_scale(df, scaler):
    df_temp = df.copy()
    df_temp = np.sqrt(scaler.var_[-1]) * df_temp + scaler.mean_[-1]
    return df_temp


#%% [markdown]
# ### Define the 1-D CNN model (https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting)
def define_model(features, settings):
    model = tf.keras.Sequential()
    model.add(
        # no stride or padding is considered
        tf.keras.layers.Conv1D(
            filters=settings["N_FILTERS"],
            kernel_size=settings["KERNEL_SIZE"],
            activation="relu",
            input_shape=(settings["N_STEPS"], len(features)),
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=settings["POOL_SIZE"]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(settings["DENSE_LAYER_SIZE"], activation="relu"))
    model.add(tf.keras.layers.Dropout(settings["DROP_PROP"]))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    return model


#%% [markdown]
# ### Save the predicted field back to Excel file
def save_back_to_Excel(df, vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/3DATX parSYNC Plus/"
        + vehicle
        + "/Processed/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["OUTPUT_TYPE"], settings["OUTPUT_INDEX"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print("Data is saved to Excel successfully!")
    return None


#%% [markdown]
# ### General settings
plt.style.use("bmh")
pd.options.mode.chained_assignment = None
EXPERIMENTS = (
    "015 VW Jetta 2016 (1.4L TC Auto)",
    "020 Honda Civic 2014 (1.8L Auto)",
    "029 Ford Escape 2006 (3.0L Auto)",
    "030 Ford Focus 2012 (2.0L Auto)",
    "031 Mazda 3 2016 (2.0L Auto)",
    "032 Toyota RAV4 2016 (2.5L Auto)",
    "033 Toyota Corolla 2019 (1.8L Auto)",
    "034 Toyota Yaris 2015 (1.5L Auto)",
    "035 Kia Rio 2013 (1.6L Auto)",
    "036 Jeep Patriot 2010 (2.4L Auto)",
    "037 Chevrolet Malibu 2019 (1.5L TC Auto)",
    "038 Kia Optima 2012 (2.4L Auto)",
    "039 Honda Fit 2009 (1.5L Auto)",
    "040 Mazda 6 2009 (2.5L Auto)",
    "041 Nissan Micra 2019 (1.6L Auto)",
    "042 Nissan Rouge 2020 (2.5L Auto)",
    "043 Mazda CX-3 2019 (2.0L Auto)",
)
EXPERIMENTS = ("015 VW Jetta 2016 (1.4L TC Auto)",)

#%% [markdown]
# ### ANN settings
SETTINGS = {
    "DEPENDENT": "FCR_LH",  # other dependents to work on are RPM, PM_mGM3, NO2_mGM3, NO_mGM3, and CO2_mGM3
    "FEATURES": ("SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG"),
    "CROSS_VALIDATION_SPLITS": 5,
    "N_EPOCHS": 200,
    "DROP_PROP": 0.1,
    "LABELS": {
        "RPM": "Observed Engine Speed (rev/min)",
        "RPM_PRED": "Predicted Engine Speed (rev/min)",
        "FCR_LH": "Observed Fuel Consumption Rate (l/hr)",
        "FCR_LH_PRED": "Predicted Fuel Consumption Rate (l/hr)",
        "PM_mGM3": "Observed Particulate Matters Concentration (mg/m3)",
        "PM_mGM3_PRED": "Predicted Particulate Matters Concentration (mg/m3)",
        "CO2_mGM3": "Observed Carbon Dioxide Concentration (mg/m3)",
        "CO2_mGM3_PRED": "Predicted Carbon Dioxide Concentration (mg/m3)",
        "NO_mGM3": "Observed Nitrogen Oxide Concentration (mg/m3)",
        "NO_mGM3_PRED": "Predicted Nitrogen Oxide Concentration (mg/m3)",
        "NO2_mGM3": "Observed Nitrogen Dioxide Concentration (mg/m3)",
        "NO2_mGM3_PRED": "Predicted Nitrogen Dioxide Concentration (mg/m3)",
        "SPD_KH": "Speed (Km/h)",
        "ACC_MS2": "Acceleration (m/s2)",
        "NO_OUTLIER_GRADE_DEG": "Road Grade (Deg)",
    },
    "LEARNING_RATE": 0.001,
    "METRIC": "mean_squared_error",
    "INPUT_TYPE": "NONE",
    "OUTPUT_TYPE": "CNN",
    "INPUT_INDEX": "03",
    "OUTPUT_INDEX": "04",
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data from Excel to a pandas dataframe
    df = load_from_Excel(vehicle, SETTINGS)
    # Scale the features
    df, scaler = scale(df, SETTINGS)
    # Save the predicted field back to Excel file
    save_back_to_Excel(df, vehicle, SETTINGS)
