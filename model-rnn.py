# %%
# Regular Recurrent Neural Network (RNN) for Energy Consumption and Emissions Rate Estimation
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required libraries and define classes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class ReportProgress(tf.keras.callbacks.Callback):
    def __init__(self, df, test_split_ratio, n_epochs):
        self.df = df
        self.test_split_ratio = test_split_ratio
        self.n_epochs = n_epochs

    def on_train_begin(self, logs):
        n_examples = len(self.df)
        n_train = int((1 - self.test_split_ratio) * n_examples)
        print(
            "Training started on {0} out of {1} available examples.".format(
                n_train, n_examples
            )
        )

    def on_epoch_end(self, epoch, logs):
        if epoch % 20 == 0 and epoch != 0 and epoch != self.n_epochs:
            print("{0} out of {1} epochs completed.".format(
                epoch, self.n_epochs))

    def on_train_end(self, logs):
        print("Training finished.")


# %%
# Load sample data from Excel to a pandas dataframe
def load_from_Excel(vehicle, sheet, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["INPUT_TYPE"], settings["INPUT_INDEX"]
    )
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return df


# %%
# Scale the features
def scale(df, features, dependent):
    df_temp = df.copy()
    all_features = features + dependent
    scaler = preprocessing.StandardScaler().fit(df_temp[all_features])
    df_temp[all_features] = scaler.transform(df_temp[all_features])
    return df_temp, scaler


# %%
# Reverse-scale the features
def reverse_scale(df, scaler):
    df_temp = df.copy()
    df_temp = np.sqrt(scaler.var_[-1]) * df_temp + scaler.mean_[-1]
    return df_temp


# %%
# Generate time-series input for the desired lookback order
def generate(df, features, dependent, lookback):
    df_X = [tuple(x) for x in df[features].to_numpy()]
    df_y = [y for y in df[dependent].to_numpy()]
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df_X[i - lookback: i + 1])
        y.append(df_y[i - lookback: i + 1])
    X, y = np.array(X), np.array(y)
    return X, y


# %%
# Split data to train and test sets and reshape arrays for modeling
def splitAndReshape(X, y, test_split_ratio, lookback):
    test_size = int(len(X) * test_split_ratio)
    split_index = len(X) - test_size
    n_features = len(X[0, 0])
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    X_train = X_train.reshape(X_train.shape(0), lookback, n_features)
    X_test = X_test.reshape(X_test.shape(0), lookback, n_features)
    return X_train, y_train, X_test, y_test


# %%
# Experiments to include in modeling
EXPERIMENTS = (  # the boolean points to whether the experiment type is obd_only or pems_included.
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
EXPERIMENTS = (("009 Renault Logan 2014 (1.6L Manual)", True),)

# %%
# Model execution and input/output settings
pd.options.mode.chained_assignment = None
plt.style.use("bmh")
SETTINGS = {
    "DEPENDENT_OBD": "FCR_LH",
    "PREDICTED_OBD": "FCR_LH_PRED",
    "DEPENDENT_PEMS": "CO2_KGS",
    "PREDICTED_PEMS": "CO2_KGS_PRED",
    "FEATURES": ("SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG"),
    "INPUT_TYPE": "NONE",
    "INPUT_INDEX": "04",
    "OUTPUT_TYPE": "RNN",
    "OUTPUT_INDEX": "05",
    "LOOKBACK": 5,
    "TEST_SPLIT_RATIO": 0.3,
}

# %%
# Batch execution on trips of all included vehicles
for index, vehicle in enumerate(EXPERIMENTS):
    experiment_type = "obd_only" if vehicle[1] == True else "pems_included"
    df = load_from_Excel(vehicle[0], "Sheet1", SETTINGS)
    features = SETTINGS["FEATURES"]
    dependent = (
        SETTINGS["DEPENDENT_OBD"]
        if experiment_type == "obd_only"
        else SETTINGS["DEPENDENT_PEMS"]
    )
    df, scaler = scale(df, features, dependent)
    train_set, test_set = split(df, SETTINGS["TEST_SPLIT_RATIO"])
    lookback_train_set = generate(train_set, SETTINGS["LAG_ORDER"])

    figure(num=1, figsize=(8, 4), dpi=150, facecolor="w", edgecolor="k")
    plt.plot(
        df[SETTINGS[dependent]][:300], color="blue",
    )
    plt.title("Fuel Consumption Rate Variations")
    plt.xlabel("Time")
    plt.ylabel("FCR (L/H)")
    plt.show()


# %%
