# %%
# RNN (Simple, GRU, LSTM) for Vehicle-based Fuel Consumption and Emissions Rate Estimation
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
# Define a class for tracking the progress of model training
import tensorflow
import keras
from keras import backend
from keras.layers import Dense, SimpleRNN, GRU, LSTM
from keras import Sequential, utils
from tensorflow.keras.callbacks import EarlyStopping  
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import csv
from datetime import datetime
import time
import resource

# %%
# Callback to track the memory usage after training each model
class MemoryCallback(tensorflow.keras.callbacks.Callback):
    def on_train_end(self, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# %%
# Load sample data from Excel to a pandas dataframe
def load_from_Excel(vehicle, sheet, settings):
    input_type = settings["INPUT_TYPE"]
    input_index = settings["INPUT_INDEX"]
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/3DATX parSYNC Plus/"
        + vehicle
        + "/Processed/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(input_type, input_index)
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return df


# %%
# Log model attributes and corresponding scores to a file (one by one)
def log_model_score(row):
    directory = "../../Google Drive/Academia/PhD Thesis/Charts, Tables, Forms, Flowcharts, Spreadsheets, Figures/"
    output_file = "Paper III - RNN Results.csv"
    output_path = directory + output_file
    with open(output_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return None


# %%
# Scale the data
def scale(df, features, dependent):
    df_temp = df.copy()
    scaler_X = StandardScaler().fit(df_temp[features])
    scaler_y = StandardScaler().fit(df_temp[[dependent]])
    df_temp[features] = scaler_X.transform(df_temp[features])
    df_temp[[dependent]] = scaler_y.transform(df_temp[[dependent]])
    return df_temp, scaler_X, scaler_y


# %%
# Generate time-series input for the desired lookback order
def generate_input(df, features, dependent, lookback):
    dataset = df[features + [dependent]].to_numpy()
    dim = len(features)
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback : i + 1, :dim])
        y.append(dataset[i, dim])
    X, y = np.array(X), np.array(y)
    return X, y


# %%
# Split data into train and test parts
def split(X, y, test_split_ratio, batch_size):
    n_examples = len(X)
    split_index = int(n_examples * (1 - test_split_ratio))
    train_X, test_X = X[:split_index], X[split_index:]
    train_y, test_y = y[:split_index], y[split_index:]
    trim_train = len(train_X) - len(train_X) % batch_size
    trim_test = len(test_X) - len(test_X) % batch_size
    train_X, test_X = train_X[:trim_train], test_X[:trim_test]
    train_y, test_y = train_y[:trim_train], test_y[:trim_test]
    return train_X, train_y, test_X, test_y


# %%
# Definition of the custom loss function
def root_mean_squared_error(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))


# %%
# Define RNN model
def define_model(rnn_type, lookback, n_stacks, n_units, settings):
    n_features = len(settings["FEATURES"])
    batch_size = settings["BATCH_SIZE"]
    drop_prop = settings["DROP_PROP"]
    common_args = (
        "n_units, dropout=drop_prop, recurrent_dropout=drop_prop, stateful=True"
    )
    model = Sequential()  # Default activation function is tanh
    if n_stacks == 1:
        model.add(
            eval(
                rnn_type
                + "("
                + common_args
                + ", batch_input_shape=(batch_size, lookback + 1, n_features)"
                + ")"
            )
        )
    elif n_stacks == 2:
        model.add(
            eval(
                rnn_type
                + "("
                + common_args
                + ", batch_input_shape=(batch_size, lookback + 1, n_features), return_sequences=True"
                + ")"
            )
        )
        model.add(eval(rnn_type + "(" + common_args + ")"))
    else:
        model.add(
            eval(
                rnn_type
                + "("
                + common_args
                + ", batch_input_shape=(batch_size, lookback + 1, n_features), return_sequences=True"
                + ")"
            )
        )
        for _ in range(n_stacks - 2):
            model.add(
                eval(rnn_type + "(" + common_args + ", return_sequences=True" + ")")
            )
        model.add(eval(rnn_type + "(" + common_args + ")"))
    model.add(Dense(1, activation="linear"))
    return model


# %%
# Train the RNN model by testing alternative architectures (different lookback orders, hidden units, and stacks)
def train_rnn(vehicle, dependent, optimizer, lookback, model, settings):
    backend.clear_session()
    start_timestamp = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features = settings["FEATURES"]
    test_split_ratio = settings["TEST_SPLIT_RATIO"]
    batch_size = settings["BATCH_SIZE"]
    n_epochs = settings["N_EPOCHS"]
    loss = settings["LOSS"]
    predicted = dependent + "_PRED"
    df = load_from_Excel(vehicle, "Sheet1", SETTINGS)
    df, scaler_X, scaler_y = scale(df, features, dependent)
    X, y = generate_input(df, features, dependent, lookback)
    train_X, train_y, test_X, test_y = split(X, y, test_split_ratio, batch_size)
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(
        train_X,
        train_y,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=(test_X, test_y),
        verbose=0,
        callbacks=[
            MemoryCallback(),
            EarlyStopping(
                monitor="val_loss", min_delta=0, patience=10, verbose=0, mode="min"
            ),
        ],
    )
#     train_y_predict = model.predict(train_X, batch_size=batch_size)
#     test_y_predict = model.predict(test_X, batch_size=batch_size)
    train_score = model.evaluate(train_X, train_y, batch_size=batch_size)
    test_score = model.evaluate(test_X, test_y, batch_size=batch_size)
    del model
    backend.clear_session()
#     train_y, train_y_predict = (
#         scaler_y.inverse_transform(train_y),
#         scaler_y.inverse_transform(train_y_predict),
#     )
#     test_y, test_y_predict = (
#         scaler_y.inverse_transform(test_y),
#         scaler_y.inverse_transform(test_y_predict),
#     )
#     train_r_squared = r2_score(train_y, train_y_predict)
#     test_r_squared = r2_score(test_y, test_y_predict)
#     r_squared = {"train": train_r_squared, "test": test_r_squared}
#     rmse = {
#         "train": history.history["loss"][-1],
#         "test": history.history["val_loss"][-1],
#     }
    rmse = {
        "train": train_score,
        "test": test_score,
    }
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = round(time.time() - start_timestamp, 1)
    return rmse, start_datetime, end_datetime, elapsed_time


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
# Model execution and input/output settings
pd.options.mode.chained_assignment = None
plt.style.use("bmh")
SETTINGS = {
    "FEATURES": ["SPD_KH", "ACC_MS2", "ALT_M"],
    "DEPENDENTS": ["FCR_LH"],
    "RNN_TYPES": ["GRU", "LSTM"],
    "LOOKBACK": range(1, 11),
    "N_UNITS": [100],
    "N_STACKS": range(1, 4),
    "N_EPOCHS": 200,
    "TEST_SPLIT_RATIO": 0.3,
    "DROP_PROP": 0.5,
    "BATCH_SIZE": 64,
    "LOSS": root_mean_squared_error,
    "OPTIMIZER": "adam",
    "INPUT_TYPE": "NONE",
    "INPUT_INDEX": "04",
    "OUTPUT_INDEX": "05",
}

# %%
# Batch execution on trips of all included vehicles
# loop through PEMS-included experiments only or obd-only data (depending on desired output)
features = SETTINGS["FEATURES"]
dependents = SETTINGS["DEPENDENTS"]
rnn_types = SETTINGS["RNN_TYPES"]
optimizer = SETTINGS["OPTIMIZER"]
drop_prop = SETTINGS["DROP_PROP"]
lookback_range = SETTINGS["LOOKBACK"]
n_stacks_range = SETTINGS["N_STACKS"]
n_units_range = SETTINGS["N_UNITS"]
vehicles = (item[0] for item in EXPERIMENTS)
for vehicle in vehicles:
    for dependent in dependents:
        for rnn_type in rnn_types:
            for lookback in lookback_range:
                for n_stacks in n_stacks_range:
                    for n_units in n_units_range:
                        print(vehicle, dependent, rnn_type, lookback, n_stacks, n_units)
                        model = define_model(
                            rnn_type, lookback, n_stacks, n_units, SETTINGS
                        )
                        rmse, start_datetime, end_datetime, elapsed_time = train_rnn(
                            vehicle, dependent, optimizer, lookback, model, SETTINGS
                        )
                        del model
                        row = (
                            vehicle,
                            dependent,
                            rnn_type,
                            optimizer,
                            drop_prop,
                            lookback,
                            n_stacks,
                            n_units,
#                             round(r_squared["train"], 2),
#                             round(r_squared["test"], 2),
                            round(rmse["train"], 3),
                            round(rmse["test"], 3),
                            start_datetime,
                            end_datetime,
                            elapsed_time
                        )
                        log_model_score(row)
# %%
