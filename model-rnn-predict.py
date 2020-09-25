# %%
# RNN (Simple, GRU, LSTM) for Vehicle-based Fuel Consumption and Emissions Rate Estimation
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
# Define a class for tracking the progress of model training
import keras
import pandas as pd
import numpy as np
import time
import resource
import h5py
from keras import Sequential, backend
from keras.layers import Dense, GRU, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# %%
# Callback to track the memory usage after training each model
class MemoryCallback(keras.callbacks.Callback):
    def on_train_end(self, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# %%
# Load settings of best models
def load_best_models_settings(sheet):
    directory = "../../Google Drive/Academia/PhD Thesis/Charts, Tables, Forms, Flowcharts, Spreadsheets, Figures/"
    input_file = "Paper III - RNN Ranking.xlsx"
    input_path = directory + input_file
    lst = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return lst


# %%
# Load sample data from Excel to a pandas dataframe
def load_from_Excel(sensor_type, vehicle, sheet, settings):
    input_type = settings["INPUT_TYPE"]
    input_index = settings["INPUT_INDEX"]
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + sensor_type
        + "/"
        + vehicle
        + "/Processed/"
        + input_type
        + "/"
    )
    input_file = "{0} - {1} - {2}.xlsx".format(vehicle, input_type, input_index)
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return df


# %%
# Save the predicted field back to Excel file
def save_to_excel(df, sensor_type, vehicle, settings):
    output_type = settings["OUTPUT_TYPE"]
    output_index = settings["OUTPUT_INDEX"]
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + sensor_type
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
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None

# %%
# Save fitted model to h5 file
def save_to_h5(model, vehicle, dependent, lookback, settings):
    output_type = settings["OUTPUT_TYPE"]
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/"
        + output_type
        + "/"
    )
    output_file = "{0} - {1} - {2} - L{3}.h5".format(vehicle, output_type, dependent, lookback)
    output_path = directory + output_file
    model.save(output_path)

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
def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.array(y_pred) - np.array(y_true))))


# %%
# Define RNN model
def define_model(rnn_type, lookback, n_stacks, settings):
    n_features = len(settings["FEATURES"])
    batch_size = settings["BATCH_SIZE"]
    drop_prop = settings["DROP_PROP"]
    n_units = settings["N_UNITS"]
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
def train_rnn(df, model, vehicle, dependent, lookback, settings):
    start_timestamp = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features = settings["FEATURES"]
    test_split_ratio = settings["TEST_SPLIT_RATIO"]
    batch_size = settings["BATCH_SIZE"]
    n_epochs = settings["N_EPOCHS"]
    loss = settings["LOSS"]
    predicted = dependent + "_PRED_L{0}".format(lookback)
    df_temp, scaler_X, scaler_y = scale(df, features, dependent)
    X, y = generate_input(df_temp, features, dependent, lookback)
    train_X, train_y, test_X, test_y = split(X, y, test_split_ratio, batch_size)
    model.compile(loss=loss, optimizer=settings["OPTIMIZER"])
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
    save_to_h5(model, vehicle, dependent, lookback, settings)
    trim_size = len(X) % batch_size
    trim_index = len(X) - trim_size
    X = X[:trim_index]
    y_predict = model.predict(X, batch_size=batch_size)
    y_predict = scaler_y.inverse_transform(y_predict)
    y_predict = np.append(y_predict, np.repeat(np.nan, trim_size + lookback))
    df[predicted] = y_predict
    del model
    backend.clear_session()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = round(time.time() - start_timestamp, 1)
    return df


# %%
# Model execution and input/output settings
pd.options.mode.chained_assignment = None
SETTINGS = {
    "FEATURES": ["SPD_KH", "ACC_MS2", "ALT_M"],
    "N_UNITS": 100,
    "N_EPOCHS": 200,
    "TEST_SPLIT_RATIO": 0.3,
    "DROP_PROP": 0.5,
    "BATCH_SIZE": 64,
    "LOSS": rmse,
    "OPTIMIZER": "adam",
    "INPUT_INDEX": "04",
    "INPUT_TYPE": "NONE",
    "OUTPUT_INDEX": "05",
    "OUTPUT_TYPE": "RNN",
}

# %%
# Batch execution on trips of all included vehicles
# loop through PEMS-included experiments only or obd-only data (depending on desired output)
best_model_settings = load_best_models_settings("Best Model Settings (1-6)")
old_vehicle = ""
for index, model_setting in best_model_settings.iterrows():
    vehicle = model_setting["VEHICLE"]
    dependent = model_setting["DEPENDENT"]
    rnn_type = model_setting["RNN_TYPE"]
    lookback = model_setting["LOOKBACK"]
    n_stacks = model_setting["N_STACKS"]
    sensor_type = "Veepeak" if dependent == "FCR_LH" else "3DATX parSYNC Plus"
    if vehicle != old_vehicle: 
        if old_vehicle != "":
            save_to_excel(df, old_sensor_type, old_vehicle, SETTINGS)
        df = load_from_Excel(sensor_type, vehicle, "Sheet1", SETTINGS)
    print(vehicle, dependent, lookback)
    model = define_model(rnn_type, lookback, n_stacks, SETTINGS)
    df = train_rnn(df, model, vehicle, dependent, lookback, SETTINGS)
    old_vehicle, old_sensor_type = vehicle, sensor_type
    del model
# %%
