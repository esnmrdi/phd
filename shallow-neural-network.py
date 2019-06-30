#%% [markdown]
# ## Shallow Neural Network for FCR Prediction
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import seaborn as sns


#%% [markdown]
# ### Loading data from Excel to a pandas dataframe
def load_from_Excel(vehicle):
    directory = "./Field Experiments/Veepeak/" + vehicle + "/Processed/"
    input_file = vehicle + " + Grade.xlsx"
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
    df = pd.DataFrame()
    for _, sheet in sheets_dict.items():
        df = df.append(sheet)
    df.reset_index(inplace=True, drop=True)
    return df


#%% [markdown]
# ### Adding lagged features to the dataframe
def add_lagged_features(df, lagged_features, lag_order):
    for feature in lagged_features:
        for i in range(lag_order):
            df[feature + "_L" + i] = df[feature].shift(i + 1)
    df.dropna(inplace=True)
    return df


#%% [markdown]
# ### Feature scaling
def scale(df, feature_names):
    scaler = preprocessing.StandardScaler().fit(df[feature_names])
    df[feature_names] = scaler.transform(df[feature_names])
    return df, scaler


#%% [markdown]
# ### General settings
EXPERIMENTS = [
    "009 Renault Logan 2014 (1.6L Manual)",
    "010 JAC J5 2015 (1.8L Auto)",
    "011 JAC S5 2017 (2.0L TC Auto)",
    "012 IKCO Dena 2016 (1.65L Manual)",
    "013 Geely Emgrand7 2014 (1.8L Auto)",
    "014 Kia Cerato 2016 (2.0L Auto)",
    "015 VW Jetta 2016 (1.4L TC Auto)",
    "016 Hyundai Sonata Sport 2019 (2.4L Auto)",
    "017 Chevrolet Trax 2019 (1.4L TC Auto)",
    "018 Hyundai Azera 2006 (3.8L Auto)",
    "019 Hyundai Elantra GT 2019 (2.0L Auto)",
    "020 Honda Civic 2014 (1.8L Auto)",
    "021 Chevrolet N300 2014 (1.2L Manual)",
    "022 Chevrolet Spark GT 2012 (1.2L Manual)",
    "023 Mazda 2 2012 (1.4L Auto)",
    "024 Renault Logan 2010 (1.4 L Manual)",
    "025 Chevrolet Captiva 2010 (2.4L Auto)",
    "026 Nissan Versa 2013 (1.6L Auto)",
    "027 Chevrolet Cruze 2011 (1.8L Manual)",
    "028 Nissan Sentra 2019 (1.8L Auto)",
    "029 Ford Escape 2006 (3.0L Auto)",
]
VEHICLE = "012 IKCO Dena 2016 (1.65L Manual)"

#%% [markdown]
# ### NN settings
DEPENDENT = "FCR_LH"
FEATURES = ["SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG"]
LAGGED_FEATURES = ["SPD_KH", "NO_OUTLIER_GRADE_DEG"]
LAG_ORDER = 1
VALIDATION_SPLIT_RATIO = 0.20
EPOCHS = 100

#%% [markdown]
# ### Load data from Excel, Perform feature scaling and discretization of dependent variable, and Split the data into train and test segments
df = load_from_Excel(VEHICLE)
df_scaled, scaler = scale(df, FEATURES + LAGGED_FEATURES + [DEPENDENT])

#%% [markdown]
# ### Define NN model structure
n_features = len(FEATURES + LAGGED_FEATURES)
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(64, input_shape=[n_features], activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
model.summary()

#%% [markdown]
# ### Compiling the NN model
model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.SGD(lr=0.001),
    metrics=["mean_absolute_error", "mean_squared_error"],
)

#%% [markdown]
# ### Display training progress by printing a single dot for each completed epoch
class ReportProgress(tf.keras.callbacks.Callback):
    def __init__(self, df):
        self.df = df

    def on_train_begin(self, logs):
        n_examples = int((1 - VALIDATION_SPLIT_RATIO) * len(self.df))
        print("Training started on {} examples.".format(n_examples))

    def on_epoch_end(self, epoch, logs):
        if epoch % 20 == 0 and epoch != 0 and epoch != EPOCHS:
            print("{0} out of {1} epochs completed.".format(epoch, EPOCHS))

    def on_train_end(self, logs):
        print("Training finished.")


#%% [markdown]
# ### Train the model
history = model.fit(
    df_scaled[FEATURES + LAGGED_FEATURES],
    df_scaled[DEPENDENT],
    batch_size=16,
    epochs=EPOCHS,
    shuffle=False,
    validation_split=VALIDATION_SPLIT_RATIO,
    verbose=0,
    callbacks=[ReportProgress(df)],
)

#%% [markdown]
# ### Evaluating the training and validation error
hist = pd.DataFrame(history.history)
epochs = history.epoch
training_mae = hist["mean_absolute_error"]
training_mse = hist["mean_squared_error"]
validation_mae = hist["val_mean_absolute_error"]
validation_mse = hist["val_mean_squared_error"]
p = figure(
    plot_width=720,
    plot_height=360,
    title="Training vs Validation Mean Absolute Error",
    toolbar_location="above",
)
p.line(epochs, training_mae, line_width=2, color="red", legend="Training")
p.line(epochs, validation_mae, line_width=2, color="blue", legend="Validation")
p.xaxis.axis_label = "# of Epochs"
p.yaxis.axis_label = "MAE"
show(p)
p = figure(
    plot_width=720,
    plot_height=360,
    title="Training vs Validation Mean Squared Error",
    toolbar_location="above",
)
p.line(epochs, training_mse, line_width=2, color="red", legend="Training")
p.line(epochs, validation_mse, line_width=2, color="blue", legend="Validation")
p.xaxis.axis_label = "# of Epochs"
p.yaxis.axis_label = "MSE"
show(p)

#%%
