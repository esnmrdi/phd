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

#%% [markdown]
# ### Display training progress
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
            print("{0} out of {1} epochs completed.".format(epoch, self.n_epochs))

    def on_train_end(self, logs):
        print("Training finished.")


#%% [markdown]
# ### Load sample data from Excel to a pandas dataframe
def load_sample_from_Excel(vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + device
        + "/"
        + vehicle
        + "/Processed/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["input_type"], settings["input_index"]
    )
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
    df = pd.DataFrame()
    for _, sheet in sheets_dict.items():
        df = df.append(sheet)
    df.reset_index(inplace=True, drop=True)
    if df.shape[0] > settings["max_sample_size"]:
        sample_size = settings["max_sample_size"]
        df = df.sample(sample_size)
    else:
        sample_size = df.shape[0]
    return df, sample_size


#%% [markdown]
# ### General settings
plt.style.use("bmh")
pd.options.mode.chained_assignment = None
EXPERIMENTS = [
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
]

#%% [markdown]
# ### ANN settings
SETTINGS = {
    "dependents": ["RPM", "FCR_LH", "PM_MGM3", "NO2_PPM", "NO_PPM", "CO2_PPM"],
    "predicted_dependents": ["RPM_PRED", "FCR_LH_PRED", "PM_MGM3_PRED", "NO2_PPM_PRED", "NO_PPM_PRED", "CO2_PPM_PRED"],
    "features": ["SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG"],
    # "features": ["SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG", "RPM_PRED"]
    "lag_order": 1,
    "max_sample_size": 5400,
    "test_split_ratio": 0.20,
    "n_epochs": 200,
    "drop_prop": 0.1,
    "labels": {
        "FCR_LH": "Observed Fuel Consumption Rate (L/H)",
        "FCR_LH_PRED": "Predicted Fuel Consumption Rate (L/H)",
        "RPM": "Observed Engine Speed (rev/min)",
        "RPM_PRED": "Predicted Engine Speed (rev/min)",
        "SPD_KH": "Speed (Km/h)",
        "ACC_MS2": "Acceleration (m/s2)",
        "NO_OUTLIER_GRADE_DEG": "Road Grade (Deg)",
    },
    "learning_rate": 0.001,
    "metrics": [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "cosine_proximity",
    ],
    "input_type": "ANN",
    "output_type": "ANN",
    "input_index": "17",
    "output_index": "19",
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
