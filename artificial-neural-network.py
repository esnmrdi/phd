#%% [markdown]
# ## Artificial Neural Network for RPM and FCR Prediction
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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
    directory = "../Field Experiments/Veepeak/" + vehicle + "/Processed/"
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
# ### Add lagged features to the dataframe
def add_lagged_features(df, settings):
    df_temp = df.copy()
    total_features = settings["features"]
    for feature in settings["lagged_features"]:
        for i in range(settings["lag_order"]):
            new_feature = feature + "_L" + str(i + 1)
            total_features.append(new_feature)
            df_temp[new_feature] = df_temp[feature].shift(i + 1)
    df_temp.dropna(inplace=True)
    return df_temp, total_features


#%% [markdown]
# ### Scale the features
def scale(df, total_features, settings):
    df_temp = df.copy()
    feature_names = total_features + [settings["dependent"]]
    scaler = preprocessing.StandardScaler().fit(df_temp[feature_names])
    df_temp[feature_names] = scaler.transform(df_temp[feature_names])
    return df_temp, scaler


#%% [markdown]
# ### Reverse-scale the features
def reverse_scale(df, scaler):
    df_temp = df.copy()
    df_temp = np.sqrt(scaler.var_[-1]) * df_temp + scaler.mean_[-1]
    return df_temp


#%% [markdown]
# ### Define alternative ANN models with different architectures
def define_models(total_features, settings):
    n_features = len(total_features)
    models = []
    for n_layers, n_neurons in settings["model_architectures"]:
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(
                n_neurons, input_shape=[n_features], activation="relu"
            )
        )
        model.add(tf.keras.layers.Dropout(settings["drop_prop"]))
        for _ in range(n_layers - 1):
            model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
            model.add(tf.keras.layers.Dropout(settings["drop_prop"]))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        models.append(model)
    return models


#%% [markdown]
# ### Report the train and test score
def calculate_score(model, train_set, test_set, total_features, settings):
    train_set[settings["predicted"]] = model.predict(train_set[total_features])
    test_set[settings["predicted"]] = model.predict(test_set[total_features])
    r2_train = r2_score(
        train_set[settings["dependent"]], train_set[settings["predicted"]]
    )
    r2_test = r2_score(test_set[settings["dependent"]], test_set[settings["predicted"]])
    score = {}
    score["train"], score["test"] = r2_train, r2_test
    return score


#%% [markdown]
# ### Tune the ANN model by testing alternative architectures (from shallow and wide to deep and narrow)
def tune_ann(df, total_features, scaler, settings):
    df_temp = df.copy()
    models = define_models(total_features, settings)
    train_set, test_set = train_test_split(
        df_temp, test_size=settings["test_split_ratio"], shuffle=True
    )
    x_train, y_train = train_set[total_features], train_set[settings["dependent"]]
    x_test, y_test = test_set[total_features], test_set[settings["dependent"]]
    histories = []
    scores = []
    for index, model in enumerate(models):
        model.compile(
            loss=settings["metrics"][0],
            optimizer=tf.keras.optimizers.RMSprop(lr=settings["learning_rate"]),
            metrics=settings["metrics"],
        )
        history = model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=settings["n_epochs"],
            validation_data=(x_test, y_test),
            shuffle=False,
            verbose=0,
            callbacks=[
                ReportProgress(
                    df_temp, settings["test_split_ratio"], settings["n_epochs"]
                )
            ],
        )
        predicted_temp = "{0}_({1},{2})".format(
            settings["predicted"],
            settings["model_architectures"][index][0],
            settings["model_architectures"][index][1],
        )
        df_temp[predicted_temp] = model.predict(df_temp[total_features])
        df_temp[predicted_temp] = reverse_scale(df_temp[predicted_temp], scaler)
        histories.append(history)
        score = calculate_score(model, train_set, test_set, total_features, settings)
        scores.append(score)
    df_temp[settings["dependent"]] = reverse_scale(
        df_temp[settings["dependent"]], scaler
    )
    return df_temp, scores, histories


#%% [markdown]
# ### Plot training history and save plot to file
def plot_training_results(vehicle, sample_size, scores, histories, settings):
    fig, axn = plt.subplots(
        len(settings["model_architectures"]),
        len(settings["metrics"]),
        figsize=(30, 5 * len(settings["model_architectures"])),
        constrained_layout=True,
    )
    for ax, metric in zip(axn[0], settings["metrics"]):
        ax.set_title(metric)
    fig.suptitle(
        "Experiment: {0}\nSample Size: {1}\n# of Epochs: {2}".format(
            vehicle, sample_size, settings["n_epochs"]
        ),
        fontsize=24,
    )
    for index, ax in enumerate(axn.flat):
        row = index // len(settings["metrics"])
        col = index % len(settings["metrics"])
        history = histories[row]
        hist = pd.DataFrame(history.history)
        epochs = history.epoch
        sns.lineplot(x=epochs, y=settings["metrics"][col], data=hist, ax=ax, label="Train")
        sns.lineplot(
            x=epochs, y="val_" + settings["metrics"][col], data=hist, ax=ax, label="Test"
        )
        ax.set(
            xlabel="# of Epochs",
            ylabel="Architecture: {0}\nTrain Score: {1} | Test Score: {2}".format(
                settings["model_architectures"][row],
                np.round(scores[row]["train"], 3),
                np.round(scores[row]["test"], 3),
            ),
        )
        ax.legend(loc="best")
    plt.show()
    fig.savefig(
        "../Modeling Outputs/{0}/{1} - {2}/{3} - Training Result.jpg".format(
            settings["output_type"],
            settings["output_index"],
            settings["model_structure"],
            vehicle,
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )


#%% [markdown]
# ### Plot predictions vs. ground-truth and save plot to file
def plot_accuracy(df, vehicle, sample_size, scores, settings):
    fig, axn = plt.subplots(
        len(settings["model_architectures"]),
        1,
        figsize=(5, 5 * len(settings["model_architectures"])),
        constrained_layout=True,
    )
    fig.suptitle(
        "Experiment: {0}\nSample Size: {1}\n# of Epochs: {2}".format(
            vehicle, sample_size, settings["n_epochs"]
        ),
        fontsize=18,
    )
    for index, ax in enumerate(axn.flat):
        predicted_temp = "{0}_({1},{2})".format(
            settings["predicted"],
            settings["model_architectures"][index][0],
            settings["model_architectures"][index][1],
        )
        sns.regplot(
            x=settings["dependent"],
            y=predicted_temp,
            data=df,
            fit_reg=True,
            ax=ax,
            scatter_kws={"color": "blue"},
            line_kws={"color": "red"},
        )
        ax.set(
            xlabel=settings["labels"][settings["dependent"]],
            ylabel=settings["labels"][settings["predicted"]],
        )
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_title(
            "Architecture: {0}\nTrain Score: {1} | Test Score: {2}".format(
                settings["model_architectures"][index],
                np.round(scores[index]["train"], 3),
                np.round(scores[index]["test"], 3),
            )
        )
    plt.show()
    fig.savefig(
        "../Modeling Outputs/{0}/{1} - {2}/{3} - Observed vs. Predicted.jpg".format(
            settings["output_type"],
            settings["output_index"],
            settings["model_structure"],
            vehicle,
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )


#%% [markdown]
# ### Save the predicted field back to Excel file
def save_back_to_Excel(df, vehicle, settings):
    directory = "../Field Experiments/Veepeak/" + vehicle + "/Processed/"
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["output_type"], settings["output_index"]
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
    "030 Ford Focus 2012 (2.0L Auto)",
    "031 Mazda 3 2016 (2.0L Auto)",
    "032 Toyota RAV4 2016 (2.5L Auto)",
]

#%% [markdown]
# ### ANN settings
SETTINGS = {
    "dependent": "FCR_LH",
    "predicted": "FCR_LH_PRED",
    "features": ["SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG"],
    "lagged_features": ["SPD_KH", "NO_OUTLIER_GRADE_DEG"],
    "lag_order": 0,
    "max_sample_size": 5400,
    "test_split_ratio": 0.20,
    "n_epochs": 100,
    "drop_prop": 0.1,
    "labels": {
        "FCR_LH": "Observed Fuel Consumption Rate (L/H)",
        "FCR_LH_PRED": "Predicted Fuel Consumption Rate (L/H)",
        "RPM": "Engine Speed (rev/min)",
        "RPM_PRED": "Predicted Engine Speed (rev/min)",
        "SPD_KH": "Speed (Km/h)",
        "ACC_MS2": "Acceleration (m/s2)",
        "NO_OUTLIER_GRADE_DEG": "Road Grade (Deg)",
    },
    "model_structure": "FCR ~ SPD + ACC + GRADE",
    "model_architectures": [(1, 128), (2, 64), (4, 32), (8, 16), (16, 8)],
    "learning_rate": 0.001,
    "metrics": [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "cosine_proximity",
    ],
    "input_type": "NONE",
    "output_type": "ANN",
    "input_index": "01",
    "output_index": "02",
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Add lagged features to the dataframe and sampling
    df, sample_size = load_sample_from_Excel(vehicle, SETTINGS)
    # Add lagged features to the dataframe
    df, total_features = add_lagged_features(df, SETTINGS)
    # Scale the features
    df, scaler = scale(df, total_features, SETTINGS)
    # Tune the ANN model by testing alternative architectures (from shallow and wide to deep and narrow)
    df, scores, histories = tune_ann(df, total_features, scaler, SETTINGS)
    # Plot training histories for all model architectures and save plots to file
    plot_training_results(vehicle, sample_size, scores, histories, SETTINGS)
    # Plot predictions vs. ground-truth and save plot to file
    plot_accuracy(df, vehicle, sample_size, scores, SETTINGS)
    # Save the predicted field back to Excel file
    save_back_to_Excel(df, vehicle, SETTINGS)


#%%
