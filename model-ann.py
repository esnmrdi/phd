# %%
# Artificial Neural Network for RPM and FCR Prediction
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# %%
# Display training progress
class ReportProgress(keras.callbacks.Callback):
    def __init__(self, sample, test_split_ratio, n_epochs):
        self.sample = sample
        self.test_split_ratio = test_split_ratio
        self.n_epochs = n_epochs

    def on_train_begin(self, logs):
        n_examples = len(self.sample)
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


# %%
# Load data from Excel to a pandas dataframe
def load_from_Excel(vehicle, settings):
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
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
    df.reset_index(inplace=True, drop=True)
    return df


# %%
# Load sample data from Excel to a pandas dataframe
def load_sample_from_Excel(vehicle, settings):
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
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
    df.reset_index(inplace=True, drop=True)
    if df.shape[0] > settings["MAX_SAMPLE_SIZE"]:
        sample_size = settings["MAX_SAMPLE_SIZE"]
        df = df.sample(sample_size)
    else:
        sample_size = df.shape[0]
    return df, sample_size


# %%
# Add lagged features to the dataframe
def add_lagged_features(df, settings, index):
    df_temp = df.copy()
    total_features = settings["FEATURES"]
    total_features = [
        "RPM_PRED_" + settings["RPM_BEST_ARCHS"][index]
        if feature == "RPM_PRED"
        else feature
        for feature in settings["FEATURES"]
    ]
    for feature in settings["LAGGED_FEATURES"]:
        for i in range(settings["LAG_ORDER"]):
            new_feature = feature + "_L" + str(i + 1)
            total_features.append(new_feature)
            df_temp[new_feature] = df_temp[feature].shift(i + 1)
    df_temp.dropna(inplace=True)
    return df_temp, total_features


# %%
# Scale the features
def scale(df, total_features, settings):
    df_temp = df.copy()
    scaler_features = preprocessing.StandardScaler().fit(df_temp[total_features])
    scaler_dependent = preprocessing.StandardScaler().fit(df_temp[[settings["DEPENDENT"]]])
    df_temp[total_features] = scaler_features.transform(df_temp[total_features])
    df_temp[[settings["DEPENDENT"]]] = scaler_dependent.transform(df_temp[[settings["DEPENDENT"]]])
    return df_temp, scaler_features, scaler_dependent


# %%
# Define alternative ANN models with different architectures
def define_models(total_features, settings):
    n_features = len(total_features)
    models = []
    for n_layers, n_neurons in settings["MODEL_ARCHITECTURES"]:
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(n_neurons, input_shape=[n_features], activation="relu")
        )
        model.add(keras.layers.Dropout(settings["DROP_PROP"]))
        for _ in range(n_layers - 1):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
            model.add(keras.layers.Dropout(settings["DROP_PROP"]))
        model.add(keras.layers.Dense(1, activation="linear"))
        models.append(model)
    return models


# %%
# Report the train and test score
def calculate_score(model, train_set, test_set, total_features, settings):
    train_set[settings["PREDICTED"]] = model.predict(train_set[total_features])
    test_set[settings["PREDICTED"]] = model.predict(test_set[total_features])
    r2_train = r2_score(
        train_set[settings["DEPENDENT"]], train_set[settings["PREDICTED"]]
    )
    r2_test = r2_score(test_set[settings["DEPENDENT"]], test_set[settings["PREDICTED"]])
    score = {}
    score["train"], score["test"] = r2_train, r2_test
    return score


# %%
# Tune the ANN model by testing alternative architectures (from shallow and wide to deep and narrow)
def tune_ann(df, total_features, scaler_features, scaler_dependent, settings):
    df_temp = df.copy()
    if df_temp.shape[0] > settings["MAX_SAMPLE_SIZE"]:
        sample_size = settings["MAX_SAMPLE_SIZE"]
        sample = df.sample(sample_size)
    else:
        sample_size = df.shape[0]
        sample = df
    models = define_models(total_features, settings)
    train_set, test_set = train_test_split(
        sample, test_size=settings["TEST_SPLIT_RATIO"], shuffle=True
    )
    x_train, y_train = train_set[total_features], train_set[settings["DEPENDENT"]]
    x_test, y_test = test_set[total_features], test_set[settings["DEPENDENT"]]
    histories = []
    scores = []
    for index, model in enumerate(models):
        model.compile(
            loss=settings["METRICS"][0],
            optimizer=keras.optimizers.RMSprop(lr=settings["LEARNING_RATE"]),
            metrics=settings["METRICS"],
        )
        history = model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=settings["N_EPOCHS"],
            validation_data=(x_test, y_test),
            shuffle=False,
            verbose=0,
            callbacks=[
                ReportProgress(
                    sample, settings["TEST_SPLIT_RATIO"], settings["N_EPOCHS"]
                )
            ],
        )
        # predicted_temp = "{0}_({1},{2})".format(
        #     settings["PREDICTED"],
        #     settings["MODEL_ARCHITECTURES"][index][0],
        #     settings["MODEL_ARCHITECTURES"][index][1],
        # )
        df_temp[settings["PREDICTED"]] = model.predict(df_temp[total_features])
        df_temp[settings["PREDICTED"]] = scaler_dependent.inverse_transform(df_temp[settings["PREDICTED"]])
        histories.append(history)
        score = calculate_score(model, train_set, test_set, total_features, settings)
        scores.append(score)
    df_temp[total_features] = scaler_features.inverse_transform(df_temp[total_features])
    df_temp[[settings["DEPENDENT"]]] = scaler_dependent.inverse_transform(df_temp[[settings["DEPENDENT"]]])
    return df_temp, scores, histories


# %%
# Plot training history and save plot to file
def plot_training_results(vehicle, sample_size, scores, histories, settings):
    fig, axn = plt.subplots(
        len(settings["MODEL_ARCHITECTURES"]),
        len(settings["METRICS"]),
        figsize=(30, 5 * len(settings["MODEL_ARCHITECTURES"])),
        constrained_layout=True,
    )
    for ax, metric in zip(axn[0], settings["METRICS"]):
        ax.set_title(metric)
    fig.suptitle(
        "Experiment: {0}\nSample Size: {1}\n# of Epochs: {2}".format(
            vehicle, sample_size, settings["N_EPOCHS"]
        ),
        fontsize=24,
    )
    for index, ax in enumerate(axn.flat):
        row = index // len(settings["METRICS"])
        col = index % len(settings["METRICS"])
        history = histories[row]
        hist = pd.DataFrame(history.history)
        epochs = history.epoch
        sns.lineplot(
            x=epochs,
            y=settings["METRICS"][col],
            data=hist,
            ax=ax,
            label="Train",
            ci=None,
        )
        sns.lineplot(
            x=epochs,
            y="val_" + settings["METRICS"][col],
            data=hist,
            ax=ax,
            label="Test",
            ci=None,
        )
        ax.set(
            xlabel="# of Epochs",
            ylabel="Architecture: {0}\nTrain Score: {1} | Test Score: {2}".format(
                settings["MODEL_ARCHITECTURES"][row],
                np.round(scores[row]["train"], 3),
                np.round(scores[row]["test"], 3),
            ),
        )
        ax.legend(loc="best")
    plt.show()
    fig.savefig(
        "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/{0}/{1} - {2}/{3} - Training Result.jpg".format(
            settings["OUTPUT_TYPE"],
            settings["OUTPUT_INDEX"],
            settings["MODEL_STRUCTURE"],
            vehicle,
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )


# %%
# Plot predictions vs. ground-truth and save plot to file
def plot_accuracy(df, vehicle, sample_size, scores, settings):
    fig, axn = plt.subplots(
        len(settings["MODEL_ARCHITECTURES"]),
        1,
        figsize=(5, 5 * len(settings["MODEL_ARCHITECTURES"])),
        constrained_layout=True,
    )
    fig.suptitle(
        "Experiment: {0}\nSample Size: {1}\n# of Epochs: {2}".format(
            vehicle, sample_size, settings["N_EPOCHS"]
        ),
        fontsize=18,
    )
    for index, ax in enumerate(axn.flat):
        predicted_temp = "{0}_({1},{2})".format(
            settings["PREDICTED"],
            settings["MODEL_ARCHITECTURES"][index][0],
            settings["MODEL_ARCHITECTURES"][index][1],
        )
        sns.regplot(
            x=settings["DEPENDENT"],
            y=predicted_temp,
            data=df,
            fit_reg=True,
            ax=ax,
            scatter_kws={"color": "blue"},
            line_kws={"color": "red"},
            ci=None,
        )
        ax.set(
            xlabel=settings["labels"][settings["DEPENDENT"]],
            ylabel=settings["labels"][settings["PREDICTED"]],
        )
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_title(
            "Architecture: {0}\nTrain Score: {1} | Test Score: {2}".format(
                settings["MODEL_ARCHITECTURES"][index],
                np.round(scores[index]["train"], 3),
                np.round(scores[index]["test"], 3),
            )
        )
    plt.show()
    fig.savefig(
        "../../Google Drive/Academia/PhD Thesis/Modeling Outputs/{0}/{1} - {2}/{3} - Observed vs. Predicted.jpg".format(
            settings["OUTPUT_TYPE"],
            settings["OUTPUT_INDEX"],
            settings["MODEL_STRUCTURE"],
            vehicle,
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )


# %%
# Save the predicted field back to Excel file
def save_to_excel(df, vehicle, settings):
    directory = (
        "../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["OUTPUT_TYPE"], settings["OUTPUT_INDEX"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


# %%
# General settings
plt.style.use("bmh")
pd.options.mode.chained_assignment = None
EXPERIMENTS = (
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
    "024 Renault Logan 2010 (1.4L Manual)",
    "025 Chevrolet Captiva 2010 (2.4L Auto)",
    "026 Nissan Versa 2013 (1.6L Auto)",
    "027 Chevrolet Cruze 2011 (1.8L Manual)",
    "028 Nissan Sentra 2019 (1.8L Auto)",
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
EXPERIMENTS = ("027 Chevrolet Cruze 2011 (1.8L Manual)",)

# %%
# ANN settings
SETTINGS = {
    "DEPENDENT": "FCR_LH",
    "PREDICTED": "CASCADED_FCR_LH",
    "FEATURES": ["SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG", "RPM_PRED"],
    # "LAGGED_FEATURES": ["SPD_KH", "NO_OUTLIER_GRADE_DEG"],
    "LAGGED_FEATURES": [],
    # "LAG_ORDER": 1,
    "LAG_ORDER": 0,
    "MAX_SAMPLE_SIZE": 5400,
    "TEST_SPLIT_RATIO": 0.20,
    "N_EPOCHS": 200,
    "DROP_PROP": 0.1,
    "LABELS": {
        "FCR_LH": "Observed Fuel Consumption Rate (L/H)",
        "FCR_LH_PRED": "Predicted Fuel Consumption Rate (L/H)",
        "RPM": "Observed Engine Speed (rev/min)",
        "RPM_PRED": "Predicted Engine Speed (rev/min)",
        "SPD_KH": "Speed (Km/h)",
        "ACC_MS2": "Acceleration (m/s2)",
        "NO_OUTLIER_GRADE_DEG": "Road Grade (Deg)",
    },
    # "MODEL_STRUCTURE": "FCR ~ SPD + SPD_L1 + ACC + GRADE + GRADE_L1 + RPM_PRED",
    "MODEL_STRUCTURE": "CASCADED_FCR_LH ~ SPD + ACC + GRADE + RPM_PRED",
    # "MODEL_ARCHITECTURES": [(1, 128), (2, 64), (4, 32), (8, 16)],
    "MODEL_ARCHITECTURES": [(4, 32)],
    "LEARNING_RATE": 0.001,
    "METRICS": [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "cosine_proximity",
    ],
    "INPUT_TYPE": "NONE",
    "OUTPUT_TYPE": "NONE",
    "INPUT_INDEX": "09",
    "OUTPUT_INDEX": "10",
    # "RPM_BEST_ARCHS": [
    #     "(1,128)",
    #     "(2,64)",
    #     "(2,64)",
    #     "(2,64)",
    #     "(1,128)",
    #     "(2,64)",
    #     "(2,64)",
    #     "(1,128)",
    #     "(2,64)",
    #     "(4,32)",
    #     "(2,64)",
    #     "(2,64)",
    #     "(1,128)",
    #     "(1,128)",
    #     "(1,128)",
    #     "(4,32)",
    #     "(2,64)",
    #     "(1,128)",
    #     "(2,64)",
    #     "(1,128)",
    #     "(1,128)",
    #     "(2,64)",
    #     "(2,64)",
    #     "(2,64)",
    #     "(1,128)",
    #     "(1,128)",
    #     "(1,128)",
    # ],
}

# %%
# Batch execution on all vehicles and their trips
for index, vehicle in enumerate(EXPERIMENTS):
    # Add lagged features to the dataframe and sampling
    df = load_from_Excel(vehicle, SETTINGS)
    # df, sample_size = load_sample_from_Excel(vehicle, SETTINGS)
    # Add lagged features to the dataframe
    # df, total_features = add_lagged_features(df, SETTINGS, index)
    total_features = SETTINGS["FEATURES"]
    # Scale the features
    df, scaler_features, scaler_dependent = scale(df, total_features, SETTINGS)
    # Tune the ANN model by testing alternative architectures (from shallow and wide to deep and narrow)
    df, scores, histories = tune_ann(df, total_features, scaler_features, scaler_dependent, SETTINGS)
    # Plot training histories for all model architectures and save plots to file
    # plot_training_results(vehicle, sample_size, scores, histories, SETTINGS)
    # Plot predictions vs. ground-truth and save plot to file
    # plot_accuracy(df, vehicle, sample_size, scores, SETTINGS)
    # Save the predicted field back to Excel file
    save_to_excel(df, vehicle, SETTINGS)

# %%

# %%
