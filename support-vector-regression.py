#%% [markdown]
# ## Support Vector Regression for FCR Prediction
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel

#%% [markdown]
# ### Load sample data from Excel to a pandas dataframe
def load_sample_from_Excel(vehicle, settings):
    directory = "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/" + vehicle + "/Processed/"
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
def add_lagged_features(df, settings, index):
    df_temp = df.copy()
    total_features = [
        "RPM_PRED_" + settings["RPM_BEST_ARCHS"][index]
        if feature == "RPM_PRED"
        else feature
        for feature in settings["features"]
    ]
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
# ### RBF and Linear, Multiplicative Kernel
def build_rbf_lin_mul(**kwargs):
    def rbf_lin_mul(x, y):
        k1 = linear_kernel(x, y)
        k2 = rbf_kernel(x, y, kwargs["gamma"])
        return k1 * k2

    return rbf_lin_mul


#%% [markdown]
# ### RBF and Linear, Linear Combination Kernel
def build_rbf_lin_lin(**kwargs):
    def rbf_lin_lin(x, y):
        k1 = linear_kernel(x, y)
        k2 = rbf_kernel(x, y, kwargs["gamma"])
        return kwargs["c1"] * k1 + kwargs["c2"] * k2

    return rbf_lin_lin


#%% [markdown]
# ### RBF and Polynomial, Multiplicative Kernel
def build_rbf_pol_mul(**kwargs):
    def rbf_pol_mul(x, y):
        k1 = polynomial_kernel(x, y, kwargs["degree"], kwargs["gamma"])
        k2 = rbf_kernel(x, y, kwargs["gamma"])
        return k1 * k2

    return rbf_pol_mul


#%% [markdown]
# ### RBF and Polynomial, Linear Combination Kernel
def build_rbf_pol_lin(**kwargs):
    def rbf_pol_lin(x, y):
        k1 = polynomial_kernel(x, y, kwargs["degree"], kwargs["gamma"])
        k2 = rbf_kernel(x, y, kwargs["gamma"])
        return kwargs["c1"] * k1 + kwargs["c2"] * k2

    return rbf_pol_lin


#%% [markdown]
# ### Tune the SVR model using grid search and cross validation
def tune_svr(df, total_features, scaler, settings):
    df_temp = df.copy()
    cv = KFold(n_splits=settings["n_splits"], shuffle=True)
    clf = GridSearchCV(
        SVR(kernel="rbf"),
        param_grid=settings["param_grid"],
        cv=cv,
        scoring="r2",
        verbose=1,
        n_jobs=-1,
    )
    clf.fit(df_temp[total_features], df_temp[settings["dependent"]])
    df_temp[settings["predicted"]] = clf.predict(df_temp[total_features])
    df_temp[[settings["dependent"]] + [settings["predicted"]]] = reverse_scale(
        df_temp[[settings["dependent"]] + [settings["predicted"]]], scaler
    )
    return df_temp, clf.best_score_, clf.best_estimator_, clf.cv_results_


#%% [markdown]
# ### Plot the grid search results and save plot to file
def plot_grid_search_results(vehicle, sample_size, best_score, cv_results, settings):
    results = pd.DataFrame()
    results["epsilon"] = cv_results["param_epsilon"]
    results["gamma"] = cv_results["param_gamma"]
    results["C"] = cv_results["param_C"]
    results["score"] = cv_results["mean_test_score"]
    results.sort_values(["epsilon", "gamma", "C"], ascending=True, inplace=True)
    fig, axn = plt.subplots(
        1,
        len(settings["param_grid"]["epsilon"]),
        figsize=(20, 5),
        constrained_layout=True,
    )
    fig.suptitle(
        "Experiment: {0}\nSample Size: {1}\nFive-Fold CV Score: {2}".format(
            vehicle, sample_size, np.round(best_score, 3)
        ),
        fontsize=18,
    )
    for index, ax in enumerate(axn.flat):
        epsilon = settings["param_grid"]["epsilon"][index]
        sub_result = results.loc[results["epsilon"] == epsilon]
        sub_result = sub_result.drop(["epsilon"], axis=1)
        matrix = sub_result.pivot("C", "gamma", "score")
        ax.set_width = 10
        ax.set_height = 10
        sns.heatmap(
            matrix,
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=1,
            annot=True,
            square=True,
            cbar=True,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set_title("epsilon = {}".format(np.round(epsilon, 5)))
    plt.show()
    fig.savefig(
        "../../../Google Drive/Academia/PhD Thesis/Modeling Outputs/{0}/{1} - {2}/{3} - Grid Search Result.jpg".format(
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
def plot_accuracy(df, vehicle, sample_size, best_score, settings):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    ax = sns.regplot(
        x=settings["dependent"],
        y=settings["predicted"],
        data=df,
        fit_reg=True,
        ax=ax,
        scatter_kws={"color": "blue"},
        line_kws={"color": "red"},
        ci=None,
    )
    ax.set(
        xlabel=settings["labels"][settings["dependent"]],
        ylabel=settings["labels"][settings["predicted"]],
    )
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_title(
        "Experiment: {0}\nSample Size: {1}\nFive-Fold CV Score: {2}\n".format(
            vehicle, sample_size, np.round(best_score, 3)
        )
    )
    plt.show()
    fig.savefig(
        "../../../Google Drive/Academia/PhD Thesis/Modeling Outputs/{0}/{1} - {2}/{3} - Observed vs. Predicted.jpg".format(
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
    directory = "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/" + vehicle + "/Processed/"
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
# ### SVR settings
SETTINGS = {
    "dependent": "FCR_LH",
    "predicted": "FCR_LH_PRED",
    "features": ["SPD_KH", "ACC_MS2", "NO_OUTLIER_GRADE_DEG", "RPM_PRED"],
    "lagged_features": ["SPD_KH", "NO_OUTLIER_GRADE_DEG"],
    "lag_order": 1,
    "max_sample_size": 5400,
    "n_splits": 5,
    "param_grid": {
        "gamma": np.logspace(-3, 1, num=5, base=10),
        "C": np.logspace(-1, 2, num=4, base=10),
        "epsilon": np.logspace(-4, 0, num=5, base=10),
    },
    "labels": {
        "FCR_LH": "Observed Fuel Consumption Rate (L/H)",
        "FCR_LH_PRED": "Predicted Fuel Consumption Rate (L/H)",
        "RPM": "Observed Engine Speed (rev/min)",
        "RPM_PRED": "Predicted Engine Speed (rev/min)",
        "SPD_KH": "Speed (Km/h)",
        "ACC_MS2": "Acceleration (m/s2)",
        "NO_OUTLIER_GRADE_DEG": "Road Grade (Deg)",
    },
    "model_structure": "FCR ~ SPD + SPD_L1 + ACC + GRADE + GRADE_L1 + RPM_PRED",
    "input_type": "ANN",
    "output_type": "SVR",
    "input_index": "17",
    "output_index": "19",
    "RPM_BEST_ARCHS": [
        "(1,128)",
        "(2,64)",
        "(2,64)",
        "(2,64)",
        "(1,128)",
        "(2,64)",
        "(2,64)",
        "(1,128)",
        "(2,64)",
        "(4,32)",
        "(2,64)",
        "(2,64)",
        "(1,128)",
        "(1,128)",
        "(1,128)",
        "(4,32)",
        "(2,64)",
        "(1,128)",
        "(2,64)",
        "(1,128)",
        "(1,128)",
        "(2,64)",
        "(2,64)",
        "(2,64)",
        "(1,128)",
        "(1,128)",
        "(1,128)",
    ],
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for index, vehicle in enumerate(EXPERIMENTS):
    # Add lagged features to the dataframe and sampling
    df, sample_size = load_sample_from_Excel(vehicle, SETTINGS)
    # Add lagged features to the dataframe
    df, total_features = add_lagged_features(df, SETTINGS, index)
    # Scale the features
    df, scaler = scale(df, total_features, SETTINGS)
    # Tune the SVR model using grid search and cross validation
    df, best_score, best_estimator, cv_results = tune_svr(
        df, total_features, scaler, SETTINGS
    )
    # Plot the grid search results and save plots to file
    plot_grid_search_results(vehicle, sample_size, best_score, cv_results, SETTINGS)
    # Plot predictions vs. ground-truth and save plot to file
    plot_accuracy(df, vehicle, sample_size, best_score, SETTINGS)
    # Save the predicted field back to Excel file
    save_back_to_Excel(df, vehicle, SETTINGS)