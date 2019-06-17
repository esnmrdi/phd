#%% [markdown]
# ## Suppoer Vector Regression
# Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Load required libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
import itertools

plt.rcParams["axes.grid"] = True

#%% [markdown]
# ### General settings
vehicle = "013 Geely Emgrand7 2014 (1.8L Auto)"
dependent = "fcr_gs"
main_features = model_features = ["spd_si", "acc_si", "corr_grade"]
lagged_features = {"spd_si": 0, "acc_si": 0, "corr_grade": 0}
split_ratio = {
    "train": 0.7,
    "dev": 0.15,
    "test": 0.15,
}  # for training, dev, and test sets
seed = 20
labels = {
    "fcr_gs": "Fuel Consumption Rate (g/s)",
    "spd_si": "Speed (m/s)",
    "acc_si": "Acceleration (m/s2)",
    "corr_grade": "Grade (deg)",
    "state": "Engine State",
}
for key, value in lagged_features.items():
    for i in range(value):
        labels.update({key + "_l" + str(i + 1): labels[key] + " - lag " + str(i + 1)})

#%%% [markdown]
# ### Loading observations from Excel into a pandas dataframe
dir = r"/Users/ehsan/Dropbox/Academia/PhD Thesis/Field Experiments/Veepeak"
file_original = r"/" + vehicle + "/Processed/" + vehicle + ".xlsx"
file_hmm = r"/" + vehicle + "/Processed/" + vehicle + " - HMM Output.xlsx"
path = dir + file_original
reader = pd.ExcelFile(path)
df = pd.read_excel(reader, "Prepared for Modeling")
path = dir + file_hmm
reader = pd.ExcelFile(path)
df["state"] = pd.read_excel(reader, "HMM Output").astype(float)

#%% [markdown]
# ### Adding custom lagged variables to dataframe
df = df[model_features + [dependent]]
for feature, lag_order in lagged_features.items():
    for i in range(lag_order):
        new_feature = feature + "_l" + str(i + 1)
        df[new_feature] = df[feature].shift(i + 1)
        model_features.append(new_feature)
df = df.dropna()

#%% [markdown]
# ### Splitting data into train and test segments and selecting features for modeling
train_set = df.sample(frac=split_ratio["train"], random_state=seed)
leftover = df.drop(train_set.index)
dev_set = leftover.sample(frac=split_ratio["dev"] / (1.0 - split_ratio["train"]))
test_set = leftover.drop(dev_set.index)

#%% [markdown]
# ### Feature scaling
train_set_scaled, dev_set_scaled, test_set_scaled = train_set, dev_set, test_set
scaler = preprocessing.StandardScaler().fit(train_set[model_features])
train_set_scaled[model_features] = scaler.transform(train_set[model_features])
dev_set_scaled[model_features] = scaler.transform(dev_set[model_features])
test_set_scaled[model_features] = scaler.transform(test_set[model_features])

#%% [markdown]
# ### SVR settings
kernel = "rbf"
C = 100
epsilon = 0.1
gamma = "scale"
degree = 4  # only for polynomial kernel
settings_report = "SVR (Kernel = {0}, C = {1}, epsilon = {2}, gamma = {3}, degree = {4})".format(
    kernel, C, epsilon, gamma, degree
)

#%% [markdown]
# ### SVR modeling
svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=degree)
svr.fit(train_set_scaled[model_features], train_set[dependent])

#%% [markdown]
# ### SVR Modeling score (R-squared) calculation
train_score = round(
    svr.score(train_set_scaled[model_features], train_set[dependent]), 2
)
dev_score = round(svr.score(dev_set_scaled[model_features], dev_set[dependent]), 2)
test_set_scaled["predictions"] = svr.predict(test_set_scaled[model_features])
score_report = "Train Score: {0} | Dev Score: {1}".format(train_score, dev_score)
print(settings_report, "\n", score_report)

#%% [markdown]
# ### Tuning the SVR model using grid search
param_grid = {
    "kernel": ["rbf"],
    "gamma": [1e-3],
    "C": [0.1, 1, 10, 100],
    "epsilon": np.linspace(0.01, 0.1, 5),
}
print("Tuning hyper-parameters ...")
clf = GridSearchCV(SVR(), param_grid, cv=5, scoring="r2")
clf.fit(train_set_scaled[model_features], train_set[dependent])
print("Best parameters set found on development set:\n")
print(clf.best_params_)

#%% [markdown]
# ### Plotting the results (Contour Plots)
pairs = list(itertools.combinations(model_features, 2))
fig = plt.figure(figsize=(12, 4 * len(pairs)))
gs = gridspec.GridSpec(len(pairs), 2, figure=fig, hspace=0.3)
gs.tight_layout(fig)
# fig.suptitle(vehicle + '\n' + settings_report + '\n' + score_report, fontsize = 16)
for index, g in enumerate(gs):
    ax = fig.add_subplot(g)
    if index == 0:
        ax.set_title("Observations")
    elif index == 1:
        ax.set_title("Predictions")
    p_index = int(index / 2)
    p_flag = True if index % 2 == 0 else False
    if p_flag:
        set, output = train_set_scaled, dependent
    else:
        set, output = test_set_scaled, "predictions"
    x = set[model_features][pairs[p_index][0]]
    y = set[model_features][pairs[p_index][1]]
    z = set[output]
    if p_flag:
        x_min, x_max, y_min, y_max, z_max = x.min(), x.max(), y.min(), y.max(), z.max()
    cntr = ax.tricontourf(
        x,
        y,
        z,
        levels=np.linspace(0, round(z_max, 1), 11),
        cmap="Spectral_r",
        antialiased=True,
    )
    ax.axis((x_min, x_max, y_min, y_max))
    ax.set_xlabel(labels[pairs[p_index][0]])
    ax.set_ylabel(labels[pairs[p_index][1]])
    ax.grid(color="k", linestyle=":", linewidth=1, alpha=0.25)
    fig.colorbar(cntr, ax=ax)

fig.savefig("svr.png", bbox_inches="tight", dpi=200)
plt.show()
