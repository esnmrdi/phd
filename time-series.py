#%% [markdown]
# ## Time-Series Modeling
# Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import preprocessing

#%% [markdown]
# ### General settings
vehicle = "009 Renault Logan 2014 (1.6L Manual)"
dependent = "fcr_gs"
main_features = model_features = ["spd_si", "acc_si", "corr_grade", "rpm"]
lagged_features = {"spd_si": 2, "acc_si": 0, "corr_grade": 2}
split_ratio = {
    "train": 0.6,
    "dev": 0.2,
    "test": 0.2,
}  # for training, dev, and test sets
seed = 20
labels = {
    "fcr_gs": "Fuel Consumption Rate (g/s)",
    "spd_si": "Speed (m/s)",
    "acc_si": "Acceleration (m/s2)",
    "corr_grade": "Grade (deg)",
    "rpm": "Engine Speed (rpm)",
    "state": "Engine State",
}
for key, value in lagged_features.items():
    for i in range(value):
        labels.update({key + "_l" + str(i + 1): labels[key] + " - lag " + str(i + 1)})

#%% [markdown]
# ### Loading observations from Excel into a pandas dataframe
dir = r"/Users/ehsan/Dropbox/Academia/PhD Thesis/Field Experiments/Veepeak"
file = r"/009 Renault Logan 2014 (1.6L Manual)/Processed/009 Renault Logan 2014 (1.6L Manual).xlsx"
path = dir + file
df = pd.read_excel(path, sheet_name="Prepared for Modeling")

#%% [markdown]
# ### Plot time-series variation of speed, acceleration, and grade
fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.6)
gs.tight_layout(fig)
for index, g in enumerate(gs):
    if index < 4:
        y = model_features[index]
    else:
        y = dependent
    ax = fig.add_subplot(g)
    ax.set_title(y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(labels[y])
    ax.grid(color="k", linestyle=":", linewidth=1, alpha=0.25)
    ax.plot(df[y][100:200])

#%% [markdown]
# ### Label the data into time-windows selected based on feature ranges
states_num = 10
states = np.linspace(df["rpm"].min() - 1, df["rpm"].max() + 1, num=states_num)
tags = [int(index + 1) for index, value in enumerate(ranges)][:-1]
df["rpm_state"] = pd.cut(df["spd_si"], bins=bins, labels=tags).astype("int64")

#%% [markdown]
# ### Plot spd_si vs. speed spd_si_bin
features = ["spd_si", "spd_si_bin"]
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6)
gs.tight_layout(fig)
for index, g in enumerate(gs):
    y = features[index]
    ax = fig.add_subplot(g)
    ax.set_title(y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(labels[y])
    ax.grid(color="k", linestyle=":", linewidth=1, alpha=0.25)
    ax.plot(df[y][:])

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
