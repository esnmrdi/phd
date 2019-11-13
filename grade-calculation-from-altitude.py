#%% [markdown]
# ## Grade calculation from altitude
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Loading required packages
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns

#%% [markdown]
# ### Loading data from Excel to a pandas dataframe
def load_from_Excel(vehicle, settings):
    directory = "../Field Experiments/Veepeak/" + vehicle + "/Processed/"
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["input_type"], settings["input_index"]
    )
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=4)
    df_dict = {}
    for sheet_name, df in sheets_dict.items():
        df_dict[sheet_name] = df
    return df_dict


#%% [markdown]
# ### Convolutional smoothing method
def convolutional_smooth(y, bandwidth):
    box = np.ones(bandwidth) / bandwidth
    y_smooth = pd.Series(np.convolve(y, box, mode="same"))
    return y_smooth


#%% [markdown]
# ### Savitzky-Golay smoothing method
def savitzky_golay_smooth(y, bandwidth, order):
    if bandwidth % 2 == 0:
        bandwidth += 1
    y_smooth = pd.Series(scipy.signal.savgol_filter(y, bandwidth, order))
    return y_smooth


#%% [markdown]
# ### Hampel outlier-filtering method
def hampel(y, bandwidth, t0):
    L = 1.4826
    rolling_median = y.rolling(bandwidth).median()
    difference = np.abs(rolling_median - y)
    median_abs_deviation = difference.rolling(bandwidth).median()
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold
    y_no_outlier = y.copy()
    y_no_outlier[outlier_idx] = np.nan
    return y_no_outlier


#%% [markdown]
# ### Trim the dataframe to the safe margins
def trim(df, bandwidth):
    safe_margin = int(bandwidth / 2)
    df_trimmed = df[safe_margin:-safe_margin]
    return df_trimmed


#%% [markdown]
# ### Calculate grade based on Savitzky-Golay smoothed altitudes
def calculate_grade(sg_altitude, distance):
    shifted_sg_altitude = sg_altitude.shift(1)
    diff_sg_altitude = sg_altitude - shifted_sg_altitude
    shifted_distance = distance.shift(1)
    diff_distance = distance - shifted_distance
    calculated_grade = pd.Series([np.nan for i in range(distance.size)])
    prev_grade = None
    for index, value in diff_distance.iteritems():
        if value > 0:
            calculated_grade[index] = prev_grade = round(
                180
                * (np.arcsin(diff_sg_altitude[index] / (diff_distance[index] * 1000)))
                / np.pi,
                1,
            )
        else:
            calculated_grade[index] = prev_grade
    return calculated_grade


#%% [markdown]
# ### Post-process calculated grade to get rid of outliers
def remove_outlier_grades(calculated_grade):
    threshold = 15
    calculated_grade_rolling_median = (
        calculated_grade.rolling(window=3, center=True)
        .median()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )
    difference = np.abs(calculated_grade - calculated_grade_rolling_median)
    outlier_idx = difference > threshold
    no_outlier_grade = calculated_grade.copy()
    no_outlier_grade[outlier_idx] = calculated_grade_rolling_median[outlier_idx]
    return no_outlier_grade


#%% [markdown]
# ### Saving the calculated field back in Excel file
def save_back_to_Excel(df, vehicle, trip, index, settings):
    df = df[1:]
    df = df.dropna()
    directory = "../Field Experiments/Veepeak/" + vehicle + "/Processed/"
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["output_type"], settings["output_index"]
    )
    output_path = directory + output_file
    write_mode = "w" if index == 0 else "a"
    with pd.ExcelWriter(output_path, engine="openpyxl", mode=write_mode) as writer:
        df.to_excel(writer, sheet_name=trip, header=True, index=None)
    print("{0} - {1} saved to Excel successfully!".format(vehicle, trip))
    return None


#%% [markdown]
# ### Plotting the RAW altitude
def plot_raw_altitude(df, vehicle, trip, settings):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    title = vehicle + " on " + trip[:10] + " @ " + trip[11:]
    fig.suptitle(title, fontsize=18)
    ax = sns.lineplot(x="DIST_KM", y="ALT_M", data=df, ax=ax, ci=None)
    ax.set(xlabel="Distance (km)", ylabel="Altitude (m)")
    plt.show()
    fig.savefig(
        "../Modeling Outputs/{0}/{1} - RAW Altitude.jpg".format(
            settings["input_type"], title
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )
    return None


#%% [markdown]
# ### Plotting the RAW altitude vs. Savitzky-Golay algorithm outputs
def plot_savitzky_golay_output(df, vehicle, trip, settings):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    title = vehicle + " on " + trip[:10] + " @ " + trip[11:]
    fig.suptitle(title, fontsize=18)
    ax = sns.lineplot(x="DIST_KM", y="ALT_M", data=df, ax=ax, label="RAW", ci=None)
    ax = sns.lineplot(
        x="DIST_KM", y="SG_ALT_M", data=df, ax=ax, label="Savitsky-Golay", ci=None
    )
    ax.set(xlabel="Distance (km)", ylabel="Altitude (m)")
    ax.legend(loc="best")
    plt.show()
    fig.savefig(
        "../Modeling Outputs/{0}/{1} - Savitzky-Golay Output.jpg".format(
            settings["input_type"], title
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )
    return None


#%% [markdown]
# ### Plotting the calculated grade
def plot_grade_estimates(df, vehicle, trip, settings):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    title = vehicle + " on " + trip[:10] + " @ " + trip[11:]
    fig.suptitle(title, fontsize=18)
    ax = sns.lineplot(
        x="DIST_KM", y="CALC_GRADE_DEG", data=df, ax=ax, label="Calc. Grade", ci=None
    )
    ax = sns.lineplot(
        x="DIST_KM",
        y="NO_OUTLIER_GRADE_DEG",
        data=df,
        ax=ax,
        label="No-Outlier Grade",
        ci=None,
    )
    ax.set(xlabel="Distance (km)", ylabel="Grade Estimate (deg)")
    plt.show()
    fig.savefig(
        "../Modeling Outputs/{0}/{1} - Estimated Grade.jpg".format(
            settings["input_type"], title
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )
    return None


#%% [markdown]
# ### General Settings
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
]
EXPERIMENTS = [
    "036 Jeep Patriot 2010 (2.4L Auto)",
    "037 Chevrolet Malibu 2019 (1.5L TC Auto)",
    "038 Kia Optima 2012 (2.4L Auto)",
    "039 Honda Fit 2009 (1.5L Auto)",
]

#%% [markdown]
# ### Grade calculation settings
SETTINGS = {
    "input_type": "NONE",
    "output_type": "NONE",
    "input_index": "00",
    "output_index": "01",
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data corresponding to vehicle and trip into a dataframe
    df_dict = load_from_Excel(vehicle, SETTINGS)
    # Loop through all sheets
    for index, (trip, df) in enumerate(df_dict.items()):
        # Plot RAW altitude
        plot_raw_altitude(df, vehicle, trip, SETTINGS)
        # Applying Savitzky-Golay smoothing on RAW altitude
        df = df.assign(
            SG_ALT_M=savitzky_golay_smooth(df["ALT_M"], bandwidth=40, order=2)
        )
        # Plot Savitzky-Golay algorithm output
        plot_savitzky_golay_output(df, vehicle, trip, SETTINGS)
        # Calculate grade
        df = df.assign(CALC_GRADE_DEG=calculate_grade(df["SG_ALT_M"], df["DIST_KM"]))
        # Post-process calculated grade
        df = df.assign(NO_OUTLIER_GRADE_DEG=remove_outlier_grades(df["CALC_GRADE_DEG"]))
        # Plot grade estimates
        plot_grade_estimates(df, vehicle, trip, SETTINGS)
        # Save dataframe to a new Excel file
        save_back_to_Excel(df, vehicle, trip, index, SETTINGS)

#%%
