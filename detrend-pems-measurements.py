#%% [markdown]
# ## Detrending PEMS measurements affected by sensor temperature changes
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ## Load required libraries
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import seaborn as sns
from obspy.signal.detrend import polynomial

#%% [markdown]
# ### Load data from Excel to pandas dataframe
def load_from_Excel(vehicle, device, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + device
        + "/"
        + vehicle
        + "/Processed/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings[device]["input_type"], settings[device]["input_index"]
    )
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
    df = pd.DataFrame()
    for _, sheet in sheets_dict.items():
        df = df.append(sheet)
    return df


#%% [markdown]
# ### Detrend PEMS measurements
def detrend(df, cols, vehicle, device, settings):
    detrended_df = pd.DataFrame()
    fig, axn = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("Experiment: {0}".format(vehicle), fontsize=18)
    df_segments = df.groupby("SHEET_PEMS")
    print(len(df_segments))
    mean = dict()
    for index, (_, segment) in enumerate(df_segments):
        if settings["mean_shift_target_segment"] == index:
            for col in cols:
                if settings["mean_shift_from_the_end"]:
                    mean[col] = segment[col][-1800:].mean()
                else:
                    mean[col] = segment[col][:1800].mean()
    for _, segment in df_segments:
        for col in cols:
            polynomial(segment[col], order=settings["detrending_order"])
            if settings["mean_shift"]:
                segment[col] += mean[col]
            segment[col][segment[col] < 0] = 0
        detrended_df = detrended_df.append(segment)
    for ax, col in zip(axn, cols):
        if settings["plot_original"]:
            sns.lineplot(data=df[col], ax=ax, label="Original", ci=None)
        if settings["plot_detrended"]:
            sns.lineplot(data=detrended_df[col], ax=ax, label="Detrended", ci=None)
        ax.legend(loc="best")
        ax.set_title(col)
    plt.show()
    fig.savefig(
        "../../../Google Drive/Academia/PhD Thesis/Modeling Outputs/{0}/{1} - Detrended PM and PN.jpg".format(
            settings[device]["output_type"], vehicle
        ),
        dpi=300,
        quality=95,
        bbox_inches="tight",
    )
    return detrended_df


#%% [markdown]
# ### Save the detrended measurements back to Excel file
def save_back_to_Excel(df, vehicle, device, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + device
        + "/"
        + vehicle
        + "/Processed/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings[device]["output_type"], settings[device]["output_index"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(
            writer, sheet_name="With Detrended PM and PN", header=True, index=None
        )
    print("{0} saved to Excel successfully!".format(vehicle))
    return None


#%% [markdown]
# ### General Settings
plt.style.use("bmh")
pd.options.mode.chained_assignment = None
EXPERIMENTS = (
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
)

#%% [markdown]
# ### Detrend settings
SETTINGS = {
    "3DATX parSYNC Plus": {
        "input_type": "NONE",
        "input_index": "01",
        "output_type": "NONE",
        "output_index": "02",
    },
    "detrending_order": 10,
    "mean_shift": True,
    "mean_shift_from_the_end": True,
    "mean_shift_target_segment": 0,
    "plot_original": True,
    "plot_detrended": True,
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data corresponding to vehicle and device into a dataframe
    df = load_from_Excel(vehicle, "3DATX parSYNC Plus", SETTINGS)
    # Detrend PEMS measurements
    df = detrend(df, ["PM_MGM3", "PN_#CM3"], vehicle, "3DATX parSYNC Plus", SETTINGS)
    # Save the detrended measurements back to Excel file
    save_back_to_Excel(df, vehicle, "3DATX parSYNC Plus", SETTINGS)


# %%
