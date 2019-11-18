#%% [markdown]
# ## Joining OBD and PEMS data
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ## Loading required libraries
import numpy as np
import pandas as pd

#%% [markdown]
# ### Loading data from Excel to pandas dataframe
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
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=1)
    merged_df = pd.DataFrame()
    for sheet_name, df in sheets_dict.items():
        temp = df
        temp["sheet"] = sheet_name
        merged_df.append(temp)
    return merged_df


#%% [markdown]
# ### Performing inner join between two pandas dataframes
def inner_join(left, right, join_key):
    return pd.merge(left, right, how="inner", on=join_key, sort=True, copy=True)


#%% [markdown]
# ### Saving the joined table back to Excel file
def save_back_to_Excel(joined_table, vehicle, device, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + device
        + "/"
        + vehicle
        + "/Processed/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["output_type"], settings["output_index"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        joined_table.to_excel(writer, sheet_name="Joined Table", header=True, index=None)
    print("{0} saved to Excel successfully!".format(vehicle))
    return None


#%% [markdown]
# ### General Settings
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
)

#%% [markdown]
# ### Join settings
SETTINGS = {
    "Veepeak": {"input_type": "NONE", "input_index": "01"},
    "3DATX parSYNC Plus": {
        "input_type": "NONE",
        "input_index": "00",
        "output_type": "NONE",
        "output_index": "01",
    },
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data corresponding to vehicle and device into a dataframe
    left = load_from_Excel(vehicle, "Veepeak", SETTINGS)
    right = load_from_Excel(vehicle, "3DATX parSYNC Plus", SETTINGS)
    joined_table = inner_join(left, right, join_key="DATETIME")
    save_back_to_Excel(joined_table, vehicle, "3DATX parSYNC Plus", SETTINGS)