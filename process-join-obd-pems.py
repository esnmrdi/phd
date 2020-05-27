# %%
# Joining OBD and PEMS data
# Ehsan Moradi, Ph.D. Candidate

# pylint: disable=abstract-class-instantiated

# %%
# Load required libraries
import pandas as pd

# %%
# Load data from Excel to pandas dataframe
def load_from_Excel(vehicle, device, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + device
        + "/"
        + vehicle
        + "/Processed/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings[device]["INPUT_TYPE"], settings[device]["INPUT_INDEX"]
    )
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
    df = pd.DataFrame()
    for _, sheet in sheets_dict.items():
        df = df.append(sheet, ignore_index=True)
    return df


# %%
# Save the joined table back to Excel file
def save_to_excel(joined_table, vehicle, device, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + device
        + "/"
        + vehicle
        + "/Processed/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings[device]["OUTPUT_TYPE"], settings[device]["OUTPUT_INDEX"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(
        output_path, engine="openpyxl", mode="w"
    ) as writer:  # pylint: disable=abstract-class-instantiated
        joined_table.to_excel(
            writer, sheet_name="Joined Table", header=True, index=None
        )
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


# %%
# General Settings
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

# %%
# Join settings
SETTINGS = {
    "Veepeak": {"INPUT_TYPE": "NONE", "INPUT_INDEX": "04"},
    "3DATX parSYNC Plus": {
        "INPUT_TYPE": "NONE",
        "INPUT_INDEX": "00",
        "OUTPUT_TYPE": "NONE",
        "OUTPUT_INDEX": "01",
    },
}

# %%
# Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data corresponding to vehicle and device into a dataframe
    left = load_from_Excel(vehicle, "Veepeak", SETTINGS)
    right = load_from_Excel(vehicle, "3DATX parSYNC Plus", SETTINGS)
    left["DATETIME"] = left["DATETIME"].dt.strftime("%Y-%m-%d %H:%M:%S")
    right["DATETIME"] = right["DATETIME"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # Join tables based on DATETIME as the common column
    joined_table = pd.merge_ordered(left, right, how="inner", on="DATETIME")
    joined_table["DATETIME"] = pd.to_datetime(
        joined_table["DATETIME"], format="%Y-%m-%d %H:%M:%S"
    )
    # Save the joined table back to Excel file
    save_to_excel(joined_table, vehicle, "3DATX parSYNC Plus", SETTINGS)

# %%
