# %% [markdown]
# Reshape sensors' output
# The output is mainly used for comparison with MOVES output
# Ehsan Moradi, Ph.D. Candidate

# pylint: disable=abstract-class-instantiated

# %%
# Load required libraries
import pandas as pd
from os import walk

# %%
# Load data from Excel to a pandas dataframe
def load_from_Excel(directory, input_file):
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
    df = pd.DataFrame()
    for _, sheet in sheets_dict.items():
        df = df.append(sheet)
    return df


# %%
# Save the aggregated result to an Excel file
def save_to_Excel(df, directory, output_file):
    output_path = directory + output_file
    with pd.ExcelWriter(
        output_path, engine="openpyxl", mode="w"
    ) as writer:  # pylint: disable=abstract-class-instantiated
        df.to_excel(writer, header=True, index=None, sheet_name="Sensor")
    print("Data is saved to Excel successfully!")
    return None


# %%
# Retrieve files with filenames starting with desired prefix
def list_filenames(directory, prefix):
    list = []
    for (_, _, filenames) in walk(directory):
        for f in sorted(filenames):
            if f.startswith(prefix):
                list.append(f)
    return list


# %%
# Convert 21hr time to 24-hr ID
def convert_to_hour_id(time):
    hour_id = int(time[:2]) + 1
    suffix = time[2:]
    if suffix == "PM":
        hour_id += 12
    return hour_id


# %%
# General settings
pd.options.mode.chained_assignment = None
EXPERIMENTS = {
    "Veepeak": (
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
    ),
    "3DATX parSYNC Plus": (
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
    ),
}

# %%
# Data transformation settings
HEADERS = {
    "FCR_JMI": {"pollutant_id": 91, "pollutant_name": "Total Energy Consumption"},
    "PM_KGMI": {"pollutant_id": 110, "pollutant_name": "Primary Exhaust PM2.5 - Total"},
    "CO2_KGMI": {"pollutant_id": 90, "pollutant_name": "Atmospheric CO2"},
    "NO2_KGMI": {"pollutant_id": 33, "pollutant_name": "Nitrogen Dioxide (NO2)"},
    "NO_KGMI": {"pollutant_id": 32, "pollutant_name": "Nitrogen Oxide (NO)"},
}


# %%
# Batch execution on all vehicles and their trips (Veepeak data)
device = "3DATX parSYNC Plus"
sensor = "PEMS"
input_index = "05"
aggregate_df = pd.DataFrame()
for vehicle in EXPERIMENTS[device]:
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/"
        + device
        + "/"
        + vehicle
        + "/Processed/"
    )
    target_filenames = list_filenames(directory, vehicle + " - NONE - " + input_index)
    for filename in target_filenames:
        # Load data from Excel to a pandas dataframe
        df = load_from_Excel(directory, filename)
        no_date_time = "_".join(
            (filename[:3], "".join(filename[-21:-11].split("-")), filename[-10:-6])
        ).lower()
        pollutant_abrvs = [key for key, index in HEADERS.items()]
        for col_name, col_data in df.iteritems():
            if col_name in pollutant_abrvs:
                for index, val in col_data.items():
                    row = {}
                    pollutant_id = HEADERS[col_name]["pollutant_id"]
                    pollutant_name = HEADERS[col_name]["pollutant_name"]
                    link_id = df["linkID"].iloc[index]
                    link_length = df["linkLength"].iloc[index]
                    row["uniqueID"] = (
                        no_date_time + "_" + str(link_id) + "_" + str(pollutant_id)
                    )
                    row["yearID"] = filename[-21:-17]
                    row["monthID"] = filename[-16:-14].strip("0")
                    row["hourID"] = convert_to_hour_id(filename[-10:-6])
                    row["linkID"] = link_id
                    row["linkLength"] = link_length
                    row["pollutantID"] = pollutant_id
                    row["pollutantName"] = pollutant_name
                    row["ratePerDistanceSensor"] = df[col_name].iloc[index]
                    # Aggregate vehicle/hour-based dataframes in another large dataframe
                    aggregate_df = aggregate_df.append(row, ignore_index=True)

aggregate_df["yearID"] = aggregate_df["yearID"].astype(int)
aggregate_df["monthID"] = aggregate_df["monthID"].astype(int)
aggregate_df["hourID"] = aggregate_df["hourID"].astype(int)
aggregate_df["linkID"] = aggregate_df["linkID"].astype(int)
aggregate_df["pollutantID"] = aggregate_df["pollutantID"].astype(int)

directory = (
    "../../../Google Drive/Academia/PhD Thesis/Field Experiments/" + device + "/"
)
save_to_Excel(aggregate_df, directory, "output_" + sensor + ".xlsx")


# %%
