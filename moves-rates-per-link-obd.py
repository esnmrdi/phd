# %%
# Convert Time-based Fuel Rates to Distance-based Rates per Link (OBD Data)
# The output is mainly used for comparison with MOVES output
# Ehsan Moradi, Ph.D. Candidate

# pylint: disable=abstract-class-instantiated

# %%
# Load required libraries
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Load data from Excel to a pandas dataframe
def load_from_Excel(vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
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
    return df


# %%
# Save the aggregated results per link back to Excel files split based on 1-hour time slots
def save_to_Excel(links, datetime_tag, vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
    )
    output_file_links = vehicle + " - {0} - {1} ({2}).xlsx".format(
        settings["OUTPUT_TYPE"], settings["OUTPUT_INDEX_LINKS"], datetime_tag
    )
    output_path_links = directory + output_file_links
    with pd.ExcelWriter(
        output_path_links, engine="openpyxl", mode="w"
    ) as writer:  # pylint: disable=abstract-class-instantiated
        links.to_excel(
            writer, header=True, index=None, sheet_name="Distance-based Rates per Link"
        )
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


# %%
# Generate links dataframe
def generate_links(df):
    links = pd.DataFrame()
    link_ids = df.LINK_ID.unique()
    for id in link_ids:
        df_slice = df[df["LINK_ID"] == id]
        new_row = pd.DataFrame()
        new_row["linkID"] = [id]
        new_row["linkLength"] = [
            round(
                (df_slice["DIST_KM"].iloc[-1] - df_slice["DIST_KM"].iloc[0]) * 0.621371,
                3,
            )
        ]
        new_row["FCR_JMI"] = df_slice["FCR_JS"].sum() / new_row["linkLength"]
        links = links.append(new_row)
    return links


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

# %%
# Data transformation settings
SETTINGS = {
    "INPUT_TYPE": "NONE",
    "INPUT_INDEX": "07",
    "OUTPUT_TYPE": "NONE",
    "OUTPUT_INDEX_LINKS": "08",
}

# %%
# Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data from Excel to a pandas dataframe
    df = load_from_Excel(vehicle, SETTINGS)
    # Split the dataframe based on 1-hour time slots
    df_split = df.groupby(pd.Grouper(key="DATETIME", freq="60T"))
    for tag, group in df_split:
        if not group.empty:
            print(tag, len(group))
            # Generate links dataframe
            links = generate_links(group)
            # Save the transformed data back to Excel file
            save_to_Excel(links, tag.strftime("%Y-%m-%d %I%p"), vehicle, SETTINGS)

# %%
