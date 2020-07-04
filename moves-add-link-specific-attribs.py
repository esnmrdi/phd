# %%
# Add link-specific attributes to MOVES final output
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required libraries
import pandas as pd
import numpy as np
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
    ) as writer:
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
# General Settings
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
# Batch execution on all vehicles and their trips (Veepeak data)
join_final = pd.DataFrame()
for vehicle in EXPERIMENTS:
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/MOVES/Analysis/On Validity of Fuel, CO2, NOx, and PM Predictions by US EPA's MOVES/input/"
        + vehicle
        + "/"
    )
    target_links_filenames = list_filenames(directory, vehicle[:4] + "Links")
    target_schedule_filenames = list_filenames(
        directory, vehicle[:4] + "Link Drive Schedules"
    )
    for (links_filename, schedule_filename) in zip(
        target_links_filenames, target_schedule_filenames
    ):
        schedule_df = load_from_Excel(directory, schedule_filename)
        left = schedule_df.groupby(["linkID"]).mean()
        right = schedule_df.groupby(["linkID"]).std()
        join_1 = pd.merge_ordered(left, right, how="inner", on="linkID")
        links_df = load_from_Excel(directory, links_filename)
        left = join_1
        right = links_df
        join_2 = pd.merge_ordered(left, right, how="inner", on="linkID")
        join_2["MOVESScenarioID"] = (
            vehicle[:3]
            + "_"
            + "".join(links_filename[-20:-10].split("-"))
            + "_"
            + links_filename[-9:-5].lower()
        )
        join_final = join_final.append(join_2)
new_df = pd.DataFrame(np.repeat(join_final.values, 11, axis=0))
new_df.columns = join_final.columns
directory = "../../../Google Drive/Academia/PhD Thesis/MOVES/Analysis/On Validity of Fuel, CO2, NOx, and PM Predictions by US EPA's MOVES/Output/"
save_to_Excel(new_df, directory, "complementary.xlsx")


# %%
