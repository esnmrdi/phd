# %%
# Update link drive shedule and links files for MOVES analyses
# This is an extra step because of grade calculation error we had
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
    df_dict = {}
    for name, sheet in sheets_dict.items():
        df_dict[name] = sheet
    return df_dict


# %%
# Save the aggregated result to an Excel file
def save_to_Excel(df_dict, directory, output_file):
    output_path = directory + output_file
    with pd.ExcelWriter(
        output_path, engine="openpyxl", mode="w"
    ) as writer:
        for name, sheet in df_dict.items():
            sheet.to_excel(writer, header=True, index=None, sheet_name=name)
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
# General settings
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
for vehicle in EXPERIMENTS:
    origin_directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
    )
    target_directory = (
        "../../../Google Drive/Academia/PhD Thesis/MOVES/Analysis/On Validity of Fuel, CO2, NOx, and PM Predictions by US EPA's MOVES/Input/"
        + vehicle
        + "/"
    )

    origin_schedule_filenames = list_filenames(
        origin_directory, vehicle + " - NONE - 05"
    )
    target_schedule_filenames = list_filenames(
        target_directory, vehicle[:4] + "Link Drive Schedules"
    )
    for (origin_filename, target_filename) in zip(
        origin_schedule_filenames, target_schedule_filenames
    ):
        schedule_df_dict = load_from_Excel(origin_directory, origin_filename)
        save_to_Excel(schedule_df_dict, target_directory, target_filename)

    origin_links_filenames = list_filenames(origin_directory, vehicle + " - NONE - 06")
    target_links_filenames = list_filenames(target_directory, vehicle[:4] + "Links")
    for (origin_filename, target_filename) in zip(
        origin_links_filenames, target_links_filenames
    ):
        origin_links_df_dict = load_from_Excel(origin_directory, origin_filename)
        target_links_df_dict = load_from_Excel(target_directory, target_filename)
        target_links_df_dict["link"]["linkID"] = origin_links_df_dict["link"]["linkID"]
        target_links_df_dict["link"]["linkLength"] = origin_links_df_dict["link"][
            "linkLength"
        ]
        target_links_df_dict["link"]["linkAvgSpeed"] = origin_links_df_dict["link"][
            "linkAvgSpeed"
        ]
        target_links_df_dict["link"]["linkAvgGrade"] = origin_links_df_dict["link"][
            "linkAvgGrade"
        ]
        save_to_Excel(target_links_df_dict, target_directory, target_filename)

# %%
