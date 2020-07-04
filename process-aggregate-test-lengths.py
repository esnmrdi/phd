# %%
# Calculate total experiment lengths (distance and duration) for each city
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required libraries
import pandas as pd

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
    df_dict = {}
    for name, sheet in sheets_dict.items():
        df_dict[name] = sheet
    return df_dict


# %%
# General settings
pd.options.mode.chained_assignment = None
EXPERIMENTS = {
    "Tehran": (
        "009 Renault Logan 2014 (1.6L Manual)",
        "010 JAC J5 2015 (1.8L Auto)",
        "011 JAC S5 2017 (2.0L TC Auto)",
        "012 IKCO Dena 2016 (1.65L Manual)",
        "013 Geely Emgrand7 2014 (1.8L Auto)",
        "014 Kia Cerato 2016 (2.0L Auto)",
    ),
    "Montreal": (
        "015 VW Jetta 2016 (1.4L TC Auto)",
        "016 Hyundai Sonata Sport 2019 (2.4L Auto)",
        "017 Chevrolet Trax 2019 (1.4L TC Auto)",
        "018 Hyundai Azera 2006 (3.8L Auto)",
        "019 Hyundai Elantra GT 2019 (2.0L Auto)",
        "020 Honda Civic 2014 (1.8L Auto)",
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
    "Bucaramanga": (
        "021 Chevrolet N300 2014 (1.2L Manual)",
        "022 Chevrolet Spark GT 2012 (1.2L Manual)",
        "023 Mazda 2 2012 (1.4L Auto)",
        "024 Renault Logan 2010 (1.4L Manual)",
        "025 Chevrolet Captiva 2010 (2.4L Auto)",
        "026 Nissan Versa 2013 (1.6L Auto)",
        "027 Chevrolet Cruze 2011 (1.8L Manual)",
    ),
}

# %%
# Data transformation settings
SETTINGS = {
    "INPUT_TYPE": "NONE",
    "INPUT_INDEX": "01",
    "CITY": "Montreal",
}

# %%
# Batch execution on all vehicles and their trips
distance = 0
duration = 0
for vehicle in EXPERIMENTS[SETTINGS["CITY"]]:
    print(vehicle)
    # Load data from Excel to a pandas dataframe
    df_dict = load_from_Excel(vehicle, SETTINGS)
    for key, df in df_dict.items():
        if "DIST_KM" in df:
            distance += df["DIST_KM"].iloc[-1]
        if "DURATION_MIN" in df:
            duration += df["DURATION_MIN"].iloc[-1]
print(SETTINGS["CITY"])
print("---------------")
print("Total Distance: {}".format(distance))
print("Total Duration: {}".format(duration))

# %%
