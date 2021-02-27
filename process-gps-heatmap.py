# %% [markdown]
# Aggregate GPS Trajectories for Building a Heatmap
# Ehsan Moradi, Ph.D. Candidate


# %% [markdown]
# Load required libraries
import pandas as pd


# %% [markdown]
# Load trajectory data from Excel to a pandas dataframe
def load_data_from_Excel(experiments, settings):
    df = pd.DataFrame()
    for vehicle in experiments[settings["CITY"]]:
        input_path = (
            "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/Aggregated GPS Trajectories/"
            + vehicle
            + ".xlsx"
        )
        sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
        for _, sheet in sheets_dict.items():
            df = df.append(sheet)
    return df


# %% [markdown]
# Save the aggregated dataframe back to a single Excel file
def save_to_excel(df, settings):
    output_path = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/Aggregated GPS Trajectories - "
        + settings["CITY"]
        + ".xlsx"
    )
    with pd.ExcelWriter(
        output_path, engine="openpyxl", mode="w"
    ) as writer:
        df.to_excel(writer, header=True, index=None)
    print("Data is saved to Excel successfully!")
    return None


# %% [markdown]
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

SETTINGS = {"CITY": "Bucaramanga"}


# %% [markdown]
# Batch execution on all vehicles and their trips
# Load trajectory data from Excel to a pandas dataframe
df = load_data_from_Excel(EXPERIMENTS, SETTINGS)
# Save the aggregated dataframe back to a single Excel file
save_to_excel(df, SETTINGS)
