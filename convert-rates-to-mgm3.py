#%% [markdown]
# ## Convert Instant Emission Concentrations to Milligrams per Cubic Meter (mGM3)
# ### Ehsan Moradi, Ph.D. Candidate


#%% [markdown]
# ### Load required libraries
import pandas as pd
import matplotlib.pyplot as plt

#%% [markdown]
# ### Load data from Excel to a pandas dataframe
def load_from_Excel(vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/3DATX parSYNC Plus/"
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


#%% [markdown]
# ### Data transformation for concentration fields
def convert_rates(df, settings):
    df["PM_mGM3"] = df["PM_MGM3"] / 1000
    df["CO2_PPM"] = (10 ** 6) * df["CO2_PERC"]
    for gas in ("CO2", "NO2", "NO"):
        # https://www.gastec.co.jp/en/technology/knowledge/concentration/
        df[gas + "_mGM3"] = (
            df[gas + "_PPM"]
            * (settings["MOLECULAR_WEIGHT"][gas] / 22.4)
            * (273 / (273 + df["INT_AIR_TMP_C"]))
            * (10 * df["BAROMETRIC_KPA"] / 1013)
        )
    return df


#%% [markdown]
# ### Save the transformed data back to Excel file
def save_to_excel(df, vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/3DATX parSYNC Plus/"
        + vehicle
        + "/Processed/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["OUTPUT_TYPE"], settings["OUTPUT_INDEX"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(
            writer, sheet_name="Concentrations in mGM3", header=True, index=None
        )
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


#%% [markdown]
# ### General settings
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
# ### Data transformation settings
SETTINGS = {
    "INPUT_TYPE": "NONE",
    "INPUT_INDEX": "02",
    "OUTPUT_TYPE": "NONE",
    "OUTPUT_INDEX": "03",
    "MOLECULAR_WEIGHT": {  # in grams/mol
        "CO2": 44.01,
        "NO2": 46.01,
        "NO": 30.01,
        "AIR": 28.97,
    },
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data from Excel to a pandas dataframe
    df = load_from_Excel(vehicle, SETTINGS)
    # Data transformation for concentration fields
    df = convert_rates(df, SETTINGS)
    # Save the transformed data back to Excel file
    save_to_excel(df, vehicle, SETTINGS)

# %%
