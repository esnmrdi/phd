#%% [markdown]
# ## Convert Emission Concentration to Mass Rate (Kg/hr)
# ### Ehsan Moradi, Ph.D. Candidate


#%% [markdown]
# ### Load required libraries
import pandas as pd


#%% [markdown]
# ### Load data from Excel to a pandas dataframe
def load_from_Excel(vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/3DATX parSYNC Plus/"
        + vehicle
        + "/Processed/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["input_type"], settings["input_index"]
    )
    input_path = directory + input_file
    sheets_dict = pd.read_excel(input_path, sheet_name=None, header=0)
    df = pd.DataFrame()
    for _, sheet in sheets_dict.items():
        df = df.append(sheet)
    return df


#%% [markdown]
# ### Data transformation for concentration fields
def convert_to_mass_rate(df, settings):
    df["CO2_PPM"] = 10000 * df["CO2_PERC"]
    for gas in ("CO2", "NO2", "NO"):
        df[gas + "ـMGM3"] = df[gas + "_PPM"] * settings["MOLECULAR_WEIGHT"]["AIR"] / settings["MOLECULAR_WEIGHT"][gas])


#%% [markdown]
# ### General settings
plt.style.use("bmh")
pd.options.mode.chained_assignment = None
EXPERIMENTS = [
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
]

EXPERIMENTS = ["015 VW Jetta 2016 (1.4L TC Auto)"]

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
    # Load data corresponding to vehicle and device into a dataframe
    df = load_from_Excel(vehicle, "3DATX parSYNC Plus", SETTINGS)
    print(df.head())

# %%
