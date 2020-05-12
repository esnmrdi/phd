#%% [markdown]
# ## Convert Instant Fuel Rate to Joules per Second (KGS and JS)
# ### Ehsan Moradi, Ph.D. Candidate


#%% [markdown]
# ### Load required libraries
import pandas as pd
import matplotlib.pyplot as plt

#%% [markdown]
# ### Load data from Excel to a pandas dataframe
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


#%% [markdown]
# ### Data transformation for concentration fields
def convert_rates(df, settings):
    if (settings["CITY"] == "Montreal"):
        efficiency = settings["E10_EFFICIENCY"]
    else:
        efficiency = 1
    df["FCR_JS"] = (
        df["FCR_LH"]
        * efficiency
        * settings["GASOLINE_ENERGY_PER_LITER"]
        * (1 / 3600)
    )
    return df


#%% [markdown]
# ### Save the transformed data back to Excel file
def save_to_excel(df, vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["OUTPUT_TYPE"], settings["OUTPUT_INDEX"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(
            writer, sheet_name="Fuel Rate in JS", header=True, index=None
        )
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


#%% [markdown]
# ### General settings
plt.style.use("bmh")
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
        "024 Renault Logan 2010 (1.4 L Manual)",
        "025 Chevrolet Captiva 2010 (2.4L Auto)",
        "026 Nissan Versa 2013 (1.6L Auto)",
        "027 Chevrolet Cruze 2011 (1.8L Manual)",
    ),
}

#%% [markdown]
# ### Data transformation settings
SETTINGS = {
    "INPUT_TYPE": "NONE",
    "INPUT_INDEX": "04",
    "OUTPUT_TYPE": "NONE",
    "OUTPUT_INDEX": "07",
    "AIR_DENSITY": 1.2929,  # kilograms per cubic meter
    "E10_EFFICIENCY": 0.967,  # E10 gasoline has less energy production efficiency
    "GASOLINE_ENERGY_PER_LITER": 31536000,  # in joules
    "CITY": "Bucaramanga",
}

#%% [markdown]
# ### Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS[SETTINGS["CITY"]]:
    # Load data from Excel to a pandas dataframe
    df = load_from_Excel(vehicle, SETTINGS)
    # Data transformation for concentration fields
    df = convert_rates(df, SETTINGS)
    # Save the transformed data back to Excel file
    save_to_excel(df, vehicle, SETTINGS)

# %%
