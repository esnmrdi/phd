# %%
# Reverse Geo-coding of GPS Logs to Assign Link IDs to the Points for MOVES Analysis
# Ehsan Moradi, Ph.D. Candidate

# pylint: disable=abstract-class-instantiated

# %% [markdown]
# Load required libraries
import pandas as pd
import geopandas as gpd
import geopy
from geopy.geocoders import GoogleV3, Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm


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
# Save the reverse geo-coded data back to Excel file
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
    with pd.ExcelWriter(
        output_path, engine="openpyxl", mode="w"
    ) as writer:  # pylint: disable=abstract-class-instantiated
        df.to_excel(writer, header=True, index=None)
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


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
# Reverse Geo-Code Settings
SETTINGS = {
    "INPUT_TYPE": "NONE",
    "OUTPUT_TYPE": "NONE",
    "INPUT_INDEX": "01",
    "OUTPUT_INDEX": "02",
    "GOOGLE_API_KEY": "",
}

# %%
# Extracting information from the API response
def get_component(location, component_type):
    for component in location.raw["address_components"]:
        if component_type in component["types"]:
            return component["long_name"]


# %%
# Batch execution on all vehicles and their trips
# For Google Geocoding API
locator = GoogleV3(api_key=SETTINGS["GOOGLE_API_KEY"], timeout=100)
# For OpenStreetMap
# locator = Nominatim(user_agent="esn.mrd@gmail.com")
# For Google Geocoding API
rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.1)
# For OpenStreetMap
# rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.1)
tqdm.pandas()
for vehicle in EXPERIMENTS:
    # Load data from Excel to a pandas dataframe
    df = load_from_Excel(vehicle, SETTINGS)
    # Perform the reverse geo-coding process
    df["LOCATION"] = df["LAT"].astype(str) + ", " + df["LNG"].astype(str)
    df["LINK"] = [
        # For Google Geocoding API
        get_component(d[0], "route")
        for d in df["LOCATION"].progress_apply(rgeocode)
        # For OpenStreetMap
        # d.raw["address"]["road"] if (d.raw["address"] and d.raw["address"]["road"]) else None for d in df["LOCATION"].progress_apply(rgeocode)
    ]
    # df = df.assign(LINK_ID=(df["LINK"]).astype("category").cat.codes)

    # Save the predicted field back to Excel file
    save_to_excel(df, vehicle, SETTINGS)

# %%
