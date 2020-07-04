# %%
# Running Virginia Tech's Comprehensive Power-based Fuel Model (VTCPFM) on OBD data
# for Comparison with the output of our ANN models in Paper I based on the paper by Rakha et al. (2011)
# https://www.sciencedirect.com/science/article/pii/S1361920911000782
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required libraries
import pandas as pd
import geopy
from geopy.geocoders import GoogleV3, Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

# %%
# Load sample data from Excel to a pandas dataframe
def load_from_Excel(vehicle, sheet, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
        + settings["INPUT_TYPE"]
        + "/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["INPUT_TYPE"], settings["INPUT_INDEX"]
    )
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return df


# %%
# Save the predicted field back to Excel file
def save_to_excel(df, vehicle, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
        + settings["OUTPUT_TYPE"]
        + "/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["OUTPUT_TYPE"], settings["OUTPUT_INDEX"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


# %%
# General settings
pd.options.mode.chained_assignment = None
EXPERIMENTS = (
    "019 Hyundai Elantra GT 2019 (2.0L Auto)",
    "025 Chevrolet Captiva 2010 (2.4L Auto)",
    "027 Chevrolet Cruze 2011 (1.8L Manual)",
    "035 Kia Rio 2013 (1.6L Auto)",
)

# %%
# Model execution settings
SETTINGS = {
    "INPUT_TYPE": "ANN",
    "INPUT_INDEX": "18",
    "OUTPUT_TYPE": "ANN",
    "OUTPUT_INDEX": "20",
    "GOOGLE_API_KEY": "AIzaSyAz9W76DA5XDkd7ZNwa-5_D45ZCITU1CG8",
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
rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.1)
# For OpenStreetMap
# locator = Nominatim(user_agent="esn.mrd@gmail.com")
# rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.1)
tqdm.pandas()
for vehicle in EXPERIMENTS:
    # Load data from Excel to a pandas dataframe
    df = load_from_Excel(vehicle, "Cascaded", SETTINGS)
    specs = load_from_Excel(vehicle, "Sheet1", SETTINGS)
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
