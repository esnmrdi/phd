#%% [markdown]
# ## Joining OBD and PEMS data
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ## Load required libraries
import numpy as np
import pandas as pd

#%% [markdown]
# ### Loading data from Excel to pandas dataframe
def load_from_Excel(vehicle, settings):
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["input_type"], settings["input_index"]
    )
    for device in settings["devices"]:
        input_path = (
            "../Field Experiments/"
            + device
            + "/"
            + vehicle
            + "/Processed/"
            + input_file
        )
        sheets_dict = pd.read_excel(input_path, sheet_name=None, header=4)
        df_dict = {}
        for sheet_name, df in sheets_dict.items():
            if device not in df_dict:
                df_dict[device] = {}
            df_dict[device][sheet_name] = df
    return df_dict


#%% [markdown]
# ### General Settings
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
    "040 Chevrolet Spark 2019 (Auto)",
)

#%% [markdown]
# ### Grade calculation settings
SETTINGS = {
    "input_type": "NONE",
    "output_type": "NONE",
    "input_index": "00",
    "output_index": "01",
    "devices": ("Veepeak", "3DATX parSYNC Plus")
}
