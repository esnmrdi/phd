#%% [markdown]
# ## Retrieve Elevation from Google
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Loading required packages
import time
import pandas as pd
import json
import urllib
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource,
    CrosshairTool,
    HoverTool,
    PanTool,
    ResetTool,
    SaveTool,
    UndoTool,
    WheelZoomTool,
)

output_notebook()

#%% [markdown]
# ### General Settings
ELEVATION_BASE_URL = "https://maps.googleapis.com/maps/api/elevation/json"
API_KEY = "AIzaSyCmRuGS_wJfQf8vZ-CvGG2_MYNCbHBohiE"
EXPERIMENTS = {
    "009 Renault Logan 2014 (1.6L Manual)": [
        "12-08-2018 16.05.22",
        "12-08-2018 14.45.38",
        "12-08-2018 14.02.38",
        "12-08-2018 10.08.03",
    ],
    "010 JAC J5 2015 (1.8L Auto)": ["12-14-2018 13.57.40"],
    "011 JAC S5 2017 (2.0L TC Auto)": ["12-20-2018 13.27.13"],
    "012 IKCO Dena 2016 (1.65L Manual)": ["01-01-2019 19.11.14", "01-01-2019 19.38.52"],
    "013 Geely Emgrand7 2014 (1.8L Auto)": ["2019-01-03 18.22.12"],
    "014 Kia Cerato 2016 (2.0L Auto)": ["01-15-2019 21.36.00"],
    "015 VW Jetta 2016 (1.4L TC Auto)": [
        "02-01-2019 18.27.23",
        "02-02-2019 14.21.01",
        "02-02-2019 17.50.19",
        "02-02-2019 19.23.32",
    ],
    "016 Hyundai Sonata Sport 2019 (2.4L Auto)": [
        "02-03-2019 10.09.28",
        "02-03-2019 10.39.10",
        "02-03-2019 10.48.03",
        "02-03-2019 11.00.02",
        "02-03-2019 12.04.20",
        "02-03-2019 12.43.04",
        "02-03-2019 13.34.30",
        "02-03-2019 14.52.39",
        "02-03-2019 17.05.33",
        "02-03-2019 17.34.21",
        "02-03-2019 18.24.35",
        "02-03-2019 18.55.57",
        "02-03-2019 19.03.35",
    ],
    "017 Chevrolet Trax 2019 (1.4L TC Auto)": [
        "02-08-2019 09.53.09",
        "02-08-2019 10.06.13",
        "02-08-2019 10.27.57",
        "02-08-2019 11.43.39",
        "02-08-2019 12.57.19",
        "02-08-2019 13.54.58",
        "02-08-2019 14.22.02",
        "02-08-2019 16.11.36",
        "02-08-2019 16.28.29",
        "02-08-2019 18.12.30",
        "02-08-2019 18.33.36",
        "02-08-2019 19.24.46",
        "02-08-2019 19.47.30",
    ],
    "018 Hyundai Azera 2006 (3.8L Auto)": [
        "02-16-2019 12.46.10",
        "02-16-2019 13.38.44",
        "02-16-2019 16.05.39",
        "02-16-2019 18.01.48",
        "02-17-2019 19.37.44",
        "02-17-2019 21.40.10",
    ],
    "019 Hyundai Elantra GT 2019 (2.0L Auto)": [
        "03-01-2019 10.16.17",
        "03-01-2019 13.48.33",
        "03-01-2019 14.46.18",
        "03-01-2019 15.38.30",
        "03-01-2019 21.29.00",
    ],
    "020 Honda Civic 2014 (1.8L Auto)": [
        "04-12-2019 16.48.33",
        "04-13-2019 16.34.17",
        "04-15-2019 20.45.31",
        "04-16-2019 10.27.06",
        "04-16-2019 19.43.24",
        "04-17-2019 19.11.19",
        "04-18-2019 08.47.24",
        "04-18-2019 20.44.51",
        "04-19-2019 11.14.57",
        "04-19-2019 16.27.09",
    ],
    "021 Chevrolet N300 2014 (1.2L Manual)": [
        "04-08-2019 10.35.17",
        "04-09-2019 12.11.16",
        "04-09-2019 13.35.57",
        "04-09-2019 14.39.25",
        "04-10-2019 12.40.29",
        "04-10-2019 19.45.02",
        "04-11-2019 08.32.28",
        "04-11-2019 13.01.58",
    ],
    "022 Chevrolet Spark GT 2012 (1.2L Manual)": [
        "04-17-2019 08.17.08",
        "04-17-2019 08.47.45",
        "04-17-2019 11.30.19",
        "04-17-2019 15.06.27",
    ],
    "023 Mazda 2 2012 (1.4L Auto)": [
        "04-20-2019 08.06.24",
        "04-20-2019 08.56.49",
        "04-20-2019 09.13.32",
        "04-20-2019 10.01.06",
        "04-20-2019 14.32.16",
        "04-20-2019 15.10.33",
    ],
    "024 Renault Logan 2010 (1.4 L Manual)": ["04-23-2019 15.09.17"],
    "025 Chevrolet Captiva 2010 (2.4L Auto)": [
        "04-30-2019 10.46.01",
        "04-30-2019 13.06.58",
        "04-30-2019 14.46.57",
        "04-30-2019 21.29.46",
        "04-30-2019 22.37.56",
        "05-01-2019 02.17.09",
        "05-01-2019 18.35.36",
        "05-01-2019 21.30.06",
        "05-01-2019 22.27.11",
        "05-02-2019 06.43.17",
        "05-02-2019 06.54.09",
    ],
    "026 Nissan Versa 2013 (1.6L Auto)": [
        "04-02-2019 11.42.13",
        "04-02-2019 15.24.06",
        "04-02-2019 17.50.30",
        "04-07-2019 16.30.50",
        "04-07-2019 17.57.37",
        "05-02-2019 23.53.28",
        "05-03-2019 15.05.16",
        "05-03-2019 15.44.30",
    ],
    "027 Chevrolet Cruze 2011 (1.8L Manual)": [
        "05-14-2019 06.28.45",
        "05-14-2019 12.34.24",
    ],
    "028 Nissan Sentra 2019 (1.8L Auto)": [
        "05-30-2019 15.19.16",
        "05-30-2019 17.18.39",
        "05-30-2019 18.21.43",
        "05-30-2019 18.55.28",
    ],
}
VEHICLE = "009 Renault Logan 2014 (1.6L Manual)"
TRIP = 2

#%% [markdown]
# ### Method to pass coordinates via an HTTP request and retrieve elevations
def get_elevations(path):
    request = urllib.request.urlopen(
        ELEVATION_BASE_URL + "?locations=" + path + "&key=" + API_KEY
    )
    try:
        results = json.load(request).get("results")
        print(results)
        if len(results) > 0:
            elevations = [result.get("elevation") for result in results]
            return elevations
        else:
            print("HTTP GET Request failed.")
    except ValueError:
        print("JSON decode failed: {}".format(str(request)))


#%% [markdown]
# ### Loading data from Excel to a pandas dataframe
directory = "./Field Experiments/Veepeak/" + VEHICLE + "/Processed/"
input_file = VEHICLE + ".xlsx"
input_path = directory + input_file
df = pd.read_excel(input_path, sheet_name=EXPERIMENTS[VEHICLE][TRIP], header=4)

#%% [markdown]
# ### Retrieve elevation data for trip coordinates from Google
segment_size = 512
no_of_segments = int(len(df) / segment_size)
df_segments = [
    df.iloc[i * segment_size : (i + 1) * segment_size].copy()
    for i in range(no_of_segments + 1)
]
elevations = []
for segment in df_segments:
    locations = segment["LAT"].map(str) + "," + segment["LNG"].map(str)
    path = "|".join(locations)
    segment_elevations = get_elevations(path)
    elevations += segment_elevations
    time.sleep(1)
df["ELEV_M"] = elevations

#%%
df["ELEV_ALT_DIFF_M"] = df["ELEV_M"] - df["ALT_M"]


#%% [markdown]
# ### Plotting the GPS altitute vs. Google elevation
source = ColumnDataSource(df)
datetime = EXPERIMENTS[VEHICLE][TRIP]
hover = HoverTool(
    tooltips=[
        ("index", "$index"),
        ("Distance (km)", "@DIST_KM"),
        ("Altitude (m)", "@ALT_M"),
        ("Elevation (m)", "@ELEV_M"),
        ("Difference (m)", "@ELEV_ALT_DIFF_M"),
    ]
)

TOOLS = [
    hover,
    CrosshairTool(),
    PanTool(),
    ResetTool(),
    SaveTool(),
    UndoTool(),
    WheelZoomTool(),
]
p = figure(
    width=790,
    height=395,
    # x_axis_type="datetime",
    title=VEHICLE[4:] + " on " + datetime[:10] + " @ " + datetime[11:],
    toolbar_location="above",
    tools=TOOLS,
)
p.line(
    x="DIST_KM",
    y="ALT_M",
    line_color="blue",
    line_width=2,
    legend="Altitude (m)",
    muted_alpha=0.1,
    source=source,
)
p.line(
    x="DIST_KM",
    y="ELEV_M",
    line_color="orange",
    line_width=2,
    legend="Elevation (m)",
    muted_alpha=0.1,
    source=source,
)
p.xaxis.axis_label = "Distance (km)"
p.legend.location = "top_left"
p.legend.click_policy = "mute"
show(p)
