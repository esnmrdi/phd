# %%
# Ramer-Douglas-Peucker Algorithm
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required libraries
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["axes.grid"] = True

# %%
# Ramer-Douglas-Peucker Algorithm
def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1])
            - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d


def rdp(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results


# %%
# General settings
VEHICLE = "009 Renault Logan 2014 (1.6L Manual)"
LABELS = {
    "FCR_GS": "Fuel Consumption Rate (g/s)",
    "SPD_SI": "Speed (m/s)",
    "ACC_SI": "Acceleration (m/s2)",
    "CORR_GRADE": "Grade (deg)",
    "ALT": "Altitude (m)",
    "STATE": "Engine State",
}

# %%
# Load data from Excel into a pandas dataframe
dir = r"/Users/ehsan/Dropbox/Academia/PhD Thesis/Field Experiments/Veepeak"
file = r"/009 Renault Logan 2014 (1.6L Manual)/Processed/009 Renault Logan 2014 (1.6L Manual).xlsx"
path = dir + file
df = pd.read_excel(path, sheet_name="Prepared for Modeling")

# %%
# Apply RDP on GPS altitude data to reduce the number of vertices
points = list(zip(df["order"][1500:2000], df["alt"][1500:2000]))
reduced = np.array(rdp(points, epsilon=2))

# %%
# Plot original altitudes curve vs. reduced curve
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5)
gs.tight_layout(fig)
ax = fig.add_subplot(gs[0])
ax.set_xlabel("Time (s)")
ax.set_ylabel(LABELS["ALT"])
ax.grid(color="k", linestyle=":", linewidth=1, alpha=0.5)
ax.plot(df["alt"][1500:2000])
# ax.plot(reduced[:, 1])
