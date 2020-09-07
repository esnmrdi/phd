# %%
# Running Virginia Tech's Comprehensive Power-based Fuel Model
# to Compare its output with our best Cascaded ANN results
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# Load sample data from Excel to a pandas dataframe
def load_from_Excel(vehicle, order, sheet, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
        + settings["INPUT_" + order + "_TYPE"]
        + "/"
    )
    input_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["INPUT_" + order + "_TYPE"], settings["INPUT_" + order + "_INDEX"]
    )
    input_path = directory + input_file
    df = pd.read_excel(input_path, sheet_name=sheet, header=0)
    return df


# %%
# Save the predicted field back to Excel file
def save_to_excel(df, vehicle, order, settings):
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/Field Experiments/Veepeak/"
        + vehicle
        + "/Processed/"
        + settings["OUTPUT_" + order + "_TYPE"]
        + "/"
    )
    output_file = vehicle + " - {0} - {1}.xlsx".format(
        settings["OUTPUT_" + order + "_TYPE"], settings["OUTPUT_" + order + "_INDEX"]
    )
    output_path = directory + output_file
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, header=True, index=None)
    print("{0} -> Data is saved to Excel successfully!".format(vehicle))
    return None


# %%
# Caclulate the resistance force (sum of aerodynamic, rolling, and grade resistance forces)
def calculate_resistance_force(df, specs):
    df["RESISTANCE_FORCE"] = (
        (specs["RHO"] / 25.92)
        * specs["C_D"]
        * (1 - 0.085 * df["ALT_M"] / 1000)
        * specs["A_F"]
        * (df["SPD_KH_ORIG"] ** 2)
        + 9.8066
        * specs["WEIGHT"]
        * (specs["C_R"] / 1000)
        * (specs["C_1"] * df["SPD_KH_ORIG"] + specs["C_2"])
        + 9.8066 * specs["WEIGHT"] * np.sin(np.radians(df["GRADE_DEG_ORIG"]))
    )
    return df


# %%
# Calculate power
def calculate_power(df, specs):
    df["POWER"] = (
        (df["RESISTANCE_FORCE"] + 1.04 * specs["WEIGHT"] * df["ACC_MS2_ORIG"])
        / (3600 * 0.9)
    ) * df["SPD_KH_ORIG"]
    df["POWER2"] = df["POWER"] ** 2
    return df


# %%
# Caculate other required variables
def calculate_variables(df, specs):
    variables = {}
    T = df["ROAD_TYPE"].groupby(df["ROAD_TYPE"]).agg(["count"]).reset_index()
    P = df["POWER"].groupby(df["ROAD_TYPE"]).agg(["sum"]).reset_index()
    P2 = df["POWER2"].groupby(df["ROAD_TYPE"]).agg(["sum"]).reset_index()
    df["FCR_LS"] = df["True FCR (L/H)"] / 3600
    F = df["FCR_LS"].groupby(df["ROAD_TYPE"]).agg(["sum"]).reset_index()
    W = df["RPM"].groupby(df["ROAD_TYPE"]).agg(["sum"]).reset_index()
    T_CITY = T.loc[T["ROAD_TYPE"] == "City", "count"].iloc[0]
    T_HWY = T.loc[T["ROAD_TYPE"] == "Highway", "count"].iloc[0]
    P_CITY = P.loc[P["ROAD_TYPE"] == "City", "sum"].iloc[0]
    P_HWY = P.loc[P["ROAD_TYPE"] == "Highway", "sum"].iloc[0]
    P2_CITY = P2.loc[P2["ROAD_TYPE"] == "City", "sum"].iloc[0]
    P2_HWY = P2.loc[P2["ROAD_TYPE"] == "Highway", "sum"].iloc[0]
    F_CITY = F.loc[F["ROAD_TYPE"] == "City", "sum"].iloc[0]
    F_HWY = F.loc[F["ROAD_TYPE"] == "Highway", "sum"].iloc[0]
    W_CITY = W.loc[F["ROAD_TYPE"] == "City", "sum"].iloc[0]
    W_HWY = W.loc[F["ROAD_TYPE"] == "Highway", "sum"].iloc[0]
    ALPHA0 = max(
        [
            (specs["P_MFO"] * specs["W_IDLE"] * specs["D"])
            / (22164 * specs["Q_FINAL"] * specs["N"]),
            (
                (F_CITY - F_HWY * (P_CITY / P_HWY))
                - (10 ** -6) * (P2_CITY - P2_HWY * (P_CITY / P_HWY))
            )
            / (T_CITY - T_HWY * (P_CITY / P_HWY)),
        ]
    )
    ALPHA2 = max(
        (
            (F_CITY - F_HWY * (P_CITY / P_HWY))
            - (T_CITY - T_HWY * (P_CITY / P_HWY)) * ALPHA0
        )
        / (P2_CITY - P2_HWY * (P_CITY / P_HWY)),
        10 ** -6,
    )
    ALPHA1 = (F_HWY - T_HWY * ALPHA0 - P2_HWY * ALPHA2) / P_HWY
    BETA0 = max(
        [
            (specs["P_MFO"] * specs["D"]) / (22164 * specs["Q_FINAL"] * specs["N"]),
            (
                (F_CITY - F_HWY * (P_CITY / P_HWY))
                - (10 ** -6) * (P2_CITY - P2_HWY * (P_CITY / P_HWY))
            )
            / (W_CITY - W_HWY * (P_CITY / P_HWY)),
        ]
    )
    BETA2 = max(
        (
            (F_CITY - F_HWY * (P_CITY / P_HWY))
            - (W_CITY - W_HWY * (P_CITY / P_HWY)) * BETA0
        )
        / (P2_CITY - P2_HWY * (P_CITY / P_HWY)),
        10 ** -6,
    )
    BETA1 = (
        ((F_CITY - W_CITY * BETA0 - P2_CITY * BETA2) / P_CITY)
        + ((F_HWY - W_HWY * BETA0 - P2_HWY * BETA2) / P_HWY)
    ) / 2
    variables = {
        "T_CITY": T_CITY,
        "T_HWY": T_HWY,
        "P_CITY": P_CITY,
        "P_HWY": P_HWY,
        "P2_CITY": P2_CITY,
        "P2_HWY": P2_HWY,
        "F_CITY": F_CITY,
        "F_HWY": F_HWY,
        "W_CITY": W_CITY,
        "W_HWY": W_HWY,
        "ALPHA0": ALPHA0,
        "ALPHA1": ALPHA1,
        "ALPHA2": ALPHA2,
        "BETA0": BETA0,
        "BETA1": BETA1,
        "BETA2": BETA2,
    }
    return pd.DataFrame(variables, index=[0,])


# %%
# Apply piecewise condition of VTCPFM model (type I)
def power_condition_1(row, alpha0, alpha1, alpha2):
    if row["POWER"] >= 0:
        val = (
            variables["ALPHA0"]
            + variables["ALPHA1"] * row["POWER"]
            + variables["ALPHA2"] * row["POWER2"]
        )
    else:
        val = variables["ALPHA0"]
    val *= 3600
    return val


# %%
# Apply piecewise condition of VTCPFM model (type II)
def power_condition_2(row, beta0, beta1, beta2, specs):
    if row["POWER"] >= 0:
        val = (
            variables["BETA0"] * row["RPM"]
            + variables["BETA1"] * row["POWER"]
            + variables["BETA2"] * row["POWER2"]
        )
    else:
        val = variables["BETA0"] * specs["W_IDLE"]
    val *= 3600
    return val


# %%
# Cacluate fuel consumption rate
def calculate_fuel_consumption_rate(df, variables, specs):
    df["VTCPFM-I FCR (L/H)"] = df.apply(
        power_condition_1,
        args=(variables["ALPHA0"], variables["ALPHA1"], variables["ALPHA2"]),
        axis=1,
    )
    df["VTCPFM-II FCR (L/H)"] = df.apply(
        power_condition_2,
        args=(variables["BETA0"], variables["BETA1"], variables["BETA2"], specs),
        axis=1,
    )
    return df


# %%
# General settings
pd.options.mode.chained_assignment = None
EXPERIMENTS = (
    "019 Hyundai Elantra GT 2019 (2.0L Auto)",
    "025 Chevrolet Captiva 2010 (2.4L Auto)",
    "027 Chevrolet Cruze 2011 (1.8L Manual)",
)

# %%
# Model execution settings
SETTINGS = {
    "INPUT_01_TYPE": "ANN",
    "INPUT_01_INDEX": "20",
    "INPUT_02_TYPE": "VTCPFM",
    "INPUT_02_INDEX": "SPECS",
    "INPUT_03_TYPE": "VTCPFM",
    "INPUT_03_INDEX": "VARIABLES",
    "OUTPUT_01_TYPE": "VTCPFM",
    "OUTPUT_01_INDEX": "VARIABLES",
    "OUTPUT_02_TYPE": "VTCPFM",
    "OUTPUT_02_INDEX": "COMPARE",
}


# %%
# Batch execution on all vehicles and their trips
for vehicle in EXPERIMENTS:
    # Load data from Excel to a pandas dataframe
    df = load_from_Excel(vehicle, "01", "Sheet1", SETTINGS)
    specs = load_from_Excel(vehicle, "02", "Sheet1", SETTINGS).iloc[0].to_dict()
    variables = load_from_Excel(vehicle, "03", "Sheet1", SETTINGS).iloc[0].to_dict()
    df = calculate_resistance_force(df, specs)
    df = calculate_power(df, specs)
    variables = calculate_variables(df, specs)
    df = calculate_fuel_consumption_rate(df, variables, specs)
    save_to_excel(variables, vehicle, "01", SETTINGS)
    save_to_excel(df, vehicle, "02", SETTINGS)


# %%
