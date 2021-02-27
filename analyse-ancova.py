# %% [markdown]
# Analysis of Covariance (ANCOVA)
# Ehsan Moradi, Ph.D. Candidate


# %%
# Load required librarie
import pandas as pd
import pingouin as pg


# %%
# Load data from Excel to a pandas dataframe
def load_sample_from_Excel():
    directory = (
        "../../../Google Drive/Academia/PhD Thesis/"
        + "Charts, Tables, Forms, Flowcharts, Spreadsheets/"
    )
    input_file = "Paper I - SVR and ANN Results.xlsx"
    input_path = directory + input_file
    sheets_dict = pd.read_excel(
        input_path, sheet_name=["SVR - ANCOVA", "ANN - ANCOVA"], header=0
    )
    df_svr = sheets_dict["SVR - ANCOVA"]
    df_ann = sheets_dict["ANN - ANCOVA"]
    return df_svr, df_ann


# %%
# Perform the ANCOVA
df_svr, df_ann = load_sample_from_Excel()
# stats.normaltest(df_svr["AGE"])
pg.ancova(data=df_svr, dv="SCORE", covar="ENGINE_DISPLACEMENT", between="CAR_SEGMENT")
