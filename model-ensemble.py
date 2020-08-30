# %%
# Ensemble Learning Applied on the Outputs of RNN Modeling
# Ehsan Moradi, Ph.D. Candidate

# %%
# Import required libraries
import keras
from keras import backend
from keras.layers import Dense, SimpleRNN, GRU, LSTM
from keras import Sequential, utils
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import openpyxl