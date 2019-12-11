#%% [markdown]
# ## 1D Convolutional Neural Network for Fuel Consumption and Emissions Rate Estimation
# ### Ehsan Moradi, Ph.D. Candidate

#%% [markdown]
# ### Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

#%% [markdown]
# ### Display training progress
class ReportProgress(tf.keras.callbacks.Callback):
    def __init__(self, df, test_split_ratio, n_epochs):
        self.df = df
        self.test_split_ratio = test_split_ratio
        self.n_epochs = n_epochs

    def on_train_begin(self, logs):
        n_examples = len(self.df)
        n_train = int((1 - self.test_split_ratio) * n_examples)
        print(
            "Training started on {0} out of {1} available examples.".format(
                n_train, n_examples
            )
        )

    def on_epoch_end(self, epoch, logs):
        if epoch % 20 == 0 and epoch != 0 and epoch != self.n_epochs:
            print("{0} out of {1} epochs completed.".format(epoch, self.n_epochs))

    def on_train_end(self, logs):
        print("Training finished.")