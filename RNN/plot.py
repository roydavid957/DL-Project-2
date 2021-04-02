""" File for containing every plotting function, to visualize the Recurrent Neural Networks' performance statistics """

import os
import matplotlib.pyplot as plt
from typing import List


def read_data(path):
    file = os.path.abspath(f".\\{path}")
    model_name = os.path.basename(file).split("_")[1]
    with open(file, "r") as file_obj:
        data = file_obj.readlines()
    data = [float(data.split("\n")[0]) for data in data]
    return model_name, data


def plot_avg_epoch_losses(losses: List[list], seq2seq_model: str, file_name=None):
    """ Plotting function for X number of epochs,
     where each X is a single value - the average loss over the given epoch
    """
    fs = 20
    figure = plt.figure(figsize=(16, 12))
    for loss in losses:
        plt.plot(loss)
    plt.legend(["Training loss", "Validation loss"], fontsize=fs)
    plt.ylabel("Cross-Entropy Loss", fontsize=fs)
    plt.xlabel("epoch", fontsize=fs)
    plt.title(f"Performance of {seq2seq_model}", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    figure.savefig(f"{seq2seq_model}_{file_name}.png")


def read_and_plot(train_path, val_path, file_ext):
    model_name, train_data = read_data(train_path)
    _, val_data = read_data(val_path)
    plot_avg_epoch_losses([train_data, val_data], model_name, file_ext)

os.chdir("txt_results\\Exp set - joint voc, hidden 300")

# -------------------------------------------------- GRU --------------------------------------------------
# ------- Gradient clipping -------
read_and_plot("train_GRU2GRU_d0.1_gc1.0_lr0.0001.txt",
              "val_GRU2GRU_d0.1_gc1.0_lr0.0001.txt",
              "gc1.0")

read_and_plot("train_GRU2GRU_d0.1_gc1.0_lr0.0001.txt",
              "val_GRU2GRU_d0.1_gc2.0_lr0.0001.txt",
              "gc2.0")

# ------- DROPOUT -------
read_and_plot("train_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "d0.1")

read_and_plot("train_GRU2GRU_d0.2_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.2_gc50.0_lr0.0001.txt",
              "d0.2")

read_and_plot("train_GRU2GRU_d0.3_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.3_gc50.0_lr0.0001.txt",
              "d0.3")

# ------- Learning rate -------
read_and_plot("train_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "lr0.0001")

read_and_plot("train_GRU2GRU_d0.1_gc50.0_lr0.001.txt",
              "val_GRU2GRU_d0.1_gc50.0_lr0.001.txt",
              "lr0.001")

# -------------------------------------------------- LSTM --------------------------------------------------
# ------- Gradient clipping -------
read_and_plot("train_LSTM2LSTM_d0.1_gc1.0_lr0.0001.txt",
              "val_LSTM2LSTM_d0.1_gc1.0_lr0.0001.txt",
              "gc1.0")

read_and_plot("train_LSTM2LSTM_d0.1_gc1.0_lr0.0001.txt",
              "val_LSTM2LSTM_d0.1_gc2.0_lr0.0001.txt",
              "gc2.0")

# ------- DROPOUT -------
# read_and_plot("train_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
#               "val_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
#               "d0.1")
#
# read_and_plot("train_LSTM2LSTM_d0.2_gc50.0_lr0.0001.txt",
#               "val_LSTM2LSTM_d0.2_gc50.0_lr0.0001.txt",
#               "d0.2")
#
# read_and_plot("train_LSTM2LSTM_d0.3_gc50.0_lr0.0001.txt",
#               "val_LSTM2LSTM_d0.3_gc50.0_lr0.0001.txt",
#               "d0.3")

# ------- Learning rate -------
read_and_plot("train_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
              "val_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
              "lr0.0001")

read_and_plot("train_LSTM2LSTM_d0.1_gc50.0_lr0.001.txt",
              "val_LSTM2LSTM_d0.1_gc50.0_lr0.001.txt",
              "lr0.001")

