""" File for containing every plotting function, to visualize the Recurrent Neural Networks' performance statistics """

import os
import matplotlib.pyplot as plt
import re
from typing import List


def read_data(path):
    file = os.path.abspath(f".\\{path}")
    model_name = re.findall(r'[A-Z]+\d[A-Z]+', os.path.basename(file))[0]
    with open(file, "r") as file_obj:
        data = file_obj.readlines()
    data = [float(data.split("\n")[0]) for data in data]
    return model_name, data


def plot_cross_entropy(losses: List[list], seq2seq_model: str, file_name=None):
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


def plot_bleu_score(scores: List[list], seq2seq_model: str, file_name=None):
    """ Plotting function for X number of epochs,
     where each X is a single value - the average bleu score the given epoch
    """
    fs = 20
    figure = plt.figure(figsize=(16, 12))
    for score in scores:
        plt.plot(score)
    plt.legend(["Training score", "Validation score"], fontsize=fs)
    plt.ylabel("BLEU Score", fontsize=fs)
    plt.xlabel("epoch", fontsize=fs)
    plt.title(f"Individual 1-Gram BLEU score of {seq2seq_model}", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    figure.savefig(f"{seq2seq_model}_{file_name}.png")


def read_and_plot(train_path, val_path, file_ext, metric):
    model_name, train_data = read_data(train_path)
    _, val_data = read_data(val_path)
    if metric == "cross-entropy":
        plot_cross_entropy([train_data, val_data], model_name, file_ext)
    elif metric == "bleu":
        plot_bleu_score([train_data, val_data], model_name, file_ext)
    else:
        raise ValueError("Wrong metric parameter was given. Possible options: ['cross-entropy', 'bleu']")


os.chdir("txt_results\\Exp set - joint voc, hidden 300")

# -------------------------------------------------- GRU --------------------------------------------------
# ------- Gradient clipping -------
read_and_plot("loss_train_GRU2GRU_d0.1_gc1.0_lr0.0001.txt",
              "loss_val_GRU2GRU_d0.1_gc1.0_lr0.0001.txt",
              "loss_gc1.0", "cross-entropy")

read_and_plot("train_GRU2GRU_d0.1_gc1.0_lr0.0001.txt",
              "val_GRU2GRU_d0.1_gc2.0_lr0.0001.txt",
              "gc2.0", "cross-entropy")

# ------- DROPOUT -------
read_and_plot("train_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "d0.1", "cross-entropy")

read_and_plot("train_GRU2GRU_d0.2_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.2_gc50.0_lr0.0001.txt",
              "d0.2", "cross-entropy")

read_and_plot("train_GRU2GRU_d0.3_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.3_gc50.0_lr0.0001.txt",
              "d0.3", "cross-entropy")

# ------- Learning rate -------
read_and_plot("train_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "val_GRU2GRU_d0.1_gc50.0_lr0.0001.txt",
              "lr0.0001", "cross-entropy")

read_and_plot("train_GRU2GRU_d0.1_gc50.0_lr0.001.txt",
              "val_GRU2GRU_d0.1_gc50.0_lr0.001.txt",
              "lr0.001", "cross-entropy")

# -------------------------------------------------- LSTM --------------------------------------------------
# ------- Gradient clipping -------
read_and_plot("train_LSTM2LSTM_d0.1_gc1.0_lr0.0001.txt",
              "val_LSTM2LSTM_d0.1_gc1.0_lr0.0001.txt",
              "gc1.0", "cross-entropy")

read_and_plot("train_LSTM2LSTM_d0.1_gc1.0_lr0.0001.txt",
              "val_LSTM2LSTM_d0.1_gc2.0_lr0.0001.txt",
              "gc2.0", "cross-entropy")

# ------- DROPOUT -------
# read_and_plot("train_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
#               "val_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
#               "d0.1", "cross-entropy)
#
# read_and_plot("train_LSTM2LSTM_d0.2_gc50.0_lr0.0001.txt",
#               "val_LSTM2LSTM_d0.2_gc50.0_lr0.0001.txt",
#               "d0.2", "cross-entropy)
#
# read_and_plot("train_LSTM2LSTM_d0.3_gc50.0_lr0.0001.txt",
#               "val_LSTM2LSTM_d0.3_gc50.0_lr0.0001.txt",
#               "d0.3", "cross-entropy)

# ------- Learning rate -------
read_and_plot("train_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
              "val_LSTM2LSTM_d0.1_gc50.0_lr0.0001.txt",
              "lr0.0001", "cross-entropy")

read_and_plot("train_LSTM2LSTM_d0.1_gc50.0_lr0.001.txt",
              "val_LSTM2LSTM_d0.1_gc50.0_lr0.001.txt",
              "lr0.001", "cross-entropy")

