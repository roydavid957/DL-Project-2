""" File for containing every plotting function, to visualize the Recurrent Neural Networks' performance statistics """

import os
import matplotlib.pyplot as plt
import re
import argparse
from typing import List


def read_data(path):
    file = os.path.abspath(f".\\{path}")
    model_name = re.findall(r'[A-Z]+\d[A-Z]+', os.path.basename(file))[0]
    metric = os.path.basename(file).split("_")[0]
    with open(file, "r") as file_obj:
        data = file_obj.readlines()
    data = [float(data.split("\n")[0]) for data in data]
    return model_name, data, metric


def plot_results(seq2seq_model: str, data: list, metric: str, file_ext=None):
    """ Plotting function for X number of epochs,
     where each X is a single value - the average loss/bleu score over the given epoch
    """
    fs = 20
    figure = plt.figure(figsize=(16, 12))
    plt.plot(data)
    if metric == "loss":
        plt.ylabel("Cross-Entropy Loss", fontsize=fs)
        plt.title(f"Performance of {seq2seq_model}", fontsize=fs)
        plt.legend(["Training loss"], fontsize=fs)
    else:
        plt.ylabel("BLEU Score", fontsize=fs)
        plt.title(f"Individual 1-Gram BLEU score of {seq2seq_model}", fontsize=fs)
        plt.legend(["Training score"], fontsize=fs)
    plt.xlabel("epoch", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    figure.savefig(f"{seq2seq_model}_{file_ext}.png")


def read_and_plot(train_path, file_ext):
    model_name, train_data, metric = read_data(train_path)
    plot_results(model_name, train_data, metric, file_ext)


# ---- Example uses of 'read_and_plot' -----
# read_and_plot("loss_train_GRU2GRU_d0.1_gc1.0_lr0.001.txt", "loss_gc1.0")
# read_and_plot("bleu_train_GRU2GRU_d0.1_gc1.0_lr0.001.txt", "bleu_gc1.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help="Path to txt file containing the experiment data",
                        type=str, default=None)
    parser.add_argument('-n', '--name', help="Additional string to the filename,"
                                             " marking the experiment type and values")
    args = vars(parser.parse_args())
    read_and_plot(args.get("path"), args.get("name"))



