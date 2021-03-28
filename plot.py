""" File for containing every plotting function, to visualize the Recurrent Neural Networks' performance statistics """

import matplotlib.pyplot as plt


# TODO
def plot_avg_epoch_losses(losses: list, seq2seq_model: str, dataset: str):
    """ Plotting function for X number of epochs,
     where each X is a single value - the average loss over the given epoch
    """
    figure = plt.figure(figsize=(16, 12))
    plt.plot(losses)
    plt.ylabel("Cross-Entropy Loss")
    plt.xlabel("epoch")
    plt.title(f"Performance of {seq2seq_model} on {dataset}.")
