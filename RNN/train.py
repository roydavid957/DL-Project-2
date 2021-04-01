import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os
import argparse
from typing import Union, Dict

from format_data import datafiles
from build_vocabulary import loadPrepareData, trimRareWords, batch2TrainData, SOS_token, MIN_COUNT
from model import EncoderRNN, LuongAttnDecoderRNN
from serialization import save_seq2seq


def maskNLLLoss(inp, target, mask):
    """
    The function calculates the average negative log likelihood of the elements
     that correspond to a 1 in the mask tensor.
    :param inp: Padded output of the decoder.
    :param target: Teacher value.
    :param mask:
    :return:
    """
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()  # mean loss of the batch
    loss = loss.to(device)
    return loss, nTotal.item()


def iterate_batches(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, batch_size, clip, phase, phase_name):
    """ Wrapper function of forward_decoder(), train() and estim_gen_error() for iterating over batches """

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # !!! Lengths for rnn packing should always be on the cpu !!!
    lengths = lengths.to("cpu")  # length of each sentence in the batch

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # ------------------------------------------------------------------------------------------------------------------
    # NOTE: The forward pass through encoder is handled in the outer scope, since it is the same for
    # all 3 of: training/validation/testing
    # ------------------------------------------------------------------------------------------------------------------
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    # encoder_outputs.size())  --> GRU/LSTM/MogLSTM: torch.Size([10, 64, 500])
    # encoder_hidden.size()) --> GRU: torch.Size([4, 64, 500])
    # encoder_hidden[0].size())  --> LSTM/MogLSTM: torch.Size([4, 64, 500])
    # encoder_hidden[1].size())  --> LSTM/MogLSTM: torch.Size([4, 64, 500])

    # -------------------------------
    # SETTING UP DECODER
    # -------------------------------
    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    name = decoder.rnn._get_name()
    # Set initial decoder hidden state to the encoder's final hidden state
    if name == "LSTM":
        decoder_hidden = tuple([encoder_state[:decoder.n_layers] for encoder_state in encoder_hidden])
        # decoder_hidden[0].shape = torch.Size([2, 64, 500]), decoder_hidden[1].shape = torch.Size([2, 64, 500])
    elif name == "MogLSTM":
        decoder_hidden = encoder_hidden  # tensors are the same dim as for LSTM, but the logic is handled elsewhere
    else:  # GRU
        decoder_hidden = encoder_hidden[:decoder.n_layers]

    if name in ["LSTM", "MogLSTM"] and not isinstance(decoder_hidden, tuple):
        raise ValueError("'decoder_hidden' was supposed to be assigned a 'tuple', assignment has failed.")
    elif name == "GRU" and not isinstance(decoder_hidden, torch.Tensor):
        raise ValueError("'decoder_hidden' was supposed to be assigned a 'torch.Tensor', assignment has failed.")

    # Nested functions that manipulate variables of run()
    # ------------------------------------------------------------------------------------------------------------------
    def forward_decoder(step: int):
        """ Forward pass through decoder and calculate loss thereof. Used by training/validation/testing identically."""
        nonlocal loss
        nonlocal n_totals
        nonlocal decoder_input
        nonlocal decoder_hidden

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # No teacher forcing: next input is decoder's own current output
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        # Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[step], mask[step])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    def exec_train():  # renamed from train() to exec_train() to avoid collision with PyTorch's train()
        """ Training function used to learn the parameters of the model(s). """
        # Ensure dropout layers are in train mode
        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # Forward batch of sequences through decoder one time step at a time
        for step in range(max_target_len):
            forward_decoder(step)
        # Perform backpropagation
        loss.backward()
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()
        return sum(print_losses) / n_totals

    def estim_gen_error():
        """ Function for estimating generalization error either on the validation or on the test set,
         thus both the hyperparameter choices and the reported generalization errors are obtained from here. """
        # Forward batch of sequences through decoder one time step at a time
        for step in range(max_target_len):
            forward_decoder(step)
        return sum(print_losses) / n_totals
    # ------------------------------------------------------------------------------------------------------------------

    if phase_name == "train":
        return exec_train()
    elif phase_name in ["val", "test"]:
        return estim_gen_error()


def run(encoder: EncoderRNN, decoder: LuongAttnDecoderRNN,
        encoder_optimizer, decoder_optimizer, epoch_num: int, batch_size: int, clip: float, phase: Dict[str, dict]):

    def run_phase(curr_phase, phase_name):
        losses_curr_epoch = []
        for curr_batch in range(1, curr_phase["bp_ep"] + 1):
            training_batch = batch2TrainData(curr_phase["voc"],
                                             [curr_phase["pairs"][j + curr_batch * j] for j in range(batch_size)])
            input_variable, lengths, target_variable, mask, max_target_len = training_batch
            # Run a TRAINING iteration with batch
            loss = iterate_batches(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                                   decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, phase, phase_name)
            losses_curr_epoch.append(loss)
            print(
                f"[{phase_name.upper()}]"
                f" Epoch: {curr_epoch + 1}"
                f" Percent complete: {round(curr_batch / curr_phase['bp_ep'] * 100, 1)}%;"
                f" Average loss: {round(loss, 5)}"
            )
        return losses_curr_epoch

    # either train+val or solely test
    for k in phase.keys():
        phase[k]["bp_ep"] = round(len(phase[k]["pairs"]) // batch_size)
        phase[k]["losses"] = []  # losses over all epochs
        print(f"Number of total pairs used for [{k.upper()}]:", len(phase[k]["pairs"]))
        print(f"Number of batches used for a [{k.upper()}] epoch:", phase[k]["bp_ep"])

    print(f"Training for {epoch_num} epochs...")
    for curr_epoch in range(epoch_num):
        # TODO: ADD EARLY STOPPING
        # Checking for early stopping criterion in each epoch
        # if losses_all_epochs[curr_epoch]
        for k in phase.keys():
            if k == "train":
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()
            phase[k]["losses"].append(run_phase(phase[k], k))


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
