import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os
from typing import Union

from build_vocabulary import voc, pairs, batch2TrainData, SOS_token
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
                    encoder_optimizer, decoder_optimizer, batch_size, clip, dataset_type):
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
        decoder_hidden = encoder_hidden
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

    if dataset_type == "training":
        return exec_train()
    elif dataset_type in ["validation", "test"]:
        return estim_gen_error()


def run(voc, pairs,
        encoder: EncoderRNN, decoder: LuongAttnDecoderRNN,
        encoder_optimizer, decoder_optimizer, epoch_num: int, batch_size: int, clip: float, dataset_type: str):
    # Shuffle dataset ONCE before the entire training (according to DL book)
    random.shuffle(pairs)

    batch_per_epoch = round(len(pairs) // batch_size)
    print("Number of total pairs used for training:", len(pairs))
    print("Number of batches used for an epoch:", batch_per_epoch)
    print("Training...")
    for curr_epoch in range(epoch_num):

        # TODO: ADD EARLY STOPPING
        # Checking for early stopping criterion in each epoch
        # if losses_all_epochs[curr_epoch]

        losses_curr_epoch = []
        for curr_batch in range(1, batch_per_epoch + 1):
            training_batch = batch2TrainData(voc, [pairs[j + curr_batch * j] for j in range(batch_size)])
            input_variable, lengths, target_variable, mask, max_target_len = training_batch
            # Run a training iteration with batch
            loss = iterate_batches(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                                   decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, dataset_type)
            losses_curr_epoch.append(loss)
            print(
                f"Epoch: {curr_epoch + 1} Percent complete: {round(curr_batch / batch_per_epoch * 100, 1)}%"
                f"; Average loss: {round(loss, 5)}"
            )
        losses_all_epochs.append(losses_curr_epoch)


if __name__ == "__main__":

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    encoder_name = "MogLSTM"
    decoder_name = "MogLSTM"

    # Choose dataset
    dataset_type = "training"
    # dataset_type = "validation"
    # dataset_type = "test"

    # Hyperparameters
    HIDDEN_SIZE = 500  # Number of dimensions of the embedding, number of features in a hidden state
    ENCODER_N_LAYERS = 2
    DECODER_N_LAYERS = 2
    DROPOUT = 0.1
    BATCH_SIZE = 64

    # Configure training/optimization
    CLIP = 50.0
    TEACHER_FORCING_RATIO = 1.0
    LR = 0.0001  # Learning rate
    DECODER_LR = 5.0
    EPOCH_NUM = 100
    PRINT_EVERY = 1
    random.seed(1)  # seed can be any number

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT, gate=encoder_name, bidirectional=True)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, HIDDEN_SIZE,
                                  voc.num_words, DECODER_N_LAYERS, DROPOUT, gate=decoder_name)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LR)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # --------------------------
    # Run training iterations
    # --------------------------
    print("Starting Training!")
    losses_all_epochs = []
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()
    run(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               EPOCH_NUM, BATCH_SIZE, CLIP, dataset_type)
    avg_losses_all_epochs = [sum(epoch)/len(epoch) for epoch in losses_all_epochs]

    # Save models, write results into txt
    save_seq2seq(encoder, decoder, encoder_name, decoder_name, encoder_optimizer, decoder_optimizer)
    os.makedirs("txt_results", exist_ok=True)
    with open(f"txt_results{os.path.sep}"
              f"{encoder_name}_{'Bi' if encoder.bidirectional else 'Uni'} - {decoder_name}.txt", "w") as output_file:
        for avg_epoch_loss in avg_losses_all_epochs:
            output_file.write(f"{str(round(avg_epoch_loss, 5))}\n")


