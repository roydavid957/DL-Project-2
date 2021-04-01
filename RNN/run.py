import argparse
import random
import os
import pprint

import torch
from torch import nn
from torch import optim
from pathos.multiprocessing import ProcessingPool as Pool

from format_data import datafiles
from build_vocabulary import trimRareWords, loadPrepareData, MIN_COUNT, Voc
from train import run
from model import EncoderRNN, LuongAttnDecoderRNN
from serialization import save_seq2seq, load_encoder, load_decoder, load_voc, load_embedding, \
                          load_optim
from chat import GreedySearchDecoder, chat


def write_results(run_mode, encoder, encoder_name, decoder_name, dropout, clip, lr, losses):
    os.makedirs("txt_results", exist_ok=True)
    with open(f"txt_results{os.path.sep}"
              f"{run_mode}_"
              f"{encoder_name}{'2' if encoder.bidirectional else '1'}{decoder_name}_"
              f"d{dropout}_gc{clip}_lr{lr}.txt", "w") as output_file:
        for loss in losses:
            output_file.write(f"{str(round(loss, 5))}\n")


def main():
    if run_mode == 'train':
        phase = {
            "train": {},
            "val": {}
        }
        # Load/Assemble voc and pairs
        phase["train"]["voc"], phase["train"]["pairs"] = loadPrepareData(datafiles["train"])
        phase["val"]["voc"], phase["val"]["pairs"] = loadPrepareData(datafiles["val"])
        # Trim voc and pairs
        phase["train"]["pairs"] = trimRareWords(phase["train"]["voc"], phase["train"]["pairs"], MIN_COUNT)
        phase["val"]["pairs"] = trimRareWords(phase["val"]["voc"], phase["val"]["pairs"], MIN_COUNT)

        # Shuffle both sets ONCE before the entire training
        random.seed(1)  # seed can be any number
        random.shuffle(phase["train"]["pairs"])
        random.shuffle(phase["val"]["pairs"])

        print('Building training set encoder and decoder ...')
        # Initialize word embeddings for both encoder and decoder
        embedding = nn.Embedding(phase["train"]["voc"].num_words, HIDDEN_SIZE).to(device)

        # Initialize encoder & decoder models
        encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT, gate=encoder_name,
                             bidirectional=BIDIRECTION)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, HIDDEN_SIZE,
                                      phase["train"]["voc"].num_words, DECODER_N_LAYERS, DROPOUT, gate=decoder_name)

        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        print('Models built and ready to go!')

        # Initialize optimizers
        print('Building optimizers ...')
        if args.get('optimizer') == "ADAM":
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)
        elif args.get('optimizer') == "SGD":
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)
        else:
            raise ValueError("Wrong optimizer type has been given as an argument.")

        # If you have cuda, configure cuda to call
        for optimizer in [encoder_optimizer, decoder_optimizer]:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        print("Starting Training!")
        run(encoder, decoder, encoder_optimizer, decoder_optimizer, EPOCH_NUM, BATCH_SIZE, CLIP, phase)
        avg_losses_train = [sum(epoch) / len(epoch) for epoch in phase["train"]["losses"]]
        avg_losses_val = [sum(epoch) / len(epoch) for epoch in phase["val"]["losses"]]

        # save_seq2seq(encoder, decoder, encoder_name, decoder_name, encoder_optimizer, decoder_optimizer,
        #              avg_losses_train, voc, embedding, DROPOUT, CLIP, LR, DECODER_LR)
        write_results("train", encoder, encoder_name, decoder_name, DROPOUT, CLIP, LR, avg_losses_train)
        write_results("val", encoder, encoder_name, decoder_name, DROPOUT, CLIP, LR, avg_losses_val)

    else:
        # Loading basic objects needed for all 3 of validation, testing and chatting
        checkpoint = torch.load(args.get('model_path'))
        embedding = load_embedding(checkpoint, HIDDEN_SIZE)
        encoder = load_encoder(checkpoint, EncoderRNN, HIDDEN_SIZE, embedding,
                               ENCODER_N_LAYERS, DROPOUT, encoder_name, BIDIRECTION)
        voc = load_voc(checkpoint)
        decoder = load_decoder(checkpoint, LuongAttnDecoderRNN,
                               attn_model, embedding, HIDDEN_SIZE, voc.num_words, DECODER_N_LAYERS, DROPOUT, decoder_name)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        if run_mode == "test":
            pass
        elif run_mode == "chat":
            # Initialize search module
            searcher = GreedySearchDecoder(encoder, decoder)
            chat(searcher, voc)

        else:
            raise ValueError("Wrong run_mode has been given, options: ['train', 'val', 'test', 'chat']")


# Experiments' parameters
parser = argparse.ArgumentParser()
# ------------------------------------------------------------------------------------------------------------
# Basics -- Uppercase arguments
# ------------------------------------------------------------------------------------------------------------
parser.add_argument('-M', '--run_mode', help="Type of run mode, options: ['train', 'val', 'test', 'chat']",
                    type=str, default=None)
parser.add_argument('-P', '--model_path',
                    help="RELATIVE path to the model to be used in any run mode different from 'train'",
                    type=str, default=None, )
parser.add_argument('-E', '--encoder', help="Type of encoder, options: ['GRU', 'LSTM', 'MogLSTM']",
                    type=str, default=None)
parser.add_argument('-ED', '--encoder_direction', help="Number of encoder directions, options: [1, 2]",
                    type=int, default=None)
parser.add_argument('-D', '--decoder', help="Type of decoder, options: ['GRU', 'LSTM', 'MogLSTM']",
                    type=str, default=None)
parser.add_argument('-O', '--optimizer', help="Type of optimizer, options: ['ADAM', 'SGD']",
                    type=str, default=None)
parser.add_argument('-EN', '--epoch_num', help="Number of epochs to run the training for",
                    type=str, default=100)
parser.add_argument('-ES', '--early_stopping', help="Whether to use early stopping or not",
                    type=bool, default=False)
# ------------------------------------------------------------------------------------------------------------
# Grid-search (non-dependent of RNN type) -- Lowercase arguments
# ------------------------------------------------------------------------------------------------------------
parser.add_argument('-d', '--dropout', help="Value of dropout, can be any float",
                    type=float, default=0.1)
parser.add_argument('-gc', '--gradient_clipping', help='Value of gradient clipping',
                    type=int, default=50)
parser.add_argument('-lr', '--lr',
                    help="Learning rate of optimization algorithms",
                    type=float, default=0.0001)
# TODO: Finish this
# ------------------------------------------------------------------------------------------------------------
# Grid-search -- MogLSTM specific parameters -- Lowercase arguments starting with 'm'
# ------------------------------------------------------------------------------------------------------------
# parser.add_argument('-md', '--moglstm_dropout', help="value of dropout, can be any float",
#                     type=float, default=None)
# ...

# Get all arguments as a dictionary
args = vars(parser.parse_args())

print(f"\n{'*' * 40}")
print(f"[RUN_MODE]: {args['run_mode']}")
print(f"[MODEL_PATH]: {args['model_path']}")
print(f"{'*' * 40}\n")

encoder_name = args.get('encoder')
decoder_name = args.get('decoder')
run_mode = args.get('run_mode')
EPOCH_NUM = int(args.get('epoch_num'))

# Get device object
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Configure attention model
attn_model = 'dot'

# Base params
HIDDEN_SIZE = 500  # Number of dimensions of the embedding; number of features in a hidden state
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
BATCH_SIZE = 64
BIDIRECTION = True

# Hyperparameters
CLIP = float(args.get('gradient_clipping'))
LR = float(args.get('lr'))
DROPOUT = float(args.get('dropout'))

main()

