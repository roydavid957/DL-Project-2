import torch
import os

from build_vocabulary import Voc
from torch import nn
from torch import optim


def save_seq2seq(encoder, decoder, encoder_name, decoder_name, encoder_optimizer, decoder_optimizer,
                 losses, scores, voc, embedding, DROPOUT, CLIP, LR):
    """ Function to save optimally trained (e.g. using early stopping) encoders, decoders, and optimizers altogether """
    folder_path = "RNN_models"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path,
                             f"{encoder_name}{'2' if encoder.bidirectional else '1'}{decoder_name}_d{DROPOUT}"
                             f"_gc{CLIP}_lr{LR}.pt")
    torch.save({
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'losses': losses,
        'scores': scores,
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
    }, file_path)


def load_encoder(checkpoint, encoder_cls,
                 HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT, encoder_name, bidirectional):
    """ Load encoder model in evaluation mode (e.g. for validation, testing or chatting) """
    model = encoder_cls(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT,
                        gate=encoder_name, bidirectional=bidirectional)
    model.load_state_dict(checkpoint['en'])
    model.eval()
    return model


def load_decoder(checkpoint, decoder_cls,
                 attn_model, embedding, HIDDEN_SIZE, VOC_SIZE, DECODER_N_LAYERS, DROPOUT, decoder_name):
    """ Load decoder model in evaluation mode (e.g. for validation, testing or chatting) """
    model = decoder_cls(attn_model, embedding, HIDDEN_SIZE, VOC_SIZE, DECODER_N_LAYERS, DROPOUT, gate=decoder_name)
    model.load_state_dict(checkpoint['de'])
    model.eval()
    return model


def load_optim(checkpoint, optim_key, optim_alg, model):
    optim_sd = checkpoint[optim_key]
    if optim_alg == "ADAM":
        optimizer = optim.Adam(model.parameters())
    elif optim_alg == "SGD":
        # TODO: Check if SGD learning rate also gets overwritten when load_state_dict is executed
        optimizer = optim.SGD(model.parameters(), 0)
    else:
        raise ValueError("Wrong optimizer name has been given.")
    optimizer.load_state_dict(optim_sd)
    return optimizer


def load_embedding(checkpoint, HIDDEN_SIZE):
    voc = load_voc(checkpoint)
    embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)
    return embedding


def load_voc(checkpoint):
    voc = Voc()
    voc.__dict__ = (checkpoint['voc_dict'])
    return voc

