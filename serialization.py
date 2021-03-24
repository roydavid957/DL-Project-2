import torch
import os


def save_seq2seq(encoder, decoder, encoder_name, decoder_name, encoder_optimizer, decoder_optimizer):
    """ Function to save optimally trained (e.g. using early stopping) encoders, decoders, and optimizers altogether """
    folder_path = "RNN_models"
    os.makedirs(folder_path, exist_ok=True)
    direction = "Bi" if encoder.bidirectional else "Uni"
    file_path = os.path.join(folder_path, f"{encoder_name}_{direction} - {decoder_name}.pt")
    torch.save({
        f"Encoder_state_dict": encoder.state_dict(),
        f"Decoder_state_dict": decoder.state_dict(),
        f"optimizer{encoder_name}_state_dict": encoder_optimizer.state_dict(),
        f"optimizer{decoder_name}_state_dict": decoder_optimizer.state_dict()
    }, file_path)


def load_encoder(file_path, encoder_cls,
                 HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT, encoder_name, bidirectional):
    """ Load encoder model in evaluation mode (e.g. for validation or testing) """
    model = encoder_cls(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT,
                        gate=encoder_name, bidirectional=bidirectional)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['Encoder_state_dict'])
    model.eval()
    return model


def load_decoder(file_path, decoder_cls,
                 attn_model, embedding, HIDDEN_SIZE, VOC_SIZE, DECODER_N_LAYERS, DROPOUT, decoder_name):
    """ Load encoder model in evaluation mode (e.g. for validation or testing) """
    model = decoder_cls(attn_model, embedding, HIDDEN_SIZE, VOC_SIZE, DECODER_N_LAYERS, DROPOUT, gate=decoder_name)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['Decoder_state_dict'])
    model.eval()
    return model
