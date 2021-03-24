import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple, List, Dict

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers, dropout, gate=None, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.bidirectional = bidirectional

        # - Initialize GRU/LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        # - dropout is set to 0 only in case of 1 hidden layer (according to PyTorch docs at least 2 are needed)
        if gate == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=self.bidirectional)
        elif gate == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                               dropout=(0 if n_layers == 1 else dropout), bidirectional=self.bidirectional)
        else:
            raise ValueError("The gated RNN's type has not been given."
                             "Possible options are: 'GRU', 'LSTM'.")

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        :param input_seq: Padded input sequence of shape (10,64) - before embedding
        :param input_lengths: A 1D tensor containing the length of each sentence within the batch in decreasing order.
                              E.g. [10,10,10,9,9,8...,3], thus size is (64).
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)  # input shape: (10, 64), output shape: (10, 64, 500)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # Forward pass through the network
        if self.rnn._get_name() == "LSTM":
            outputs, (hidden, cell_state) = self.rnn(packed, hidden)
            hidden = (hidden, cell_state)
        else:
            outputs, hidden = self.rnn(packed, hidden)

        # # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size,
                 n_layers, dropout, gate=None):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gate = gate

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)

        if self.gate == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        elif self.gate == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        else:
            raise ValueError("The gated RNN's type has not been given."
                             "Possible options are: 'GRU', 'LSTM'.")

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        :param input_step: torch.Size([1, 64])
        :param last_hidden: Tuple of tensors containing hidden and cell state: torch.Size([2, 64, 500]), torch.Size([2, 64, 500])
        :param encoder_outputs: torch.Size([10, 64, 500])
        :return:
        """
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional GRU/LSTM
        if self.gate == "LSTM":
            # rnn_output size: torch.Size([1, 64, 500])
            # hidden size: torch.Size([2, 64, 500])
            # cell_state size: torch.Size([2, 64, 500])
            rnn_output, (hidden, cell_state) = self.rnn(embedded, last_hidden)
            hidden = (hidden, cell_state)
        else:
            rnn_output, hidden = self.rnn(embedded, last_hidden)

        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden
