import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from build_vocabulary import SOS_token

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class MogLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, dropout=0, bidirectional=False, mogrify_steps=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mogrifier_lstm_layer1 = MogLSTMCell(input_size, hidden_size, mogrify_steps)
        self.mogrifier_lstm_layer2 = MogLSTMCell(hidden_size, hidden_size, mogrify_steps)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_seq, hidden=None):
        """
        :param input_seq: Default expected Input tensor of shape (seq_len, batch, input_size)
        :param hidden: Hidden state and cell state. Not provided for encoder, only for decoder.
        """
        batch_size = input_seq.shape[1]
        seq_len = input_seq[0]
        h1, c1 = [torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)]
        h2, c2 = [torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)]
        hidden_states = []
        outputs = []
        for step in range(seq_len):
            x = self.drop(input_seq[step, :, :])
            h1, c1 = self.mogrifier_lstm_layer1(x, (h1, c1))  # dropout in default LSTM PyTorch doc. is true
            h2, c2 = self.mogrifier_lstm_layer2(h1, (h2, c2))
            out = self.fc(self.drop(h2))
            hidden_states.append(h2.unsqueeze(1))
            outputs.append(out.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=0)  # (seq_len, batch, input_size)
        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, input_size)

        return outputs, hidden_states


class MogLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps=5):
        super(MogLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q --> update x
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r --> update h

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i + 1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states: Tuple):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, gate=None, bidirectional=False, vocab_size=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # - Initialize GRU/LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        # - dropout is set to 0 only in case of 1 hidden layer (according to PyTorch docs at least 2 are needed)
        if gate == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)
        elif gate == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                               dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)
        elif gate == "MogLSTM":
            self.rnn = MogLSTM(hidden_size, hidden_size, vocab_size, dropout=dropout)
        else:
            raise ValueError("The gated RNN's type has not been given."
                             "Possible options are: 'GRU', 'LSTM', 'MogLSTM'.")

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        :param input_seq: Padded input sequence of shape (10,64) - before embedding
        :param input_lengths: A 1D tensor containing the length of each sentence within the batch in decreasing order.
                              E.g. [10,10,10,9,9,8...,3]
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)  # input shape: (10, 64), output shape: (10, 64, 500)
        print("Input_lengths:", input_lengths)
        print("Input_lengths size:", input_lengths.size())
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        print("PACKED:", packed)

        # Forward pass through the network
        if self.rnn._get_name() in ["LSTM", "MogLSTM"]:
            outputs, (hidden, cell_state) = self.rnn(packed, hidden)
            hidden = (hidden, cell_state)
        else:
            outputs, hidden = self.rnn(packed, hidden)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # print(f"Encoder RNN type:{self.rnn._get_name()}, bidirectional:{self.rnn.bidirectional}")
        # print("Encoder Output shape:", outputs.size())
        # try:
        #     print("Encoder Hidden size:", hidden.size())
        # except AttributeError:
        #     print("Encoder Hidden[0] size:", hidden[0].size())
        #     print("Encoder Hidden[1] size:", hidden[1].size())

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
                 n_layers=1, dropout=0.1, gate=None, bidirectional=False):
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
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)
        elif self.gate == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                               dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)
        else:
            raise ValueError("The gated RNN's type has not been given."
                             "Possible options are: 'GRU', 'LSTM'.")

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size + bidirectional * hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through bidirectional GRU
        # rnn_output, hidden = self.rnn(embedded, last_hidden)
        if self.gate == "GRU":
            rnn_output, hidden = self.rnn(embedded, last_hidden)
        elif self.gate == "LSTM":
            rnn_output, (hidden, cell_state) = self.rnn(embedded, last_hidden)
            hidden = (hidden, cell_state)
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

        # # Return output and final hidden state
        # print(f"Decoder RNN type:{self.rnn._get_name()}, bidirectional:{self.rnn.bidirectional}")
        # print("Decoder Input_step:", input_step.size())
        # print("Decoder Input shape:", encoder_outputs.size())
        # try:
        #     print("Decoder Hidden size:", last_hidden.size())
        # except AttributeError:
        #     print("Decoder Hidden[0] size:", last_hidden[0].size())
        #     print("Decoder Hidden[1] size:", last_hidden[1].size())

        return output, hidden