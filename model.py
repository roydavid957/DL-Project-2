import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple, List, Dict

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


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

    def forward(self, x, states):
        ht, ct = states
        x, ht, ct = [old.to(device) for old in [x, ht, ct]]
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct


class MogLSTM_BiDir(nn.Module):
    """ Bidirectional Mogrifier LSTM, can be used for Encoder only """

    def __init__(self, input_size, hidden_size, dropout=0, mogrify_steps=5, cell_num=2):
        super().__init__()
        self.bidirectional = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fwd_layers = nn.ModuleList([MogLSTMCell(input_size, hidden_size, mogrify_steps) for _ in range(cell_num)])
        self.bwd_layers = nn.ModuleList([MogLSTMCell(input_size, hidden_size, mogrify_steps) for _ in range(cell_num)])
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def init_states(batch_size, hidden_size):
        """ Function for creating zero-valued hidden_state & cell_state tensors list """
        return [torch.zeros(batch_size, hidden_size),
                torch.zeros(batch_size, hidden_size)]

    def forward(self, input_seq, hidden=None):
        """
        :param input_seq:
        :param hidden: Hidden state and cell state. Not used for encoder, only for decoder.
        """
        batch_size = input_seq.shape[1]
        seq_len = input_seq.shape[0]

        fwd_h1, fwd_c1 = self.init_states(batch_size, self.hidden_size)
        fwd_h2, fwd_c2 = self.init_states(batch_size, self.hidden_size)
        bwd_h1, bwd_c1 = self.init_states(batch_size, self.hidden_size)
        bwd_h2, bwd_c2 = self.init_states(batch_size, self.hidden_size)

        outputs = []
        for step in range(seq_len):
            fwd_x = self.drop(input_seq[step, :])
            fwd_h1, fwd_c1 = self.fwd_layers[0](fwd_x, (fwd_h1, fwd_c1))
            fwd_h2, fwd_c2 = self.fwd_layers[1](fwd_h1, (fwd_h2, fwd_c2))

            bwd_x = self.drop(input_seq[seq_len - (step + 1), :])
            bwd_h1, bwd_c1 = self.bwd_layers[0](bwd_x, (bwd_h1, bwd_c1))
            bwd_h2, bwd_c2 = self.bwd_layers[1](bwd_h1, (bwd_h2, bwd_c2))

            out = self.drop(torch.cat((fwd_h2, bwd_h2), dim=1))  # [(64, 500), (64, 500)] --> [(64, 1000)]
            outputs.append(out.unsqueeze(0))  # [[1, 64, 1000], [1, 64, 1000], ..., [1, 64, 1000]]

        # Same type of seshaping applied to h1, c1, h2, c2, that is: [(64, 500), (64,500)] --> [(2, 64, 500)]
        h1, c1 = torch.cat((fwd_h1.unsqueeze(0), bwd_h1.unsqueeze(0))), torch.cat((fwd_c1.unsqueeze(0), bwd_c1.unsqueeze(0)))
        h2, c2 = torch.cat((fwd_h2.unsqueeze(0), bwd_h2.unsqueeze(0))), torch.cat((fwd_c2.unsqueeze(0), bwd_c2.unsqueeze(0)))

        # Shapes: (num_layers*directions, batch_size, input_size)
        last_hidden = torch.cat((h1, h2))  # [(2, 64, 500)] --> [(4, 64, 500)]
        last_cell = torch.cat((c1, c2))  # [(2, 64, 500)] --> [(4, 64, 500)]

        outputs = torch.cat(outputs, dim=0)  # list of 10 tensors of (64, 1000) --> [(10, 64, 1000)]
        # Shapes: (seq_len, batch, input_size)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # [(10, 64, 1000)] --> [(10, 64, 500)]

        return outputs, (last_hidden, last_cell)


class MogLSTM_UniDir(nn.Module):
    """ Unidirectional Mogrifier LSTM, can be used for both Encoder and Decoder """

    def __init__(self, input_size, hidden_size, dropout=0, mogrify_steps=5, cell_num=2):
        super().__init__()
        self.bidirectional = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([MogLSTMCell(input_size, hidden_size, mogrify_steps) for _ in range(cell_num)])
        self.drop = nn.Dropout(dropout)

    def forward(self, input_seq, hidden):
        """
        :param input_seq:
        :param hidden: Hidden state and cell state. Not used for encoder, only for decoder.
        """
        batch_size = input_seq.shape[1]
        seq_len = input_seq.shape[0]

        if hidden: # as mentioned above, only used for decoder
            hidden_states = hidden[0]
            cell_states = hidden[1]
            h1, c1 = hidden_states[0], cell_states[0]
            h2, c2 = hidden_states[1], cell_states[1]
        else:
            h1, c1 = MogLSTM_BiDir.init_states(batch_size, self.hidden_size)
            h2, c2 = MogLSTM_BiDir.init_states(batch_size, self.hidden_size)

        outputs = []
        for step in range(seq_len):
            x = self.drop(input_seq[step, :])
            h1, c1 = self.layers[0](x, (h1, c1))
            h2, c2 = self.layers[1](h1, (h2, c2))
            out = self.drop(h2)
            outputs.append(out.unsqueeze(0))

        # Shapes: (seq_len, batch, input_size)
        last_hidden = torch.cat((h1.unsqueeze(0), h2.unsqueeze(0)))  # torch.Size([2, 64, 500])
        last_cell = torch.cat((c1.unsqueeze(0), c2.unsqueeze(0)))  # torch.Size([2, 64, 500])
        outputs = torch.cat(outputs, dim=0)  # torch.Size([10, 64, 500])

        return outputs, (last_hidden, last_cell)


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers, dropout, gate=None, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.bidirectional = bidirectional
        self.gate = gate

        # - dropout is set to 0 only in case of 1 hidden layer
        # (according to PyTorch docs at least 2 layers are needed)
        if self.gate == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=self.bidirectional)
        elif self.gate == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                               dropout=(0 if n_layers == 1 else dropout), bidirectional=self.bidirectional)
        elif self.gate == "MogLSTM":
            if self.bidirectional:
                self.rnn = MogLSTM_BiDir(hidden_size, hidden_size, dropout=dropout)
            else:
                self.rnn = MogLSTM_UniDir(hidden_size, hidden_size, dropout=dropout)
        else:
            raise ValueError("The gated Encoder RNN's type has not been given."
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
        if self.gate == "MogLSTM":
            outputs, (hidden, cell_state) = self.rnn(embedded, hidden)
            hidden = (hidden, cell_state)
            return outputs, hidden
        elif self.gate == "LSTM":
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
    def __init__(self, attn_model, embedding, hidden_size, vocab_size,
                 n_layers, dropout, gate=None):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
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
        elif self.gate == "MogLSTM":
            self.rnn = MogLSTM_UniDir(hidden_size, hidden_size, dropout=dropout)
        else:
            raise ValueError("The gated Decoder RNN's type has not been given."
                             "Possible options are: 'GRU', 'LSTM', 'MogLSTM'.")

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
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
