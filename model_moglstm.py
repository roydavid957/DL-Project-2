import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple, List, Dict

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class MogLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, mogrify_steps=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mogrifier_lstm_layer1 = MogLSTMCell(input_size, hidden_size, mogrify_steps)
        self.mogrifier_lstm_layer2 = MogLSTMCell(hidden_size, hidden_size, mogrify_steps)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_seq, hidden=None):
        """
        :param input_seq:
        :param hidden: Hidden state and cell state. Not provided for encoder, only for decoder.
        """
        batch_size = input_seq.shape[1]
        seq_len = input_seq.shape[0]

        if not hidden:
            h1, c1 = [torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)]
            h2, c2 = [torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)]
        else:
            hidden_states = hidden[0]
            cell_states = hidden[1]
            h1, c1 = hidden_states[0], cell_states[0]
            h2, c2 = hidden_states[1], cell_states[1]

        outputs = []
        for step in range(seq_len):
            x = self.drop(input_seq[step, :])
            h1, c1 = self.mogrifier_lstm_layer1(x, (h1, c1))  # dropout in default LSTM PyTorch doc. is true
            h2, c2 = self.mogrifier_lstm_layer2(h1, (h2, c2))
            out = self.drop(h2)
            outputs.append(out.unsqueeze(0))

        # Shapes: (seq_len, batch, input_size)
        last_hidden = torch.cat((h1.unsqueeze(0), h2.unsqueeze(0)))  # torch.Size([2, 64, 500])
        last_cell = torch.cat((c1.unsqueeze(0), c2.unsqueeze(0)))  # torch.Size([2, 64, 500])
        outputs = torch.cat(outputs, dim=0)  # torch.Size([10, 64, 500])

        return outputs, (last_hidden, last_cell)


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
        ht = ht.to(device)
        ct = ct.to(device)
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct


class EncoderMogLSTM(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers, dropout):
        super(EncoderMogLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # - Initialize RNN; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.rnn = MogLSTM(hidden_size, hidden_size, dropout=dropout)

    # TODO: Add PackedSequence once the basic method works
    def forward(self, input_seq, input_lengths, hidden=None):  # input_lengths is only used with PackedSequence
        """
        :param input_seq: Padded input sequence of shape (10,64) - before embedding
        :param input_lengths: A 1D tensor containing the length of each sentence within the batch in decreasing order.
                              E.g. [10,10,10,9,9,8...,3], thus size is (64). Required for PackedSequence.
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)  # input shape: (10, 64), output shape: (10, 64, 500)
        outputs, hidden = self.rnn(embedded, hidden)  # 'hidden' is used for decoder only

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


class LuongAttnDecoderMogLSTM(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, vocab_size, n_layers, dropout):

        super(LuongAttnDecoderMogLSTM, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = MogLSTM(hidden_size, hidden_size, dropout=dropout)

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        :param input_step:
        :param last_hidden: Tuple of tensors containing hidden and cell state with shapes (2, 64, 500), (2, 64, 500)
        :param encoder_outputs: Sequence of outputs of shape torch.Size([10, 64, 500])
        """
        # NOTE: we run this one step (word) at a time, parallel in each batch

        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional decoder MogLSTM
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
