import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple, List, Dict
from model import Attn

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


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


class EncoderMogLSTM(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers, dropout, bidirectional):
        super(EncoderMogLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.rnn = MogLSTM_BiDir(hidden_size, hidden_size, dropout=dropout)
        else:
            self.rnn = MogLSTM_UniDir(hidden_size, hidden_size, dropout=dropout)

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


class LuongAttnDecoderMogLSTM(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, vocab_size, n_layers, dropout):

        super(LuongAttnDecoderMogLSTM, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = MogLSTM_UniDir(hidden_size, hidden_size, dropout=dropout)

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
