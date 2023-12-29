import json
import os

import numpy as np
import torch
import torch.nn

from .custom_rnn import CustomRNN
from .dropout import embedded_dropout  # similar to one in  ENAS/models/shared_rnn.py
from .dropout import LockedDropout  # the same as in ENAS/models/shared_rnn.py
from .utils import recipe2rnntype


class Embeddings(torch.nn.Module):
    """
    Torch module to perform trainable embedding of categorical data (in integer format)
    and linear transformation of real valued data.
    Each entry will be embedded independently and resulted vectors will be concatenation of obtained values.
    """

    def __init__(self, ntokens, emb_dim=5, num_emb_dim=5):
        super().__init__()
        self.encoders = torch.nn.ModuleList()
        self.embeds = []
        self.linears = []
        for i, ntoken in enumerate(ntokens):
            if ntoken >= 0:
                self.embeds.append(i)
                self.encoders.append(torch.nn.Embedding(ntoken, emb_dim, padding_idx=0))
            else:
                self.linears.append(i)
        if len(self.linears) > 0:
            self.linear = torch.nn.Linear(len(self.linears), num_emb_dim, bias=False)

    def forward(self, x, lengths=None):
        outs = [emb(x[:, :, i].long()) for i, emb in zip(self.embeds, self.encoders)]
        if len(self.linears) > 0:
            _x = x[:, :, self.linears]
            _x = _x.view(*x.shape[:2], len(self.linears))
            outs.append(self.linear(_x.float()))
        out = torch.cat(outs, -1)
        return out


class BaseRNNModel(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        rnn_cell_types,
        ntoken,
        ninp,
        nhid,
        dropout=0.5,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        tie_weights=False,
        recipes=None,
        bidirectional=False,
        verbose=True,
    ):
        super(BaseRNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = torch.nn.Dropout(dropouti)
        self.hdrop = torch.nn.Dropout(dropouth)
        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(ntoken, ninp)
        self.verbose = verbose
        self.bidirectional = bidirectional

        self.rnns = []
        nlayers = len(rnn_cell_types)

        for i, rnn_cell_type in enumerate(rnn_cell_types):
            _nhid = nhid * 2 if self.bidirectional else nhid
            input_size = ninp if i == 0 else _nhid
            hidden_size = nhid if i != nlayers - 1 else (ninp if tie_weights else nhid)
            if rnn_cell_type == "LSTM":
                self.rnns.append(
                    torch.nn.LSTM(
                        input_size, hidden_size, bidirectional=self.bidirectional
                    )
                )
            elif rnn_cell_type == "GRU":
                self.rnns.append(
                    torch.nn.GRU(
                        input_size, hidden_size, bidirectional=self.bidirectional
                    )
                )
            elif rnn_cell_type == "RNN":
                self.rnns.append(
                    torch.nn.RNN(
                        input_size, hidden_size, bidirectional=self.bidirectional
                    )
                )
            else:
                self.rnns.append(
                    CustomRNN(
                        input_size,
                        hidden_size,
                        recipes[i],
                        bidirectional=self.bidirectional,
                    )
                )

        if self.verbose:
            print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = torch.nn.Linear(nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_cell_types = rnn_cell_types
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.recipes = recipes

    def reset(self):
        pass

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(
            self.encoder, input, dropout=self.dropoute if self.training else 0
        )
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for i, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[i])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        result = output.view(output.size(0) * output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden = []
        for i in range(self.nlayers):
            hidden_size = (
                self.nhid
                if i != self.nlayers - 1
                else (self.ninp if self.tie_weights else self.nhid)
            )

            if self.rnn_cell_types[i] in ["RNN", "GRU"]:
                hidden_tuple_size = 1
            elif self.rnn_cell_types[i] == "LSTM":
                hidden_tuple_size = 2
            elif self.rnn_cell_types[i] == "CustomRNN":
                hidden_tuple_size = self.rnns[0].cell_fw.hidden_tuple_size

            _n = 2 if self.bidirectional else 1
            hidden.append(
                tuple(
                    [
                        weight.new(_n, bsz, hidden_size).zero_()
                        for _ in range(hidden_tuple_size)
                    ]
                )
            )

        return hidden


class RecipeModel(BaseRNNModel):
    """
    Torch model based on the recipe.
    """

    def __init__(
        self,
        recipes,  # list with rnn cell (dict) computation graphs
        bidirectional,  # bidirectional RNN if True
        ntokens,  # list with numbers of tokens for each dim in data element
        emb_dim,  # embedding size for categorical features
        num_emb_dim,  # embedding size for numerical features
        nhid,  # dim of hidden state
        dense_dim,  # dense_dim*nhid = dim of decoder linear layer
        out_dim,  # number of categories for classification tasks
        dropout=0.5,  # dropout after rnn cells and in decoder
        dropouth=0.5,  # dropout between rnn cells
        dropouti=0.5,  # dropout after embedding
        dropoute=0.1,  # dropout for embedding
        tie_weights=False,  # share weights between encoder and decoder
        verbose=True,
    ):
        self.recurrent = True
        ninp = emb_dim * sum([_n > 0 for _n in ntokens]) + num_emb_dim
        if os.environ.get("NASLIB_OPTIMIZED_RNN", "0") == "1":
            rnn_types_list = [recipe2rnntype(r) for r in recipes]
        else:
            rnn_types_list = ["CustomRNN"] * len(recipes)
        super().__init__(
            rnn_types_list,
            1,
            ninp,
            nhid,
            dropout,
            dropouth,
            dropouti,
            dropoute,
            tie_weights,
            recipes,
            bidirectional,
            verbose,
        )
        self.encoder = Embeddings(ntokens, emb_dim, num_emb_dim)
        self.encoder.apply(self.init_fn)
        last_hidden_size = self.rnns[-1].hidden_size
        if self.bidirectional:
            last_hidden_size *= 2
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(last_hidden_size, int(dense_dim * last_hidden_size)),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(dense_dim * last_hidden_size), out_dim),
        )

    def init_fn(self, m):
        initrange = 0.1
        if m.__class__ == torch.nn.modules.sparse.Embedding:
            torch.nn.init.uniform_(m.weight, -initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = self.encoder(input)
        # emb = self.idrop(emb)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for i, rnn in enumerate(self.rnns):
            if (
                os.environ.get("NASLIB_OPTIMIZED_RNN", "0") == "1"
                and self.rnn_cell_types[i] in ["RNN", "GRU"]
                and isinstance(hidden[i], (list, tuple))
            ):
                _hidden = hidden[i][0]
            else:
                _hidden = hidden[i]

            raw_output, new_h = rnn(raw_output, _hidden)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.nlayers - 1:
                # self.hdrop(raw_output) add???
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        result = output
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden
