import torch
import torch.nn as nn
import numpy as np
from ..search_spaces.basic_ops import Cell, LayerChoice
from ..search_spaces.multi_trail_model import (
    RandomRNN,
    SinglePathRandom,
    SinglePathRandomSimpleFirst,
    CountPReLU,
)
from ..search_spaces.omnimodels import omnimodel
from .modules.layers import SkipConnection, LockedDropout, Augmentation, TimeConv
from . import register_model
from apex.normalization import FusedLayerNorm


@register_model("TimeSeriesModelRnnCell")
@omnimodel([(Cell, RandomRNN), (LayerChoice, SinglePathRandom), (nn.Mish, CountPReLU)])
class TimeSeriesModelRnnCell(nn.Module):
    """
    Searchable RNN model to bananas with RandomRNN cell.

    :param hidden_size: dimensionality of nidden layers
    :type hidden_size: int
    :param output_size: dimensionality of output
    :type output_size: int
    :param embeddings_hidden: dimensionality of emdedding to each feature
    :type embeddings_hidden: int
    :param cat_cardinalities: cardinalities of dataset
    :type cat_cardinalities: list
    :param continious: list of continuous features name
    :type continious: list of string
    :param seq_len: length of sequention in dataset
    :type seq_len: int
    :param augmentations: class of augmentation layer, defaults to Augmentation
    :type augmentations: python class variable, optional
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        dropout=0.1,
        augmentations=Augmentation,
    ):
        super().__init__()
        self.input_size = (
            len(cat_cardinalities) * embeddings_hidden
            + len(continious) * embeddings_hidden
        )
        self.with_hidden = True
        self.output_size = output_size
        # main mutable object Cell -> RandomRNN
        self.hidden_size = hidden_size
        self.dropout_rate = dropout

        self.rnn = Cell(input_size=self.input_size, hidden_size=hidden_size)

        self.lockdrop = LockedDropout()

        self.categorical_features = torch.nn.ModuleDict(modules=None)
        self.linear_features = torch.nn.ModuleDict(modules=None)

        for cat_name, size in cat_cardinalities:
            self.categorical_features[cat_name] = nn.Embedding(
                size + 1, embeddings_hidden
            )

        for lin_name in continious:
            # linear projection along time axis
            self.linear_features[lin_name] = nn.Conv1d(
                1, embeddings_hidden, kernel_size=1, padding=0
            )

        self.aug = augmentations
        if augmentations is not None:
            self.aug = augmentations(seq_len)

        # self.linear = nn.Linear(hidden_size, output_size)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Mish(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, batch):
        features = []
        for cat_name in self.categorical_features:
            x = batch[cat_name].type(torch.long)
            emb = self.categorical_features[cat_name](x)
            features.append(emb)

        for lin_name in self.linear_features:
            x = batch[lin_name].unsqueeze(1)
            features.append(self.linear_features[lin_name](x).transpose(1, 2))

        # shape features = list(elem * (num_cat_features + num_lin_features) )
        # shape elem = batch x max_len x embeddings_hidden

        if self.aug is None:
            features = torch.cat(features, 2)
        else:
            features = self.aug(torch.cat(features, 2), self.training)

        # shape features = batch x max_len x (num_cat_features + num_lin_features)*embeddings_hidden

        features = features.transpose(0, 1)

        # sequence first
        # shape features = max_len x batch  x (num_cat_features + num_lin_features)*embeddings_hidden

        # features are iterated within the model / rnn cell
        # hidden=None
        # output, hidden = self.gru(features)
        if self.with_hidden:
            output, hidden = self.rnn(features)
        else:
            output = self.rnn(features)

        output = self.lockdrop(output, self.dropout_rate)

        # use last hidden outputs to predict series length
        output = self.decoder(output[-1].reshape(-1, self.hidden_size))
        return {"preds": output}


@register_model("TimeSeriesModelGRU")
class TimeSeriesModelGRU(TimeSeriesModelRnnCell):
    """
    Not searchable GRU model, child of TimeSeriesModelRnnCell. All passed parameters are equal
    to parent. Change self.rnn to nn.GRU.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        dropout=0.1,
        augmentations=Augmentation,
    ):
        super().__init__(
            hidden_size,
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            dropout,
            augmentations,
        )

        self.rnn = torch.nn.GRU(self.input_size, hidden_size)


@register_model("TimeSeriesModelLSTM")
class TimeSeriesModelLSTM(TimeSeriesModelRnnCell):
    """
    Not searchable LSTM model, child of TimeSeriesModelRnnCell. All passed parameters are equal
    to parent. Change self.rnn to nn.LSTM.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        dropout=0.1,
        augmentations=Augmentation,
    ):
        super().__init__(
            hidden_size,
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            dropout,
            augmentations,
        )

        self.rnn = torch.nn.LSTM(self.input_size, hidden_size)


@register_model("TimeSeriesModelHyperband")
class TimeSeriesModelHyperband(TimeSeriesModelRnnCell):
    """
    Searchable RNN model with fixed searchspace, child of TimeSeriesModelRnnCell. All passed parameters are equal
    to parent. Change self.rnn and self.decoder to searchable modules.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=Augmentation,
    ):
        super().__init__(
            hidden_size,
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations,
        )

        self.rnn = torch.nn.Sequential(
            LayerChoice(
                [
                    nn.LSTM(self.input_size, self.hidden_size),
                    nn.GRU(self.input_size, self.hidden_size),
                    nn.RNN(self.input_size, self.hidden_size),
                    nn.Linear(self.input_size, self.hidden_size),
                    TimeConv(self.input_size, self.hidden_size, 5),
                    TimeConv(self.input_size, self.hidden_size, 7),
                ]
            ),
            nn.Mish(),
            LayerChoice(
                [
                    nn.LSTM(self.hidden_size, self.hidden_size),
                    nn.GRU(self.hidden_size, self.hidden_size),
                    nn.RNN(self.hidden_size, self.hidden_size),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    TimeConv(self.hidden_size, self.hidden_size, 5),
                    TimeConv(self.hidden_size, self.hidden_size, 7),
                    SkipConnection(),
                ]
            ),
            nn.Mish(),
            LayerChoice(
                [
                    nn.LSTM(self.hidden_size, self.hidden_size),
                    nn.GRU(self.hidden_size, self.hidden_size),
                    nn.RNN(self.hidden_size, self.hidden_size),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    TimeConv(self.hidden_size, self.hidden_size, 5),
                    TimeConv(self.hidden_size, self.hidden_size, 7),
                    SkipConnection(),
                ]
            ),
            nn.Mish(),
            LayerChoice(
                [
                    nn.LSTM(self.hidden_size, self.hidden_size),
                    nn.GRU(self.hidden_size, self.hidden_size),
                    nn.RNN(self.hidden_size, self.hidden_size),
                    TimeConv(self.hidden_size, self.hidden_size, 5),
                    TimeConv(self.hidden_size, self.hidden_size, 7),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    SkipConnection(),
                ]
            ),
            nn.Mish(),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Mish(),
            LayerChoice(
                [
                    nn.Dropout(0.1),
                    nn.Dropout(0.3),
                    SkipConnection(),
                ]
            ),
            LayerChoice(
                [
                    FusedLayerNorm(self.hidden_size // 2),
                    SkipConnection(),
                ]
            ),
            LayerChoice(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                        nn.Mish(),
                        nn.Dropout(0.1),
                    ),
                    nn.Sequential(
                        nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                        nn.Mish(),
                        FusedLayerNorm(self.hidden_size // 2),
                        nn.Dropout(0.1),
                    ),
                    SkipConnection(),
                ]
            ),
            nn.Linear(self.hidden_size // 2, self.output_size),
        )
        self.with_hidden = False

    # def reinit(self):
    #     self.hidden_size_ = np.random.choice(
    #         np.linspace(
    #             self.min_hidden_size,
    #             self.max_hidden_size,
    #             self.hidden_size_steps,
    #         ).astype(int)
    #     )
    #     self.rnn = torch.nn.Sequential(
    #         LayerChoice(
    #             [
    #                 nn.LSTM(self.input_size, self.hidden_size_),
    #                 nn.GRU(self.input_size, self.hidden_size_),
    #             ]
    #         ),
    #         LayerChoice(
    #             [nn.Mish(), nn.ReLU(), nn.LeakyReLU(0.4), SkipConnection()]
    #         ),
    #         LayerChoice(
    #             [
    #                 nn.LSTM(self.hidden_size_, self.hidden_size_),
    #                 nn.GRU(self.hidden_size_, self.hidden_size_),
    #                 SkipConnection(),
    #             ]
    #         ),
    #         LayerChoice(
    #             [nn.Mish(), nn.ReLU(), nn.LeakyReLU(0.4), SkipConnection()]
    #         ),
    #         LayerChoice(
    #             [
    #                 nn.LSTM(self.hidden_size_, self.hidden_size_),
    #                 nn.GRU(self.hidden_size_, self.hidden_size_),
    #                 SkipConnection(),
    #             ]
    #         ),
    #         LayerChoice(
    #             [nn.Mish(), nn.ReLU(), nn.LeakyReLU(0.4), SkipConnection()]
    #         ),
    #         nn.Linear(self.hidden_size_, self.hidden_size),
    #     )


@register_model("TimeSeriesModelRandomStack")
@omnimodel(
    [
        (Cell, RandomRNN),
        (LayerChoice, SinglePathRandomSimpleFirst),
        (nn.Mish, CountPReLU),
    ]
)
class TimeSeriesModelRandomStack(nn.Module):
    """
    Searchable RNN model with fixed searchspace, child of TimeSeriesModelRnnCell. All passed parameters are equal
    to parent. Change self.rnn and self.decoder to searchable modules.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=Augmentation,
    ):
        super().__init__()

        self.input_size = (
            len(cat_cardinalities) * embeddings_hidden
            + len(continious) * embeddings_hidden
        )
        self.output_size = output_size
        # main mutable object Cell -> RandomRNN
        self.hidden_size = hidden_size
        self.lockdrop = LockedDropout()

        self.categorical_features = torch.nn.ModuleDict(modules=None)
        self.linear_features = torch.nn.ModuleDict(modules=None)

        for cat_name, size in cat_cardinalities:
            self.categorical_features[cat_name] = nn.Embedding(
                size + 1, embeddings_hidden
            )

        for lin_name in continious:
            # linear projection along time axis
            self.linear_features[lin_name] = nn.Conv1d(
                1, embeddings_hidden, kernel_size=1, padding=0
            )

        self.aug = augmentations
        if augmentations is not None:
            self.aug = augmentations(seq_len)

        self.rnn = torch.nn.Sequential(
            LayerChoice(
                [
                    nn.GRU(self.input_size, self.hidden_size),
                    nn.LSTM(self.input_size, self.hidden_size),
                    nn.RNN(self.input_size, self.hidden_size),
                    nn.Linear(self.input_size, self.hidden_size),
                    TimeConv(self.input_size, self.hidden_size, 5),
                    TimeConv(self.input_size, self.hidden_size, 7),
                ]
            ),
            nn.Mish(),
            LayerChoice(
                [
                    SkipConnection(),
                    nn.GRU(self.hidden_size, self.hidden_size),
                    nn.LSTM(self.hidden_size, self.hidden_size),
                    nn.RNN(self.hidden_size, self.hidden_size),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    TimeConv(self.hidden_size, self.hidden_size, 5),
                    TimeConv(self.hidden_size, self.hidden_size, 7),
                ]
            ),
            nn.Mish(),
            LayerChoice(
                [
                    SkipConnection(),
                    nn.GRU(self.hidden_size, self.hidden_size),
                    nn.LSTM(self.hidden_size, self.hidden_size),
                    nn.RNN(self.hidden_size, self.hidden_size),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    TimeConv(self.hidden_size, self.hidden_size, 5),
                    TimeConv(self.hidden_size, self.hidden_size, 7),
                ]
            ),
            nn.Mish(),
            LayerChoice(
                [
                    SkipConnection(),
                    nn.GRU(self.hidden_size, self.hidden_size),
                    nn.LSTM(self.hidden_size, self.hidden_size),
                    nn.RNN(self.hidden_size, self.hidden_size),
                    TimeConv(self.hidden_size, self.hidden_size, 5),
                    TimeConv(self.hidden_size, self.hidden_size, 7),
                    nn.Linear(self.hidden_size, self.hidden_size),
                ]
            ),
            nn.Mish(),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Mish(),
            LayerChoice(
                [
                    nn.Dropout(0.1),
                    SkipConnection(),
                    nn.Dropout(0.3),
                ]
            ),
            LayerChoice(
                [
                    SkipConnection(),
                    FusedLayerNorm(self.hidden_size // 2),
                ]
            ),
            LayerChoice(
                [
                    SkipConnection(),
                    nn.Sequential(
                        nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                        nn.Mish(),
                        nn.Dropout(0.1),
                    ),
                    nn.Sequential(
                        nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                        nn.Mish(),
                        FusedLayerNorm(self.hidden_size // 2),
                        nn.Dropout(0.1),
                    ),
                ]
            ),
            nn.Linear(self.hidden_size // 2, self.output_size),
        )
        self.with_hidden = False

    def forward(self, batch):
        features = []
        for cat_name in self.categorical_features:
            x = batch[cat_name].type(torch.long)
            emb = self.categorical_features[cat_name](x)
            features.append(emb)

        for lin_name in self.linear_features:
            x = batch[lin_name].unsqueeze(1)
            features.append(self.linear_features[lin_name](x).transpose(1, 2))

        # shape features = list(elem * (num_cat_features + num_lin_features) )
        # shape elem = batch x max_len x embeddings_hidden

        if self.aug is None:
            features = torch.cat(features, 2)
        else:
            features = self.aug(torch.cat(features, 2), self.training)

        # shape features = batch x max_len x (num_cat_features + num_lin_features)*embeddings_hidden

        features = features.transpose(0, 1)

        # sequence first
        # shape features = max_len x batch  x (num_cat_features + num_lin_features)*embeddings_hidden

        # features are iterated within the model / rnn cell
        # hidden=None
        # output, hidden = self.gru(features)
        if self.with_hidden:
            output, hidden = self.rnn(features)
        else:
            output = self.rnn(features)

        output = self.lockdrop(output, 0.1)

        # use last hidden outputs to predict series length
        output = self.decoder(output[-1].reshape(-1, self.hidden_size))
        return {"preds": output}


@register_model("SpeechGRU")
class SpeechGRU(nn.Module):
    """
    Not searchable GRU model for speech data. All parameters are equal to TimeSeriesModelRnnCell's ones.

    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=None,  # Augmentation(),
    ):
        super().__init__()

        self.lockdrop = LockedDropout()

        # Lout = (Lin + 2padding - kernel )/2 + 1
        kernels = [5, 7, 9, 11, 25, 55]
        stride = 8

        self.cnn_features = nn.ModuleList(
            [
                nn.Conv1d(1, 10, kernel_size=k, padding=k // 2, stride=stride)
                for k in kernels
            ]
        )

        self.hidden_size = hidden_size
        self.rnn = torch.nn.GRU(60, hidden_size)

        self.aug = augmentations
        if augmentations is not None:
            self.aug = augmentations(seq_len)

        # self.linear = nn.Linear(hidden_size, output_size)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, batch):
        features = []
        for cnn in self.cnn_features:
            features.append(cnn(batch.unsqueeze(1)))
        features = torch.cat(features, 1)
        if not self.aug is None:
            features = self.aug(features, self.training)

        # shape features = batch x channels x max_len

        features = features.transpose(0, 2)
        features = features.transpose(1, 2)

        # sequence first
        # shape features = max_len x batch  x (num_cat_features + num_lin_features)*embeddings_hidden

        # features are iterated within the model / rnn cell
        # hidden=None
        # output, hidden = self.gru(features)

        output, hidden = self.rnn(features)

        # use last hidden outputs to predict series length
        output = self.decoder(output[-1].reshape(-1, self.hidden_size))
        return {"preds": output}
