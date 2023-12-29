import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from .modules.convolutional import (
    get_padding,
    get_min_seq_len,
    FreqFilter,
)

from ..search_spaces.basic_ops import LayerChoice
from ..search_spaces.omnimodels import omnimodel
from ..search_spaces.multi_trail_model import (
    DiffLayerSoftMax,
    SinglePathRandom,
    IdentityLayer,
    CountPReLU,
    CountReLU,
)
from . import register_model


@register_model("TimeFreqLinearModel")
class TimeFreqLinearModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len=200,
        periods=[2, 5, 10, 20],
        reduction=2,
        return_embeddings=True,
    ):
        super().__init__()
        self.return_embeddings = return_embeddings
        print("Initializing TimeFreqLinearModel")
        min_seq_len = get_min_seq_len(seq_len, periods, reduction)
        print(f"Min sequence length after projection is {min_seq_len}")

        self.categorical_features = torch.nn.ModuleDict(modules=None)
        self.linear_features = torch.nn.ModuleDict(modules=None)

        for cat_name, size in cat_cardinalities:
            self.categorical_features[cat_name] = nn.Embedding(
                size + 1, embeddings_hidden
            )

        for lin_name in continious:
            # linear projection along time axis
            self.linear_features[lin_name] = nn.Conv1d(
                1, embeddings_hidden, kernel_size=5, padding=2
            )

        self.hidden_size = hidden_size
        self.embeddings_hidden = embeddings_hidden
        input_size = (len(cat_cardinalities) + len(continious)) * self.embeddings_hidden

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )

        self.final_out = torch.nn.Linear(hidden_size, output_size)

        self.frequencies = nn.ModuleList(
            [
                FreqFilter(
                    p,
                    reduction,
                    input_size,
                    seq_len,
                    min_seq_len=min_seq_len,
                )
                for p in periods
            ]
        )

        self.alphas = nn.Parameter(torch.ones(len(self.frequencies)))

    def forward(self, batch):
        features = []
        for cat_name in self.categorical_features:
            x = batch[cat_name].type(torch.long)
            features.append(self.categorical_features[cat_name](x))

        for lin_name in self.linear_features:
            x = batch[lin_name].unsqueeze(1)
            features.append(self.linear_features[lin_name](x).transpose(1, 2))

        # shape features = list(elem * (num_cat_features + num_lin_features) )
        # shape elem = batch x max_len x embeddings_hidden

        features = torch.cat(features, 2)

        first = True
        for fr in self.frequencies:
            if first:
                frequencies = fr(features)
                first = False
            else:
                frequencies = frequencies + fr(features)

        # shape features = batch x max_len x (num_cat_features + num_lin_features)*embeddings_hidden

        features = self.decoder(
            torch.cat(
                [
                    frequencies.mean(axis=-1),
                    frequencies.std(axis=-1, unbiased=False),
                ],
                axis=-1,
            )
        )

        output = self.final_out(features)
        if self.return_embeddings:
            return {"preds": output, "features": features}
        else:
            return {"preds": output}


@register_model("TimeFreqLinearSpeech")
class TimeFreqLinearSpeech(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len=64000,
        periods=[3, 10, 25],
        reduction=100,
        dropout=0.1,
    ):
        super().__init__()

        embeddings_hidden = 48

        print("Initializing TimeFreqLinearModel")
        min_seq_len = int(get_min_seq_len(seq_len, periods, reduction))
        print(f"Min sequence length after projection is {min_seq_len}")

        self.bnn = nn.BatchNorm1d(1)
        self.cnn_in = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(1, 6, kernel_size=11, padding=11 // 2),
            nn.ReLU(),
            nn.Conv1d(6, 24, kernel_size=9, padding=9 // 2),
            nn.ReLU(),
            nn.Conv1d(24, 48, kernel_size=7, padding=7 // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        dim_in = len(periods) * embeddings_hidden

        self.out = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_in // 2),
            nn.PReLU(),
            nn.Linear(dim_in // 2, 1),
            nn.PReLU(),
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(min_seq_len, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.frequencies = nn.ModuleList(
            [
                FreqFilter(
                    p,
                    reduction,
                    embeddings_hidden,
                    seq_len,
                    min_seq_len=min_seq_len,
                )
                for p in periods
            ]
        )

    def forward(self, batch):
        features = batch.unsqueeze(1)
        features = self.bnn(features)
        features = self.cnn_in(features)
        features = features.transpose(1, 2)

        frequencies = []
        for fr in self.frequencies:
            frequencies.append(fr(features))

        features = torch.cat(frequencies, 1).transpose(1, 2)
        features = self.out(features).squeeze(-1)
        output = self.decoder(features)
        return {"preds": output}


@register_model("TimeFreqLinearSpeechSearchableDIFF")
@omnimodel([(LayerChoice, DiffLayerSoftMax)])
class TimeFreqLinearSpeechSearchableDIFF(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len=64000,
        periods=[3, 5, 8, 12, 20, 30, 50],
        reduction=2,
        dropout=0.1,
    ):
        super().__init__()

        FR_LAYERS = 3

        print("Initializing TimeFreqLinearModel")
        min_seq_len = int(get_min_seq_len(seq_len, periods, reduction))
        print(f"Min sequence length after projection is {min_seq_len}")

        self.bnn = nn.BatchNorm1d(1)
        self.cnn_in = nn.Sequential(
            nn.Dropout(dropout),
            LayerChoice(
                [
                    nn.Conv1d(1, 6, kernel_size=k, padding=k // 2)
                    for k in range(1, 16, 2)
                ]
            ),
            nn.ReLU(),
            LayerChoice(
                [
                    nn.Conv1d(6, 24, kernel_size=k, padding=k // 2)
                    for k in range(1, 16, 2)
                ]
            ),
            nn.ReLU(),
            LayerChoice(
                [
                    nn.Conv1d(24, embeddings_hidden, kernel_size=k, padding=k // 2)
                    for k in range(1, 16, 2)
                ]
            ),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        dim_in = FR_LAYERS * embeddings_hidden

        self.linear = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_in // 2),
            nn.PReLU(),
            nn.Linear(dim_in // 2, 1),
            nn.PReLU(),
        )

        self.frequencies = nn.ModuleList()
        for i in range(FR_LAYERS):
            f = LayerChoice(
                [
                    FreqFilter(
                        p,
                        reduction,
                        embeddings_hidden,
                        seq_len,
                        min_seq_len=min_seq_len,
                    )
                    for p in periods
                ]
            )
            self.frequencies.append(f)

        self.decoder = torch.nn.Sequential(
            nn.Linear(min_seq_len, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, batch):
        features = batch.unsqueeze(1)
        features = self.bnn(features)
        features = self.cnn_in(features)
        features = features.transpose(1, 2)

        frequencies = []
        for fr in self.frequencies:
            frequencies.append(fr(features))

        features = torch.cat(frequencies, 1).transpose(1, 2)
        features = self.linear(features).squeeze(-1)
        output = self.decoder(features)
        return {"preds": output}


@register_model("TimeFreqLinearSpeechSearchableRND")
@omnimodel(
    [
        (LayerChoice, SinglePathRandom),
        (nn.PReLU, CountPReLU),
        (nn.ReLU, CountReLU),
    ]
)
class TimeFreqLinearSpeechSearchableRND(TimeFreqLinearSpeechSearchableDIFF):
    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len=64000,
        periods=[2, 5, 8, 16, 25, 50],
        reduction=100,
        dropout=0.1,
    ):
        super().__init__(
            hidden_size,
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            periods,
            reduction,
            dropout,
        )


@register_model("TimeFreqLinearSpeechSearchablePT")
@omnimodel([(LayerChoice, IdentityLayer)])
class TimeFreqLinearSpeechSearchablePT(TimeFreqLinearSpeechSearchableDIFF):
    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len=64000,
        periods=[3, 5, 8, 11, 12, 15, 25, 30, 35, 40, 50],
        reduction=100,
        dropout=0.1,
    ):
        super().__init__(
            hidden_size,
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            periods,
            reduction,
            dropout,
        )
