import torch
import torch.nn as nn
from ..search_spaces.basic_ops import LayerChoice
from ..search_spaces.multi_trail_model import SinglePathRandom
from ..search_spaces.omnimodels import omnimodel

from .rnn_model import LockedDropout
from . import register_model


@register_model("TimeSeriesLinearModel")
@omnimodel([(LayerChoice, SinglePathRandom)])
class TimeSeriesLinearModel(nn.Module):
    """
    Linear searchable model.

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
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
    ):
        super().__init__()

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
        self.lockdrop = LockedDropout()
        input_size = (
            (len(cat_cardinalities) + len(continious)) * self.embeddings_hidden * 2
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size // 2, output_size),
        )

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

        # shape features = batch x max_len x (num_cat_features + num_lin_features)*embeddings_hidden

        features = features.transpose(0, 1)

        # sequence first
        # shape features = max_len x batch  x (num_cat_features + num_lin_features)*embeddings_hidden

        # features are iterated within the model / rnn cell
        # hidden=None
        # use last hidden outputs to predict series length
        output = self.decoder(
            torch.cat(
                [features.mean(axis=0), features.std(axis=0, unbiased=False)],
                axis=-1,
            )
        )
        # output = self.decoder(features.mean(axis = 0))
        return {"preds": output}
