import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import register_model
from .modules.layers import Augmentation
from ..search_spaces.basic_ops import (
    LayerSkip,
    LayerChoice,
)
from ..search_spaces.omnimodels import omnimodel
from ..search_spaces.multi_trail_model import (
    SkipSampler,
    SinglePathRandom,
)


@register_model("FlexibleStumb")
@omnimodel(
    [
        (LayerChoice, SinglePathRandom),
        (LayerSkip, SkipSampler),
    ]
)
class FlexibleStumb(nn.Module):
    """Stem + head searchable model.

    The model consists only of stem, and head.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        num_embeddings_hidden="auto",
        augmentations=None,
        dropout=0.1,
        batch_size=None,
    ):
        """
        :param hidden_size: QK dimenstionality inside the transformer
        :type hidden_size: int
        :param output_size: model output dimenstionality
        :type output_size: int
        :param embeddings_hidden: categorical features dimenstionality, can be chosen automatically
        :type embeddings_hidden: int or literal "auto"
        :param cat_cardinalities: cardinalities of the categorical features
        :type cat_cardinalities: list of tuples (str, int)
        :param continious: continious features names
        :type continious: list of str
        :param seq_len: length of sequences in the dataset
        :type seq_len: int
        :param num_embeddings_hidden: numerical features dimenstionality, can be chosen automatically
        :type num_embeddings_hidden: int or literal "auto"
        :param augmentations: optional augmentations to apply
        :param dropout: dropout rate
        :type dropout: float
        :param batch_size: batch size
        :type batch_size: int
        """

        super().__init__()
        self.batch_size = batch_size
        self.accepts_time = False

        self.stem = Stem(
            cat_embeddings_hidden=embeddings_hidden,
            num_embeddings_hidden=num_embeddings_hidden,
            cat_cardinalities=cat_cardinalities,
            continious=continious,
            seq_len=seq_len,
            dropout=dropout,
            augmentations=augmentations,
        )
        emb_size = self.stem.get_n_features()

        self.head = Head(emb_size, output_size)

    def get_feature_vector(self, arch):
        """
        returns feature vector: stem features,
                                position encoder feature,
                                FlexibleEncoder features,
                                FlexibleDecoder features,
                                head features
        """
        stem_features = self.stem.get_feature_vector(arch)
        head_features = self.head.get_feature_vector(arch)
        return stem_features + head_features

    def forward(self, batch):
        embedded_sequence = self.stem(batch)
        return self.head(embedded_sequence)


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@register_model("Stem")
class Stem(nn.Module):
    """Common stem for models.

    Performs basic features preprocessing and encoding. Embedding layer is used for categorical features.
    Embedding size can be specified or determined automatically. Numerical features are standardized and
    preocessed with convolutions.
    """

    def __init__(
        self,
        cat_embeddings_hidden,
        num_embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        dropout=0.2,
        augmentations=Augmentation,
    ):
        """Intitialize model.

        :param cat_embeddings_hidden: dimensions per categorical feature or "auto"
        :type cat_embeddings_hidden: int ot str
        :param cat_embeddings_hidden: dimensions per numerical feature or "auto"
        :type cat_embeddings_hidden: int ot str
        :param cat_cardinalities: cardinalities of the categorical features
        :type cat_cardinalities: list of tuples (str, int)
        :param continious: continious features names
        :type continious: list of str
        :param seq_len: length of sequences in the dataset
        :type seq_len: int
        :param dropout: dropout rate
        :type dropout: float
        :param augmentations: optional augmentations to apply
        """
        super().__init__()

        self.seq_len = seq_len
        self.aug = augmentations
        if augmentations is not None:
            self.aug = augmentations(seq_len)

        self.categorical_features = nn.ModuleDict(modules=None)
        self.linear_features = nn.ModuleDict(modules=None)

        self._n_features = 0
        cat_emb_sizes = []
        for cat_name, size in cat_cardinalities:
            if cat_embeddings_hidden == "auto":
                emb_size = int(min(600, round(1.6 * size**0.56)))
            else:
                emb_size = cat_embeddings_hidden
            cat_emb_sizes.append(emb_size)

            self._n_features += emb_size

            self.categorical_features[cat_name] = nn.Embedding(size + 1, emb_size)

        n_channels = len(continious)
        if num_embeddings_hidden == "auto":
            if cat_emb_sizes:
                num_embeddings_hidden = int(np.mean(cat_emb_sizes))
            else:
                raise ValueError(
                    "Can't automatically determine embedding size for numerical features. "
                    "Specify explicitly `num_embeddings_hidden`."
                )

        self.linear_features = nn.ModuleDict({k: Dummy() for k in continious})

        self.linear_prep = None
        if n_channels > 0:
            self.linear_prep = torch.nn.Sequential(
                nn.BatchNorm1d(n_channels),
                LayerChoice(
                    [
                        nn.Conv1d(
                            n_channels,
                            num_embeddings_hidden * n_channels,
                            groups=n_channels,
                            kernel_size=1,
                        ),
                        nn.Conv1d(
                            n_channels,
                            num_embeddings_hidden * n_channels,
                            groups=n_channels,
                            kernel_size=3,
                            padding=3 // 2,
                        ),
                        nn.Conv1d(
                            n_channels,
                            num_embeddings_hidden * n_channels,
                            groups=n_channels,
                            kernel_size=5,
                            padding=5 // 2,
                        ),
                        nn.Conv1d(
                            n_channels,
                            num_embeddings_hidden * n_channels,
                            groups=n_channels,
                            kernel_size=min(1 + seq_len // 4 * 2, 11),
                            padding="same",
                        ),
                    ]
                ),
                LayerSkip(nn.Dropout(dropout)),
            )
        self._n_features += num_embeddings_hidden * n_channels

    def get_n_features(self):
        return self._n_features

    def get_feature_vector(self, arch):
        """
        returns feature vector: 4 features - for LayerChoice,
                                1 - LayerSkip
        """
        stem_features = arch["stem.linear_prep.1"] + [arch["stem.linear_prep.2"]]
        return stem_features

    def forward(self, batch):
        cat_features = []
        for cat_name in self.categorical_features:
            x = batch[cat_name].type(torch.long)
            emb = self.categorical_features[cat_name](x)
            cat_features.append(emb)

        if self.linear_prep:
            lin_features = [batch[lin].unsqueeze(1) for lin in self.linear_features]
            lin_features = torch.cat(lin_features, 1)
            lin_features = self.linear_prep(lin_features).transpose(1, 2)
            features = torch.cat([*cat_features, lin_features], 2)
        else:
            features = torch.cat(cat_features, 2)

        # shape features = list(elem * (num_cat_features + num_lin_features) )
        # shape elem = batch x max_len x embeddings_hidden

        if self.aug is not None:
            embdeded_sequence = self.aug(features, self.training)
        else:
            embdeded_sequence = features

        embdeded_sequence = F.pad(
            embdeded_sequence,
            (0, 0, 0, self.seq_len - embdeded_sequence.shape[1]),
            "constant",
            0,
        )

        return embdeded_sequence


# assuming x has shape (batch_size, seq_len, n_features)
class SpatialDropout(nn.Module):
    """Randomly zero entire sequence elements."""

    def __init__(self, p):
        """
        :param p: dropout rate
        :type p: float
        """
        super().__init__()
        self.dropout = nn.Dropout1d(p)

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, seq_len, n_features)
        """
        return self.dropout(x)


class TakeLastHidden(nn.Module):
    """Aggregate tensor along temporal dimension by taking the last element."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, seq_len, n_features)
        """
        return x[:, -1]


class GlobalMaxPooling(nn.Module):
    """Aggregate tensor along temporal dimension by taking the maximum along each channel."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, seq_len, n_features)
        """
        return torch.max(x, dim=1).values


class GlobalAvgPooling(nn.Module):
    """Aggregate tensor along temporal dimension by taking the average along each channel."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, seq_len, n_features)
        """
        return torch.mean(x, dim=1)


class GlobalMixedPooling(nn.Module):
    """Aggregate tensor along temporal dimension by taking the max and mean along each channel."""

    def __init__(self):
        super().__init__()
        self.mp = GlobalMaxPooling()
        self.ap = GlobalAvgPooling()

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, seq_len, n_features)
        """
        m = self.mp(x)
        a = self.ap(x)
        return (m + a) / 2


@register_model("Head")
class Head(nn.Module):
    """Common head.

    Aggregate sequence tensor along the temporal dimension and project to desired dimensionality.
    """

    def __init__(self, hidden_size, output_size):
        """Intitialize model.

        :param hidden_size: number of channels of input tensor
        :type hidden_size: int
        :param output_size: desired output dimensionality
        :type hidden_size: int
        """

        super().__init__()
        self.layers = nn.Sequential(
            nn.Mish(inplace=True),
            LayerSkip(SpatialDropout(0.2)),
            LayerChoice([GlobalAvgPooling(), GlobalMaxPooling(), GlobalMixedPooling()]),
            nn.Linear(hidden_size, output_size),
        )

    def get_feature_vector(self, arch):
        """
        returns feature vector: 1 - LayerSkip,
                                3 features - for LayerChoice,
        """
        head_features = [arch["head.layers.1"]] + arch["head.layers.2"]
        return head_features

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, seq_len, n_features)
        """
        return {"preds": self.layers(x)}
