import torch
from torch import nn
import torch.nn.functional as F
from math import log, sqrt
from omegaconf import ListConfig
import numpy as np
from itertools import chain, compress, combinations

from ..modules.layers import Augmentation
from .searchable_ops import MixtureOfHeads, AttentionSelector
from ...search_spaces.basic_ops import (
    Repeat,
    LayerSkip,
    RepeatParallel,
    LayerChoice,
)
from ...search_spaces.omnimodels import omnimodel
from ...search_spaces.multi_trail_model import (
    RepeatRandom,
    ParallelRepsRandom,
    SkipSampler,
    SkipSampler2,
    SinglePathRandom,
)
from .basic_ops import (
    MHALayer,
    EncoderMHALayer,
    EncoderOmniLayer,
    DecoderMHALayer,
    DecoderOmniLayer,
    SinPosEncoding,
    SinTimeEncoding,
)

from ..modules.convolutional import get_min_seq_len
from ..common import Stem, Head
from .. import register_model
from ...utils.misc import disp_arch


def _non_empty_subsets(it):
    return list(chain.from_iterable(combinations(it, n) for n in range(1, len(it) + 1)))


class FlexibleEncoder(nn.Module):  # Feed forward transformer
    def __init__(
        self,
        num_layers: list = [1, 2, 4],
        dim_in: int = 128,
        dim_qk: int = 16,
        dim_feedforward: int = 512,
        mha_instance=MixtureOfHeads,
        mha_instance_kwargs={},
        heads="learnable",
        seq_len=100,
        dropout=0.1,
        tricks="off",
        available_tricks=None,
    ):
        super().__init__()
        assert tricks in ("off", "all_or_nothing", "choice")
        assert available_tricks is None or all(
            it in {"attention", "gru", "conv"} for it in available_tricks
        )

        self.dim_in = dim_in
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_heads_encoder = None
        self.tricks = tricks
        self.available_tricks = available_tricks or ["attention", "gru", "conv"]
        self.available_trick_combs = _non_empty_subsets(self.available_tricks)

        if "n_heads" in mha_instance_kwargs.keys():
            self.num_heads_encoder = mha_instance_kwargs["n_heads"]

        if tricks == "off":
            self.layers = Repeat(
                EncoderMHALayer(
                    dim_in,
                    dim_qk,
                    dim_feedforward,
                    heads,
                    mha_instance,
                    mha_instance_kwargs,
                    dropout,
                    return_att=False,
                ),
                num_layers,
            )
            return

        if isinstance(heads, list):
            raise ValueError(
                "Passing list to `heads` with enabled tricks is not supported. "
                "Consider passing single head type as a string or `omni=False`."
            )

        if tricks == "all_or_nothing":
            self.layers = LayerChoice(
                [
                    Repeat(
                        EncoderOmniLayer(
                            dim_in,
                            dim_qk,
                            dim_feedforward,
                            heads,
                            mha_instance,
                            mha_instance_kwargs,
                            dropout,
                            layers=self.available_tricks,
                        ),
                        num_layers,
                    ),
                    Repeat(
                        EncoderMHALayer(
                            dim_in,
                            dim_qk,
                            dim_feedforward,
                            heads,
                            mha_instance,
                            mha_instance_kwargs,
                            dropout,
                            return_att=False,
                        ),
                        num_layers,
                    ),
                ]
            )
        else:
            layer_combinations = self.available_trick_combs
            self.layers = Repeat(
                LayerChoice(
                    [
                        EncoderOmniLayer(
                            dim_in,
                            dim_qk,
                            dim_feedforward,
                            heads,
                            mha_instance,
                            mha_instance_kwargs,
                            dropout,
                            layers=layer_comb,
                        )
                        for layer_comb in layer_combinations
                    ]
                ),
                num_layers,
            )

    def get_feature_vector(self, arch):
        # define plugs
        # disp_arch(arch)
        layers_on_feat = [0] * max(self.num_layers)
        tricks_comb_feat = [
            [0] * len(self.available_trick_combs) for _ in range(max(self.num_layers))
        ]
        att_heads_feat = [
            [0] * max(self.num_heads_encoder) for _ in range(max(self.num_layers))
        ]

        # early return if no encoder
        if arch["encoder"] == 0:
            return list(
                chain(
                    layers_on_feat,
                    chain.from_iterable(tricks_comb_feat),
                    chain.from_iterable(att_heads_feat),
                )
            )

        # find the used layers
        key_to_all_layers = "encoder.layer.layers"
        chosen_op = None
        if self.tricks == "all_or_nothing":
            chosen_op = arch[key_to_all_layers].index(1)
            key_to_all_layers += f".ops.{chosen_op}"
        layers_on_feat = arch[key_to_all_layers][:]

        # features for used layers
        for layer_idx in compress(range(len(layers_on_feat)), layers_on_feat):
            key_to_layer = key_to_all_layers + f".ops.{layer_idx}"
            key_to_att = key_to_layer

            chosen_trick = 0  # no tricks
            if self.tricks == "all_or_nothing":
                if chosen_op == 0:  # all tricks
                    key_to_att += ".split_layer.layers.0.heads"
                    chosen_trick = len(self.available_trick_combs) - 1  # all
                else:  # nothing
                    key_to_att += ".attention.heads"
            elif self.tricks == "choice":
                chosen_trick = arch[key_to_layer].index(1)
                if "attention" in self.available_trick_combs[chosen_trick]:
                    key_to_att += f".ops.{chosen_trick}.split_layer.layers.0.heads"
                else:
                    key_to_att = None
            else:  # off
                key_to_att += ".attention.heads"

            tricks_comb_feat[layer_idx][chosen_trick] = 1
            if key_to_att is not None:
                att_heads_feat[layer_idx] = arch[key_to_att][:]

        return list(
            chain(
                layers_on_feat,
                chain.from_iterable(tricks_comb_feat),
                chain.from_iterable(att_heads_feat),
            )
        )

    def forward(self, x):
        return self.layers(x)


class RepeatedDecoderLayerWrapper(nn.Module):
    """Adaptor for layer choice"""

    def __init__(self, repeated_layer: Repeat):
        super().__init__()
        self.repeated_layer = repeated_layer

    def forward(self, x):
        decoder_out, _ = self.repeated_layer(x)
        return decoder_out


class FlexibleDecoder(nn.Module):  # Feed forward transformer
    def __init__(
        self,
        num_layers: list = [1, 2, 4],
        dim_in: int = 128,
        dim_qk: int = 16,
        dim_feedforward: int = 512,
        mha_instance=MixtureOfHeads,
        mha_instance_kwargs={},
        heads=["learnable"],
        seq_len=100,
        dropout=0.1,
        tricks="off",
        available_tricks=None,
    ):
        super().__init__()
        assert tricks in ("off", "all_or_nothing", "choice")
        assert available_tricks is None or all(
            it in {"attention", "gru", "conv"} for it in available_tricks
        )

        self.dim_in = dim_in
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_heads_decoder = None
        self.tricks = tricks
        self.available_tricks = available_tricks or ["attention", "gru", "conv"]
        self.available_trick_combs = _non_empty_subsets(self.available_tricks)

        if "n_heads" in mha_instance_kwargs.keys():
            self.num_heads_decoder = mha_instance_kwargs["n_heads"]

        if tricks == "off":
            self.layers = RepeatedDecoderLayerWrapper(
                Repeat(
                    DecoderMHALayer(
                        dim_in,
                        dim_qk,
                        dim_feedforward,
                        heads,
                        mha_instance,
                        mha_instance_kwargs,
                        dropout,
                    ),
                    num_layers,
                )
            )
            return

        if isinstance(heads, list):
            raise ValueError(
                "Passing list to `heads` with enabled tricks is not supported. "
                "Consider passing single head type as a string or `omni=False`."
            )

        if tricks == "all_or_nothing":
            self.layers = LayerChoice(
                [
                    RepeatedDecoderLayerWrapper(
                        Repeat(
                            DecoderOmniLayer(
                                dim_in,
                                dim_qk,
                                dim_feedforward,
                                heads,
                                mha_instance,
                                mha_instance_kwargs,
                                dropout,
                                layers=self.available_tricks,
                            ),
                            num_layers,
                        )
                    ),
                    RepeatedDecoderLayerWrapper(
                        Repeat(
                            DecoderMHALayer(
                                dim_in,
                                dim_qk,
                                dim_feedforward,
                                heads,
                                mha_instance,
                                mha_instance_kwargs,
                                dropout,
                            ),
                            num_layers,
                        )
                    ),
                ]
            )
        else:
            layer_combinations = self.available_trick_combs
            self.layers = RepeatedDecoderLayerWrapper(
                Repeat(
                    LayerChoice(
                        [
                            DecoderOmniLayer(
                                dim_in,
                                dim_qk,
                                dim_feedforward,
                                heads,
                                mha_instance,
                                mha_instance_kwargs,
                                dropout,
                                layers=layer_comb,
                            )
                            for layer_comb in layer_combinations
                        ]
                    ),
                    num_layers,
                )
            )

    def get_feature_vector(self, arch):
        # define plugs
        # disp_arch(arch)
        layers_on_feat = [0] * max(self.num_layers)
        tricks_comb_feat = [
            [0] * len(self.available_trick_combs) for _ in range(max(self.num_layers))
        ]
        self_att_heads_feat = [
            [0] * max(self.num_heads_decoder) for _ in range(max(self.num_layers))
        ]
        enc_att_heads_feat = [
            [0] * max(self.num_heads_decoder) for _ in range(max(self.num_layers))
        ]

        # early return if decoder is optional (FlexTf) and it's turned off
        if "decoder" in arch and arch["decoder"] == 0:
            return list(
                chain(
                    layers_on_feat,
                    chain.from_iterable(tricks_comb_feat),
                    chain.from_iterable(self_att_heads_feat),
                    chain.from_iterable(enc_att_heads_feat),
                )
            )

        # find the used layers
        key_to_all_layers = "decoder"
        if "decoder" in arch:  # it's optional
            key_to_all_layers += ".layer"
        key_to_all_layers += ".layers"

        chosen_op = None
        if self.tricks == "all_or_nothing":
            chosen_op = arch[key_to_all_layers].index(1)
            key_to_all_layers += f".ops.{chosen_op}"
        key_to_all_layers += ".repeated_layer"
        layers_on_feat = arch[key_to_all_layers][:]

        # features for used layers
        for layer_idx in compress(range(len(layers_on_feat)), layers_on_feat):
            key_to_layer = key_to_all_layers + f".ops.{layer_idx}"
            key_to_self_att = key_to_layer
            key_to_enc_att = key_to_layer

            chosen_trick = 0  # no tricks
            if self.tricks == "all_or_nothing":
                key_to_enc_att += ".enc_attention.heads"
                if chosen_op == 0:  # all tricks
                    key_to_self_att += ".split_layer.layers.0.heads"
                    chosen_trick = len(self.available_trick_combs) - 1  # all
                else:  # nothing
                    key_to_self_att += ".self_attention.heads"
            elif self.tricks == "choice":
                chosen_trick = arch[key_to_layer].index(1)
                key_to_enc_att += f".ops.{chosen_trick}.enc_attention.heads"
                if "attention" in self.available_trick_combs[chosen_trick]:
                    key_to_self_att += f".ops.{chosen_trick}.split_layer.layers.0.heads"
                else:
                    key_to_self_att = None
            else:  # off
                key_to_self_att += ".self_attention.heads"
                key_to_enc_att += ".enc_attention.heads"

            tricks_comb_feat[layer_idx][chosen_trick] = 1
            enc_att_heads_feat[layer_idx] = arch[key_to_enc_att][:]
            if key_to_self_att is not None:
                self_att_heads_feat[layer_idx] = arch[key_to_self_att][:]

        return list(
            chain(
                layers_on_feat,
                chain.from_iterable(tricks_comb_feat),
                chain.from_iterable(self_att_heads_feat),
                chain.from_iterable(enc_att_heads_feat),
            )
        )

    def forward(self, x):
        query, encoded_sequence = x
        return self.layers((query, encoded_sequence))


@register_model("FlexibleTransformer")
@omnimodel(
    [
        (LayerChoice, SinglePathRandom),
        (Repeat, RepeatRandom),
        (RepeatParallel, ParallelRepsRandom),
        (LayerSkip, SkipSampler),
    ]
)
class FlexibleTransformer(nn.Module):
    """Encoder-decoder transformer searchable model.

    The model consists of stem, optional encoder, optional decoder and head.
    Encoder and decoder support random number of layers and heads per layer. Head type can be sampled randomly as well.
    Encoder supports tricks. Instead of usual MHSA operation the following operation is performed.
    Embedding is split on several parts and each part is processed by MHA, GRU ot convolution.
    See README for more information.
    The tricks can be completely turned off, all enabled or all disabled synchronously in all layers or independentyl chosen in each layer.
    If tricks are enabled, the model supports only single head type.
    """

    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        num_embeddings_hidden="auto",
        augmentations=None,
        heads_encoder="learnable",
        num_heads_encoder=[1, 2, 4, 8],
        num_layers_encoder=[1, 2, 4],
        heads_decoder="learnable",
        num_heads_decoder=[1, 2, 4, 8],
        num_layers_decoder=[1, 2],
        dropout=0.1,
        batch_size=None,
        decoder_input_len=None,
        tricks="all_or_nothing",
        available_tricks=None,
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
        :param heads_encoder: head types to use in encoder
        :type heads_encoder: str ot list of str
        :param num_heads_encoder: possible number of heads per layer in encoder
        :type num_heads_encoder: list if int
        :param num_layers_encoder: possible number of layers in encoder
        :type num_layers_encoder: list if int
        :param heads_decoder: head types to use in decoder
        :type heads_decoder: str ot list of str
        :param num_heads_decoder: possible number of heads per layer in decoder
        :type num_heads_decoder: list if int
        :param num_layers_decoder: possible number of layers in decoder
        :type num_layers_decoder: list if int
        :param dropout: dropout rate
        :type dropout: float
        :param batch_size: batch size
        :type batch_size: int
        :param decoder_input_len: length of learnable sequence passed to decoder
        :type decoder_input_len: int
        :param tricks: tricks and how they are used in encoder.
        :type tricks: str literal: "off", "all_or_nothing" or "choice"
        :param available_tricks: a list of available trikcs. All included by default
        :type available_tricks: list of str
        """

        super().__init__()
        self.batch_size = batch_size
        self.accepts_time = True

        DEFAULT_HEADS = ["learnable"] * 9 + [
            "cross",
            "diag1",
            "diag_1",
            "bottom",
            "top",
            "bottom_inv",
            "top_inv",
            "left",
            "right",
        ]

        if heads_encoder == "default":
            heads_encoder = DEFAULT_HEADS[:]
        if heads_decoder == "default":
            heads_decoder = DEFAULT_HEADS[:]

        if isinstance(heads_encoder, ListConfig):
            heads_encoder = list(heads_encoder)
        if isinstance(heads_decoder, ListConfig):
            heads_decoder = list(heads_decoder)

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

        self.position_encoding = LayerSkip(
            SinTimeEncoding(seq_len, emb_size, batch_size)
        )
        self.encoder = LayerSkip(
            FlexibleEncoder(
                num_layers=num_layers_encoder,
                dim_in=emb_size,
                dim_qk=hidden_size,
                dim_feedforward=hidden_size,
                mha_instance=MixtureOfHeads,
                mha_instance_kwargs={"n_heads": num_heads_encoder},
                heads=heads_encoder,
                seq_len=seq_len,
                tricks=tricks,
                available_tricks=available_tricks,
            )
        )

        self.enc_dec_hidden = torch.nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(emb_size, emb_size)
        )

        if decoder_input_len is None:
            decoder_input_len = output_size

        self.query = nn.parameter.Parameter(
            data=torch.randn(1, decoder_input_len, emb_size), requires_grad=True
        )

        self.decoder = LayerSkip(
            FlexibleDecoder(
                num_layers=num_layers_decoder,
                dim_in=emb_size,
                dim_qk=hidden_size,
                dim_feedforward=hidden_size,
                mha_instance=MixtureOfHeads,
                mha_instance_kwargs={"n_heads": num_heads_decoder},
                heads=heads_decoder,
                seq_len=seq_len,
            )
        )

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
        pos_encoding_feature = [arch["position_encoding"]]
        encoder_features = self.encoder.layer.get_feature_vector(arch)
        decoder_features = self.decoder.layer.get_feature_vector(arch)
        head_features = self.head.get_feature_vector(arch)
        flex_features = (
            stem_features
            + pos_encoding_feature
            + encoder_features
            + decoder_features
            + head_features
        )
        return flex_features

    def __init_weights__(self):
        """
        Call after sampling random arch and before setting final and clean to
        initialize model weights as in https://arxiv.org/abs/2210.06423
        """
        assert self.can_sample, "Should call before call to `set_final_and_clean`"

        arch = self.get_arch()
        enc_gain = dec_gain = 1
        enc_layers = sum(arch.get("encoder.layer.layers", [1]))
        dec_layers = sum(arch.get("decoder.layer.layers", [1]))

        if arch["encoder"] == 1 and arch["decoder"] == 1:
            enc_gain = sqrt(log(3 * dec_layers) * log(2 * enc_layers) / 3)
            dec_gain = sqrt(log(3 * dec_layers))
        elif arch["encoder"] == 1:
            enc_gain = sqrt(log(2 * enc_layers))
        elif arch["decoder"] == 1:
            dec_gain = sqrt(log(2 * dec_layers))

        def gain_for_key(key: str):
            if not key.startswith("encoder") and not key.startswith("decoder"):
                return None
            if "norm" in key:
                return None
            if not key.endswith("weight"):
                return None
            if "attention.project" in key:
                if key[0] == "e":
                    return enc_gain
                return dec_gain
            if "feed_forward" in key:
                # sequential, return gain only for linear layers
                if key[-8] not in ("1", "5"):
                    return None
                if key[0] == "e":
                    return enc_gain
                return dec_gain

            # k, q, v projection, symbol before 'weight'
            if "attention.heads" in key:
                if key[-8] in ("q", "k"):
                    return 1
                elif key[-8] == "v":
                    if key[0] == "e":
                        return enc_gain
                    return dec_gain
            return None

        for k, v in self.named_parameters():
            gain = gain_for_key(k)
            if gain is not None:
                nn.init.xavier_normal_(v, gain=gain)

    def forward(self, batch, time):
        embedded_sequence = self.stem(batch)
        embedded_sequence = self.position_encoding((time, embedded_sequence))

        encoded_sequence = self.encoder(embedded_sequence)

        query = torch.cat([self.query] * self.batch_size, 0)
        decoder_out_or_encoded_seq = self.decoder((query, encoded_sequence))

        return self.head(decoder_out_or_encoded_seq)


class Query(nn.Module):
    def __init__(self, seq_len, emb_size):
        super().__init__()
        self.query = nn.parameter.Parameter(
            data=torch.randn(1, seq_len, emb_size), requires_grad=True
        )

    def forward(self, batch_size):
        return torch.cat([self.query] * batch_size, 0)


@register_model("FlexibleTransformerDecoder")
@omnimodel(
    [
        (LayerChoice, SinglePathRandom),
        (Repeat, RepeatRandom),
        (RepeatParallel, ParallelRepsRandom),
        (LayerSkip, SkipSampler),
    ]
)
class FlexibleTransformerDecoder(nn.Module):
    """Decoder only transformer searchable model.

    The model consists of stem, decoder and head.
    Decoder support random number of layers and heads per layer.
    Dncoder supports tricks. Instead of usual MHSA operation the following operation is performed.
    Embedding is split on several parts and each part is processed by MHA, GRU ot convolution.
    See README for more information.
    The tricks can be completely turned off, all enabled or all disabled synchronously in all layers or independentyl chosen in each layer.
    Only usual attention heads are supported.
    """

    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        num_embeddings_hidden="auto",
        augmentations=None,
        num_heads_decoder=[1, 2, 4, 8],
        num_layers_decoder=[1, 2, 3, 4],
        dropout=0.1,
        batch_size=None,
        decoder_input_len=[2, 4, 8],
        tricks="all_or_nothing",
        available_tricks=None,
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
        :param num_heads_decoder: possible number of heads per layer in decoder
        :type num_heads_decoder: list if int
        :param num_layers_decoder: possible number of layers in decoder
        :type num_layers_decoder: list if int
        :param dropout: dropout rate
        :type dropout: float
        :param batch_size: batch size
        :type batch_size: int
        :param decoder_input_len: length of learnable sequence passed to decoder
        :type decoder_input_len: int
        :param tricks: tricks and how they are used in encoder
        :type tricks: str literal: "off", "all_or_nothing" or "choice"
        :param available_tricks: a list of available trikcs. All included by default
        :type available_tricks: list of str
        """

        super().__init__()
        self.batch_size = batch_size
        self.accepts_time = True

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

        self.position_encoding = LayerSkip(
            SinTimeEncoding(seq_len, emb_size, batch_size)
        )

        if decoder_input_len is None:
            decoder_input_len = output_size
        if not isinstance(decoder_input_len, list):
            if isinstance(decoder_input_len, ListConfig):
                decoder_input_len = list(decoder_input_len)
            else:
                decoder_input_len = [decoder_input_len]

        self.get_query = LayerChoice(
            [Query(seq_len, emb_size) for seq_len in decoder_input_len]
        )

        self.decoder = FlexibleDecoder(
            num_layers=num_layers_decoder,
            dim_in=emb_size,
            dim_qk=hidden_size,
            dim_feedforward=hidden_size,
            mha_instance=MixtureOfHeads,
            mha_instance_kwargs={"n_heads": num_heads_decoder},
            heads="learnable",
            seq_len=seq_len,
            tricks=tricks,
            available_tricks=available_tricks,
        )

        self.head = Head(emb_size, output_size)

    def get_feature_vector(self, arch):
        """
        returns feature vector: stem features,
                                1 feature - for position encoding,
                                len(decoder_input_len) - for get_query,
                                FlexibleDecoder features,
                                head features
        """
        stem_features = self.stem.get_feature_vector(arch)
        pos_encoding_feature = [arch["position_encoding"]]
        get_query_features = arch["get_query"]
        decoder_features = self.decoder.get_feature_vector(arch)
        head_features = self.head.get_feature_vector(arch)
        flex_decoder_features = (
            stem_features
            + pos_encoding_feature
            + get_query_features
            + decoder_features
            + head_features
        )
        return flex_decoder_features

    def __init_weights__(self):
        """
        Call after sampling random arch and before setting final and clean to
        initialize model weights as in https://arxiv.org/abs/2210.06423
        """
        assert self.can_sample, "Should call before call to `set_final_and_clean`"

        arch = self.get_arch()
        dec_layers = sum(arch.get("decoder.layer.layers", [1]))
        dec_gain = sqrt(log(2 * dec_layers))

        def gain_for_key(key: str):
            if not key.startswith("encoder") and not key.startswith("decoder"):
                return None
            if "norm" in key:
                return None
            if not key.endswith("weight"):
                return None
            if "attention.project" in key:
                return dec_gain
            if "feed_forward" in key:
                # sequential, return gain only for linear layers
                if key[-8] not in ("1", "5"):
                    return None
                return dec_gain

            # k, q, v projection, symbol before 'weight'
            if "attention.heads" in key:
                if key[-8] in ("q", "k"):
                    return 1
                elif key[-8] == "v":
                    return dec_gain
            return None

        for k, v in self.named_parameters():
            gain = gain_for_key(k)
            if gain is not None:
                nn.init.xavier_normal_(v, gain=gain)

    def forward(self, batch, time):
        embedded_sequence = self.stem(batch)
        embedded_sequence = self.position_encoding((time, embedded_sequence))
        query = self.get_query(self.batch_size)
        decoder_out = self.decoder((query, embedded_sequence))
        return self.head(decoder_out)
