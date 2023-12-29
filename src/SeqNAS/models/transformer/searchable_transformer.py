from ...search_spaces.omnimodels import omnimodel
from ...search_spaces.basic_ops import LayerChoice, Repeat
from ...search_spaces.multi_trail_model import (
    DiffLayerGumbel,
    DiffLayerSoftMax,
    IdentityLayer,
    SinglePathRandom,
    CountReLU,
    ParallelRepsRandom,
)
from .transformer_models import (
    EncoderDecoderModel,
    EncoderDecoderProjected,
)
from torch import nn
from ..modules.convolutional import get_min_seq_len

from .searchable_ops import AttentionSelector, MixtureOfHeads
from .basic_ops import (
    MHAMixer,
    FFTEncoder,
    FFTDecoder,
    available_attentions,
    SinPosEncoding,
    ProjectedMHAMLayer,
)
from .. import register_model

HEADS = [
    "learnable",
    "cnn_projection",
    "cross",
    "diag1",
    "diag_1",
    "bottom",
    "top",
    "bottom_inv",
    "top_inv",
    "left",
    "right",
    "sin_mask",
    "cos_mask",
]


@register_model("EncoderDecoderModelPT")
@omnimodel([(LayerChoice, IdentityLayer)])
class EncoderDecoderModelPT(EncoderDecoderModel):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        batch_size,
        augmentations=None,
        dropout=0.1,
        heads_encoder=[
            "learnable",
            "diag1",
            "diag_1",
            "cos_mask",
        ],
        heads_decoder=["learnable"] * 3,
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations=augmentations,
            heads_encoder=heads_encoder,
            heads_decoder=heads_decoder,
            batch_size=batch_size,
        )

        self.encoder = FFTEncoder(
            num_layers=2,
            dim_in=self.input_size,
            dim_qk=hidden_size,
            dim_feedforward=self.input_size,
            heads=heads_encoder,
            mha_instance=AttentionSelector,
            dropout=dropout,
            position_encoding=None,  # SinPosEncoding,
            seq_len=seq_len,
        )

        self.decoder = FFTDecoder(
            num_layers=2,
            dim_in=self.input_size,
            dim_qk=hidden_size,
            dim_feedforward=self.input_size,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=None,
            seq_len=seq_len,
        )


@register_model("EncoderDecoderModelDiff")
@omnimodel([(LayerChoice, DiffLayerGumbel)])
class EncoderDecoderModelDiff(EncoderDecoderModel):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        batch_size,
        augmentations=None,
        dropout=0.1,
        heads_encoder=[
            "learnable",
            "diag1",
            "diag_1",
            "cos_mask",
        ],
        heads_decoder=["learnable"] * 3,
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations=augmentations,
            heads_encoder=heads_encoder,
            heads_decoder=heads_decoder,
            batch_size=batch_size,
        )

        self.encoder = FFTEncoder(
            num_layers=2,
            dim_in=self.input_size,
            dim_qk=hidden_size,
            dim_feedforward=self.input_size,
            heads=heads_encoder,
            mha_instance=AttentionSelector,
            dropout=dropout,
            position_encoding=None,  # SinPosEncoding,
            seq_len=seq_len,
        )

        self.decoder = FFTDecoder(
            num_layers=2,
            dim_in=self.input_size,
            dim_qk=hidden_size,
            dim_feedforward=self.input_size,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=None,
            seq_len=seq_len,
        )


@register_model("EncoderDecoderModelRandom")
@omnimodel([(LayerChoice, SinglePathRandom), (nn.ReLU, CountReLU)])
class EncoderDecoderModelRandom(EncoderDecoderModel):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        batch_size,
        dropout=0.1,
        augmentations=None,
        heads_encoder=[
            "learnable",
            "diag1",
            "diag_1",
            "cos_mask",
        ],
        heads_decoder=["learnable"] * 3,
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            dropout=dropout,
            augmentations=augmentations,
            heads_encoder=heads_encoder,
            heads_decoder=heads_decoder,
        )

        self.batch_size = batch_size
        input_size = (
            len(cat_cardinalities) * embeddings_hidden
            + len(continious) * embeddings_hidden
        )

        self.encoder = FFTEncoder(
            num_layers=2,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=heads_encoder,
            mha_instance=AttentionSelector,
            dropout=dropout,
            position_encoding=None,
            seq_len=seq_len,
        )

        self.decoder = FFTDecoder(
            num_layers=1,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=None,
            seq_len=seq_len,
        )


@register_model("EncoderDecoderModelRepeated")
@omnimodel([(Repeat, ParallelRepsRandom), (nn.ReLU, CountReLU)])
class EncoderDecoderModelRepeated(EncoderDecoderModel):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        batch_size,
        augmentations=None,
        n_heads_encoder=[1, 2, 4, 8],
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations=augmentations,
        )

        self.batch_size = batch_size
        input_size = (
            len(cat_cardinalities) * embeddings_hidden
            + len(continious) * embeddings_hidden
        )

        self.encoder = FFTEncoder(
            num_layers=2,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=n_heads_encoder,
            mha_instance=MixtureOfHeads,
            dropout=0.1,
            position_encoding=SinPosEncoding,
            seq_len=seq_len,
        )

        self.decoder = FFTDecoder(
            num_layers=1,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=["learnable"] * 4,
            mha_instance=MHAMixer,
            dropout=0.1,
            position_encoding=None,
            seq_len=seq_len,
        )


@register_model("EncoderDecoderProjectedPT")
@omnimodel([(LayerChoice, IdentityLayer)])
class EncoderDecoderProjectedPT(EncoderDecoderProjected):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=None,
        heads_projections=[1, 3, 5, 7, 9, 11],
        heads_encoder=HEADS,
        heads_decoder=["learnable"] * 7,
        window_reduction=3,
        dropout=0.1,
        batch_size=1,
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations=augmentations,
            heads_encoder=heads_encoder,
            heads_decoder=heads_decoder,
            window_reduction=window_reduction,
            dropout=dropout,
            batch_size=batch_size,
        )

        min_seq_len = get_min_seq_len(seq_len, heads_projections, window_reduction)
        input_size = (
            len(cat_cardinalities) * embeddings_hidden
            + len(continious) * embeddings_hidden
        )

        self.projectedMHA = ProjectedMHAMLayer(
            dim_in=input_size,
            dim_qk=hidden_size,
            seq_len=seq_len,
            dim_feedforward=embeddings_hidden,
            heads_periods=heads_projections,
            window_reduction=window_reduction,
            dropout=dropout,
        )

        self.encoder = FFTEncoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=heads_encoder,
            mha_instance=AttentionSelector,
            dropout=0.1,
            position_encoding=SinPosEncoding,
            seq_len=min_seq_len,
        )

        self.decoder = FFTDecoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=512,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=0.1,
            position_encoding=None,
            seq_len=min_seq_len,
        )


@register_model("EncoderDecoderProjectedRandom")
@omnimodel([(LayerChoice, SinglePathRandom)])
class EncoderDecoderProjectedRandom(EncoderDecoderProjectedPT):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=None,
        heads_projections=[1, 3, 5, 7, 9, 11],
        heads_encoder=HEADS,
        heads_decoder=["learnable"] * 7,
        window_reduction=3,
        dropout=0.1,
        batch_size=1,
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations=augmentations,
            heads_encoder=heads_encoder,
            heads_decoder=heads_decoder,
            window_reduction=window_reduction,
            dropout=dropout,
            batch_size=batch_size,
        )


@register_model("EncoderDecoderProjectedDiff")
@omnimodel([(LayerChoice, DiffLayerSoftMax)])
class EncoderDecoderProjectedDiff(EncoderDecoderProjectedPT):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=None,
        heads_projections=[1, 3, 5, 7, 9, 11],
        heads_encoder=HEADS,
        heads_decoder=["learnable"] * 7,
        window_reduction=3,
        dropout=0.1,
        batch_size=1,
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations=augmentations,
            heads_encoder=heads_encoder,
            heads_decoder=heads_decoder,
            window_reduction=window_reduction,
            dropout=dropout,
            batch_size=batch_size,
        )
