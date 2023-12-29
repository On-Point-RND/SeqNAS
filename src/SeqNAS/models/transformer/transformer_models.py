import torch
from torch import nn
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig

from ..modules.layers import Augmentation
from ...models.common import Dummy

from .basic_ops import (
    MHAMixer,
    MHASPMixer,
    FFTEncoder,
    FFTDecoder,
    ProjectedMHAMLayer,
    SinPosEncoding,
)

from ..modules.convolutional import get_min_seq_len
from .. import register_model


@register_model("EncoderDecoderModel")
class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=None,
        num_layers_encoder=1,
        num_layers_decoder=1,
        heads_encoder=2,
        heads_decoder=2,
        dropout=0.3,
        batch_size=None,
    ):
        if isinstance(heads_encoder, int):
            heads_encoder = ["learnable"] * heads_encoder
        if isinstance(heads_decoder, int):
            heads_decoder = ["learnable"] * heads_decoder

        self.batch_size = batch_size
        super().__init__()
        input_size = len(cat_cardinalities) * embeddings_hidden + len(continious)
        self.input_size = input_size

        self.aug = augmentations
        if augmentations is not None:
            self.aug = augmentations(seq_len)

        self.query = nn.parameter.Parameter(
            data=torch.randn(1, output_size, input_size), requires_grad=True
        )
        self.encoder = FFTEncoder(
            num_layers=num_layers_encoder,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=hidden_size,
            heads=heads_encoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=None,
            seq_len=seq_len,
        )

        self.enc_dec_hidden = torch.nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(input_size, input_size)
        )

        self.decoder = FFTDecoder(
            num_layers=num_layers_decoder,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=hidden_size,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=None,
            seq_len=seq_len,
        )

        self.categorical_features = torch.nn.ModuleDict(modules=None)
        self.linear_features = torch.nn.ModuleDict(modules=None)

        for cat_name, size in cat_cardinalities:
            self.categorical_features[cat_name] = nn.Embedding(
                size + 1, embeddings_hidden
            )

        n_channels = len(continious)
        num_embeddings_hidden = len(continious)
        self.linear_features = nn.ModuleDict({k: Dummy() for k in continious})
        self.linear_prep = torch.nn.Sequential(
            nn.BatchNorm1d(n_channels),
            nn.Conv1d(n_channels, num_embeddings_hidden, kernel_size=3, padding=3 // 2),
            nn.Dropout(dropout),
        )

        self.output = torch.nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size * output_size, output_size),
        )

    def forward(self, batch):
        cat_features = []
        for cat_name in self.categorical_features:
            x = batch[cat_name].type(torch.long)
            emb = self.categorical_features[cat_name](x)
            cat_features.append(emb)

        lin_features = [batch[lin].unsqueeze(1) for lin in self.linear_features]
        lin_features = torch.cat(lin_features, 1)
        lin_features = self.linear_prep(lin_features).transpose(1, 2)
        features = torch.cat([*cat_features, lin_features], 2)

        # shape features = list(elem * (num_cat_features + num_lin_features) )
        # shape elem = batch x max_len x embeddings_hidden

        if not self.aug is None:
            embdeded_sequence = self.aug(features, self.training)
        else:
            embdeded_sequence = features
        # embdeded_sequence = torch.cat(features, 2)

        encoded_sequence, attentions_e = self.encoder(embdeded_sequence, mask=None)
        encoded_sequence = self.enc_dec_hidden(encoded_sequence)
        # encoded_sequence shape (batch_size, seq_len, features)

        # TODO only one target token CLS
        # only one learnable querry with respect to all the values
        # check the paper one more time
        query = torch.cat([self.query] * self.batch_size, 0)
        decoder_output, attentions_d = self.decoder(query, encoded_sequence, mask=None)
        # decoder_output shape (batch_size, output_size, input_size)

        decoder_output = decoder_output.reshape(self.batch_size, -1)
        return {
            "preds": self.output(decoder_output),
            "attention": (attentions_e, attentions_d),
        }


@register_model("EncoderDecoderModelMHASP")
class EncoderDecoderModelMHASP(EncoderDecoderModel):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        heads_encoder=["learnable"] * 6,
        heads_decoder=["learnable"] * 4,
        augmentations=None,
    ):
        super().__init__(
            hidden_size,  # dim_k size
            output_size,
            embeddings_hidden,
            cat_cardinalities,
            continious,
            seq_len,
            augmentations,
        )

        input_size = len(cat_cardinalities) * embeddings_hidden + len(continious) * 10

        self.encoder = FFTEncoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=heads_encoder,
            mha_instance=MHASPMixer,
            dropout=0.1,
            position_encoding=None,
            seq_len=seq_len,
        )

        self.decoder = FFTDecoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=heads_decoder,
            mha_instance=MHASPMixer,
            dropout=0.1,
            position_encoding=None,
            seq_len=seq_len,
        )


@register_model("EncoderDecoderModelSpeech")
class EncoderDecoderModelSpeech(nn.Module):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=None,  # Augmentation(),
        heads_encoder=["learnable"] * 6,
        heads_decoder=["learnable"] * 4,
    ):
        super().__init__()

        self.bnn_one = nn.BatchNorm1d(1)
        self.bnn_two = nn.BatchNorm1d(5)

        self.aug = augmentations
        if augmentations is not None:
            self.aug = augmentations(seq_len)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=2, padding=2 // 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(5, 5, kernel_size=4, padding=2 // 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(5, 5, kernel_size=4, padding=2 // 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(5, 5, kernel_size=4, padding=2 // 2, stride=2),
            nn.ReLU(),
        )

        kernels = [5, 7, 11, 15]
        stride = 4
        cout = 10

        self.cnn_features = nn.ModuleList(
            [
                nn.Conv1d(5, cout, kernel_size=k, padding=k // 2, stride=stride)
                for k in kernels
            ]
        )

        input_size = cout * len(kernels)

        self.query = nn.parameter.Parameter(
            data=torch.randn(1, 3, input_size), requires_grad=True
        )
        self.encoder = FFTEncoder(
            num_layers=8,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=512,
            heads=heads_encoder,
            mha_instance=MHAMixer,
            dropout=0.1,
            position_encoding=None,
            seq_len=seq_len,
        )

        self.enc_dec_hidden = torch.nn.Sequential(nn.Dropout(0.1))

        self.decoder = FFTDecoder(
            num_layers=4,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=512,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=0.1,
            position_encoding=None,
            seq_len=seq_len,
        )

        self.output = torch.nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(input_size, output_size)
        )
        self.masks = None

    def forward(self, batch):
        features = []

        x = self.bnn_one(batch.unsqueeze(1))
        x = self.cnn(x)
        x = self.bnn_two(x)

        for cnn in self.cnn_features:
            features.append(cnn(x))

        embdeded_sequence = torch.cat(features, 1)

        if not self.aug is None:
            embdeded_sequence = self.aug(embdeded_sequence, self.training)

        # shape features = list(elem * (num_cat_features + num_lin_features) )

        embdeded_sequence = embdeded_sequence.transpose(1, 2)
        print(embdeded_sequence.shape)

        encoded_sequence, attentions_e = self.encoder(embdeded_sequence, mask=None)
        encoded_sequence = self.enc_dec_hidden(encoded_sequence)

        bs = encoded_sequence.size(0)

        query = torch.cat([self.query] * bs, 0)

        decoder_output, attentions_d = self.decoder(query, encoded_sequence, mask=None)

        decoder_output = decoder_output.mean(1)
        return {
            "preds": self.output(decoder_output),
            "attention": (attentions_e, attentions_d),
        }


@register_model("EncoderDecoderProjected")
class EncoderDecoderProjected(nn.Module):
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
        heads_encoder=[
            "learnable",
            "diag1",
            "diag_1",
            "learnable",
            "learnable",
            "learnable",
        ],
        heads_decoder=["learnable"] * 7,
        window_reduction=3,
        dropout=0.1,
        batch_size=1,
    ):
        super().__init__()

        self.batch_size = batch_size

        input_size = (
            len(cat_cardinalities) * embeddings_hidden
            + len(continious) * embeddings_hidden
        )

        min_seq_len = get_min_seq_len(seq_len, heads_projections, window_reduction)
        self.projectedMHA = ProjectedMHAMLayer(
            dim_in=input_size,
            dim_qk=hidden_size,
            seq_len=seq_len,
            dim_feedforward=embeddings_hidden,
            heads_periods=heads_projections,
            window_reduction=window_reduction,
            dropout=dropout,
        )

        self.query = nn.parameter.Parameter(
            data=torch.randn(1, output_size, input_size), requires_grad=True
        )

        self.encoder = FFTEncoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=heads_encoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=None,
            seq_len=min_seq_len,
        )

        self.drop_relu = torch.nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

        self.decoder = FFTDecoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=hidden_size,
            dim_feedforward=128,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=0.1,
            position_encoding=None,
            seq_len=min_seq_len,
        )

        self.output = torch.nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(input_size, output_size)
        )

        self.categorical_features = torch.nn.ModuleDict(modules=None)
        self.linear_features = torch.nn.ModuleDict(modules=None)

        for cat_name, size in cat_cardinalities:
            self.categorical_features[cat_name] = nn.Embedding(
                size + 1, embeddings_hidden
            )

        for lin_name in continious:
            # linear projection along time axis
            self.linear_features[lin_name] = nn.Conv1d(
                1, embeddings_hidden, kernel_size=3, padding=3 // 2
            )

        self.aug = augmentations

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
        # embdeded_sequence = embdeded_sequence.transpose(1, 2)

        if not self.aug is None:
            embdeded_sequence = self.aug(torch.cat(features, 2), self.training)
        else:
            embdeded_sequence = torch.cat(features, 2)

        # shape elem = batch x max_len x embeddings_hidden
        projected_sequence, attentions_e = self.projectedMHA(
            embdeded_sequence, mask=None
        )
        encoded_sequence = self.drop_relu(projected_sequence)

        encoded_sequence, attentions_e = self.encoder(projected_sequence, mask=None)
        encoded_sequence = self.drop_relu(encoded_sequence)

        query = torch.cat([self.query] * self.batch_size, 0)

        decoder_output, attentions_d = self.decoder(query, encoded_sequence, mask=None)

        decoder_output = decoder_output.mean(1)
        return {
            "preds": self.output(decoder_output),
            "attention": (attentions_e, attentions_d),
        }


@register_model("EncoderDecoderProjectedSpeech")
class EncoderDecoderProjectedSpeech(nn.Module):
    def __init__(
        self,
        hidden_size,  # dim_k size
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        augmentations=None,  # Augmentation,
        heads_periods=[1, 3, 5, 7, 9, 11],
        heads_encoder=["learnable", "learnable", "learnable", "learnable"],
        heads_decoder=["learnable"] * 6,
        window_reduction=50,
        dropout=0.25,
        batch_size=None,
    ):
        super().__init__()

        self.bnn_one = nn.BatchNorm1d(1)
        self.bnn_two = nn.BatchNorm1d(5)

        if augmentations is not None:
            self.aug = augmentations(seq_len)
        else:
            self.aug = None

        self.batch_size = batch_size
        kernels = [1, 3, 5, 7, 11, 15, 25, 35]

        cout = 15

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=2, padding=2 // 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(5, 5, kernel_size=4, padding=2 // 2, stride=1),
        )

        self.cnn_features = nn.ModuleList(
            [nn.Conv1d(5, cout, kernel_size=k, padding=k // 2) for k in kernels]
        )

        input_size = cout * len(kernels)
        seq_len = seq_len // 2

        self.projectedMHA = ProjectedMHAMLayer(
            dim_in=input_size,
            dim_qk=512,
            seq_len=seq_len,
            dim_feedforward=512,
            heads_periods=heads_periods,
            window_reduction=window_reduction,
            dropout=dropout,
        )

        min_seq_len = get_min_seq_len(seq_len, heads_periods, window_reduction)

        self.query = nn.parameter.Parameter(
            data=torch.randn(1, output_size, input_size), requires_grad=True
        )

        self.encoder = FFTEncoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=512,
            dim_feedforward=512,
            heads=heads_encoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=SinPosEncoding,
            seq_len=min_seq_len,
        )

        self.drop_relu = torch.nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

        self.decoder = FFTDecoder(
            num_layers=3,
            dim_in=input_size,
            dim_qk=512,
            dim_feedforward=512,
            heads=heads_decoder,
            mha_instance=MHAMixer,
            dropout=dropout,
            position_encoding=None,
            seq_len=seq_len,
        )

        self.output = torch.nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(input_size, output_size)
        )

    def forward(self, batch):
        features = []
        if self.aug is not None:
            batch = self.aug(batch, self.training)

        x = self.bnn_one(batch.unsqueeze(1))
        x = self.cnn(x)

        for cnn in self.cnn_features:
            features.append(cnn(x))
        embdeded_sequence = torch.cat(features, 1)
        embdeded_sequence = embdeded_sequence.transpose(1, 2)
        # shape elem = batch x max_len x embeddings_hidden
        projected_sequence, attentions_e = self.projectedMHA(
            embdeded_sequence, mask=None
        )
        encoded_sequence = self.drop_relu(projected_sequence)

        encoded_sequence, attentions_e = self.encoder(projected_sequence, mask=None)
        encoded_sequence = self.drop_relu(encoded_sequence)

        query = torch.cat([self.query] * encoded_sequence.size(0), 0)

        decoder_output, attentions_d = self.decoder(query, encoded_sequence, mask=None)

        decoder_output = decoder_output.mean(1)
        return {
            "preds": self.output(decoder_output),
            "attention": (attentions_e, attentions_d),
        }
