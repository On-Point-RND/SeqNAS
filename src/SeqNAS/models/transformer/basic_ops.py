from logging import raiseExceptions
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.fx import Proxy
from apex.normalization import FusedLayerNorm

from ..modules.convolutional import FreqFilter, get_min_seq_len
from ..transformer.attention_masks import (
    cross,
    diag1,
    diag_1,
    bottom,
    top,
    bottom_inv,
    top_inv,
    left,
    right,
    sin_mask,
    cos_mask,
    noise_one,
    noise_two,
    zero,
)


available_attentions = {
    "cross": cross,
    "diag1": diag1,
    "diag_1": diag_1,
    "bottom": bottom,
    "top": top,
    "bottom_inv": bottom_inv,
    "top_inv": top_inv,
    "left": left,
    "right": right,
    "sin_mask": sin_mask,
    "cos_mask": cos_mask,
    "noise_one": noise_one,
    "noise_two": noise_two,
    "zero": zero,
}


class KQVProduct(nn.Module):
    """
    Compute attention mask with traditional linear projection and multiplicate it on value
    """

    def __init__(self, dim_in: int, dim_q: int, dim_k: int, return_att=True):
        super().__init__()

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        self.scale = nn.Parameter(torch.tensor(1.0))

        self.return_att = return_att

    def forward(self, query, key, value, mask=None):
        query = self.q(query)
        key = self.k(key)  # [bs,T,dim_k]
        value = self.v(value)

        scale = query.size(-1) ** 0.5
        attention = query.bmm(key.transpose(-2, -1))

        if mask is not None:
            attention = attention.masked_fill(mask, -torch.tensor(float("inf")))

        attention = F.softmax(attention / scale, dim=-1)
        if self.return_att:
            return self.scale * attention.bmm(value), attention
        else:
            return self.scale * attention.bmm(value)


class KQVPCNN(nn.Module):
    """
    Compute attention mask with convolutional projection and multiplicate it on value
    """

    def __init__(self, dim_in: int, dim_q: int, dim_k: int, return_att=True):
        super().__init__()

        # TODO projections can be optional and learnable
        self.q = nn.Conv1d(dim_in, dim_q, kernel_size=3, padding=3 // 2)
        self.k = nn.Conv1d(dim_in, dim_k, kernel_size=3, padding=3 // 2)
        self.v = nn.Conv1d(dim_in, dim_k, kernel_size=3, padding=3 // 2)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.return_att = return_att

    def forward(self, query, key, value, mask=None):
        query = self.q(query.transpose(1, 2)).transpose(1, 2)
        key = self.k(key.transpose(1, 2)).transpose(1, 2)
        value = self.v(value.transpose(1, 2)).transpose(1, 2)

        scale = query.size(-1) ** 0.5
        attention = query.bmm(key.transpose(-2, -1))

        if mask is not None:
            attention = attention.masked_fill(mask, -torch.tensor(float("inf")))

        attention = F.softmax(attention / scale, dim=-1)
        if self.return_att:
            return self.scale * attention.bmm(value), attention
        else:
            return self.scale * attention.bmm(value)


class FixedAttention(nn.Module):
    """
    Multiplicate fixed attention mask with type mask_type (must be in available_attentions)
    """

    def __init__(self, dim_in: int, dim_k: int, mask_type, return_att=True):
        super().__init__()
        self.v = nn.Linear(dim_in, dim_k)
        self.dim_k = dim_k
        self.mask_type = mask_type
        self.available_masks = available_attentions
        self.return_att = return_att
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.attention_mask = None

    def forward(self, query, key, value, mask=None):
        value = self.v(value)

        if self.attention_mask is None:
            self.__initmask__(value)

        attention = self.attention_mask.mul(self.mask_weights)
        attention = attention.expand(value.size(0), self.l, self.l)

        if mask is not None:
            # mask ex.: mask = torch.tensor([1,0,0,1])
            attention = attention.masked_fill(mask, -torch.tensor(float("inf")))
        attention = F.softmax(attention, dim=-1)

        if self.return_att:
            return self.scale * attention.bmm(value), attention
        else:
            return self.scale * attention.bmm(value)

    def __initmask__(self, value):
        self.l = value.shape[1]
        device = value.device
        self.attention_mask = self.available_masks[self.mask_type](self.l, device)
        self.mask_weights = nn.Parameter(torch.ones(self.l)).to(device)


def init_attentions(heads, dim_in, dim_qk, return_att=True):
    """
    Create attention masks for types in heads

    Args:
        heads (list of str): list of attention masks to create
        dim_in (int): dimensional of timestep
        dim_qk (int): hidden dimensional
        return_att (bool): if True, attention mask will be returned
    """
    attentions = []
    for head_type in heads:
        if not head_type in list(available_attentions.keys()) + [
            "learnable",
            "cnn_projection",
        ]:
            raise Exception(
                f"Attention type {head_type} not in available types {attentions}"
            )

        elif head_type == "learnable":
            head = KQVProduct(dim_in, dim_qk, dim_qk, return_att=return_att)

        elif head_type == "cnn_projection":
            head = KQVPCNN(dim_in, dim_qk, dim_qk, return_att=return_att)
        else:
            head = FixedAttention(dim_in, dim_qk, head_type, return_att=return_att)
        attentions.append(head)
    return attentions


class MHAMixer(nn.Module):
    """
    Multihead attention with types from heads
    """

    def __init__(
        self,
        dim_in,
        dim_qk,
        heads=["learnable", "diag1", "diag-1", "bot", "top", "left", "right"],
    ):
        super().__init__()
        self.heads = nn.ModuleList(init_attentions(heads, dim_in, dim_qk))

        self.project = nn.Linear(len(heads) * dim_qk, dim_in)

    def forward(self, query, key, value, mask=None):
        heads_outputs = []
        attentions = []
        for head in self.heads:
            v, a = head(query, key, value, mask)
            heads_outputs.append(v)
            attentions.append(a)

        return self.project(torch.cat(heads_outputs, dim=-1)), attentions


class ProjectedMHAMLayer(nn.Module):
    def __init__(
        self,
        dim_in: int = 128,
        dim_qk: int = 128,
        seq_len: int = 0,
        dim_feedforward: int = 512,
        heads_periods=[2, 8, 10, 12, 20],
        window_reduction=5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = FusedLayerNorm(dim_qk)
        self.norm_in = FusedLayerNorm(dim_in)

        min_seq_len = get_min_seq_len(seq_len, heads_periods, window_reduction)
        self.feed_forward = FeedForward(dim_in, dim_feedforward, dropout)

        print(f"Min sequence length after projection is {min_seq_len}")

        self.heads_one = nn.ModuleList(
            init_attentions(["learnable"] * len(heads_periods), dim_in, dim_qk)
        )

        self.heads_two = nn.ModuleList(
            init_attentions(["learnable"] * len(heads_periods), dim_qk, dim_qk)
        )

        self.feed_forward_one = FeedForward(dim_qk, dim_qk, dropout)
        self.feed_forward_two = FeedForward(dim_qk, dim_qk, dropout)

        self.project = nn.Linear(len(heads_periods) * dim_qk, dim_in)

        self.frequencies = nn.ModuleList(
            [
                FreqFilter(
                    p,
                    window_reduction,
                    dim_in,
                    seq_len,
                    min_seq_len=min_seq_len,
                )
                for p in heads_periods
            ]
        )

    def forward(self, value, mask=None):
        heads_outputs = []
        attentions = []
        for freq_proj, head_one, head_two in zip(
            self.frequencies, self.heads_one, self.heads_two
        ):
            projected = freq_proj(value).transpose(1, 2)
            projected = self.norm_in(projected)

            v_one, a = head_one(projected, projected, projected, mask)
            v_one = self.feed_forward_one(v_one)
            # v_two, a = head_two(v_one, v_one, v_one, mask)
            # v_two = self.feed_forward_two(v_two)

            heads_outputs.append(self.norm(v_one))
            attentions.append(a)

        return (
            self.project(torch.cat(heads_outputs, dim=-1)) + projected,
            attentions,
        )


class MHASPMixer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_qk,
        heads=["learnable", "diag1", "diag-1", "bot", "top", "left", "right"],
    ):
        super().__init__()
        self.heads = nn.ModuleList(init_attentions(heads, dim_in, dim_qk))

        self.projections = nn.ModuleList([nn.Linear(dim_qk, dim_in)] * len(heads))

    def forward(self, query, key, value, mask):
        heads_outputs = []
        attentions = []
        for head, projection in zip(self.heads, self.projections):
            v, a = head(query, key, value, mask)
            heads_outputs.append(projection(v).unsqueeze(0))
            attentions.append(a)

        # NOTE
        # apply several projections on each head spearetly and sum ouptuts
        # concat and project with a linear layer is doing almost the same anyway
        # but by doing so we get a weighted sum for the heads and estimate ones importance
        return torch.sum(torch.cat(heads_outputs, axis=0), dim=0), attentions


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_feedforward, dropout: float = 0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim_in, dim_feedforward),
            nn.Mish(),
            nn.Linear(dim_feedforward, dim_in),
            nn.Mish(),
        )
        self.norm = FusedLayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_tensor):
        projected = self.projection(in_tensor)
        return self.norm(in_tensor + self.dropout(projected))


class FeedForward2(nn.Module):
    def __init__(self, dim_in, dim_feedforward, dropout: float = 0.1):
        super().__init__()

        self.projection = nn.Sequential(
            FusedLayerNorm(dim_in),
            nn.Linear(dim_in, dim_feedforward),
            nn.Dropout(dropout),
            nn.Mish(),
            FusedLayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, dim_in),
            nn.Dropout(dropout),
        )

    def forward(self, in_tensor):
        return self.projection(in_tensor)


class SinPosEncoding(nn.Module):
    def __init__(self, seq_len: int, dim_model: int):
        super().__init__()
        init_dim_model = dim_model
        dim_model = (
            2 * (dim_model // 2) + 2
        )  # make it always odd by increasing the size
        pos = torch.arange(seq_len).reshape(1, -1, 1)
        dim = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )
        pe = torch.zeros(1, seq_len, dim_model)
        pe[0, :, 0::2] = torch.sin(pos * dim)
        pe[0, :, 1::2] = torch.cos(pos * dim)
        self.pe = pe[:, :, :init_dim_model]

    def forward(self, x):
        return x + self.pe.to(x.device)


class SinTimeEncoding(nn.Module):
    def __init__(self, seq_len: int, dim_model: int, batch_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.init_dim_model = dim_model
        self.dim_model = (
            2 * (dim_model // 2) + 2
        )  # make it always odd by increasing the size
        self.bs = batch_size

    def forward(self, time_and_input):
        time, x = time_and_input
        time = F.pad(time, (0, self.seq_len - time.shape[1]), "constant", 1)
        min_t, max_t = torch.aminmax(time, dim=-1, keepdim=True)
        time = (time - min_t) / (max_t - min_t)
        pos = time[..., None]
        device = x.device
        dim = torch.exp(
            torch.arange(0, self.dim_model, 2) * (-math.log(10000.0) / self.dim_model)
        ).to(device)
        pe = torch.zeros(self.bs, self.seq_len, self.dim_model).to(device)
        pe[:, :, 0::2] = torch.sin(pos * dim)
        pe[:, :, 1::2] = torch.cos(pos * dim)
        pe = pe[:, :, : self.init_dim_model]
        return x + pe


# DONE
class MHALayer(nn.Module):
    def __init__(
        self,
        dim_in: int = 128,
        dim_qk=16,
        dim_feedforward: int = 512,
        heads: list = ["learnable"] * 8,
        mha_instance=MHAMixer,
        dropout: float = 0.1,
        return_att=True,
    ):
        super().__init__()

        self.norm_in_q = FusedLayerNorm(dim_in)
        self.norm_in_k = FusedLayerNorm(dim_in)
        self.norm_in_v = FusedLayerNorm(dim_in)

        self.norm_out = FusedLayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)

        self.attention = mha_instance(dim_in, dim_qk, heads)
        self.return_att = return_att

        self.feed_forward = FeedForward(dim_in, dim_feedforward, dropout)

    def forward(self, q, k, v, mask):
        q = self.norm_in_q(q)
        k = self.norm_in_k(k)
        v = self.norm_in_v(v)

        out, attentions = self.attention(q, k, v, mask)
        out = self.norm_out(self.dropout(out) + q)
        if self.return_att:
            return self.feed_forward(out) + out, attentions

        return self.feed_forward(out) + out


class EmbeddingSplitLayers(nn.Module):
    def __init__(
        self,
        layers: list = ["attention", "gru", "conv"],
        dim_in: int = 128,
        dim_qk=16,
        heads="learnable",
        mha_instance=MHAMixer,
        mha_instance_kwargs=None,
    ):
        super().__init__()
        if mha_instance_kwargs is None:
            mha_instance_kwargs = {}
        self.n_chunks = len(layers)
        emb_sz = [
            sum(it.tolist())
            for it in torch.chunk(torch.tensor([1] * dim_in), self.n_chunks)
        ]
        if len(emb_sz) < self.n_chunks:
            raise ValueError(
                f"Embedding dimensionality {dim_in} is too small for "
                f"{len(layers)} different layer types"
            )

        class Conv1dChannelsLast(nn.Conv1d):
            def forward(self, x):
                return super().forward(x.permute((0, 2, 1))).permute((0, 2, 1))

        class SelfAttention(mha_instance):
            def forward(self, x):
                res = super().forward(x, x, x)
                if isinstance(res, tuple):
                    res = res[0]
                return res

        class GRUAllHiddens(nn.GRU):
            def forward(self, x):
                return super().forward(x)[0]

        self.layers = nn.ModuleList()
        for layer_type, dim in zip(layers, emb_sz):
            if layer_type == "attention":
                self.layers.append(
                    SelfAttention(dim, dim_qk, heads, **mha_instance_kwargs)
                )
            elif layer_type == "gru":
                self.layers.append(GRUAllHiddens(dim, dim, batch_first=True))
            elif layer_type == "conv":
                self.layers.append(Conv1dChannelsLast(dim, dim, 3, padding="same"))
            else:
                raise ValueError(f"Unsupported layer type {layer_type}")

    def forward(self, x):
        xs = torch.chunk(x, self.n_chunks, dim=-1)
        if isinstance(xs, Proxy):
            xs = [xs] * len(self.layers)

        outs = []
        for inp, layer in zip(xs, self.layers):
            outs.append(layer(inp))
        return torch.cat(outs, dim=-1)


class EncoderMHALayer(nn.Module):
    def __init__(
        self,
        dim_in: int = 128,
        dim_qk=16,
        dim_feedforward: int = 512,
        heads: list = ["learnable"] * 8,
        mha_instance=MHAMixer,
        mha_instance_kwargs={},
        dropout: float = 0.1,
        return_att=True,
    ):
        super().__init__()

        self.pre_norm = FusedLayerNorm(dim_in)

        self.norm_out = FusedLayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)

        self.attention = mha_instance(dim_in, dim_qk, heads, **mha_instance_kwargs)
        self.return_att = return_att

        self.feed_forward = FeedForward2(dim_in, dim_feedforward, dropout)

    def forward(self, x):
        y = self.pre_norm(x)
        y, attentions = self.attention(y, y, y)
        y = self.dropout(y) + x

        z = self.norm_out(y)
        z = self.feed_forward(z) + y

        if self.return_att:
            return z, attentions
        return z


class EncoderOmniLayer(nn.Module):
    def __init__(
        self,
        dim_in: int = 128,
        dim_qk=16,
        dim_feedforward: int = 512,
        heads: list = "learnable",
        mha_instance=MHAMixer,
        mha_instance_kwargs={},
        dropout: float = 0.1,
        layers=None,
    ):
        super().__init__()

        if layers is None:
            layers = ["attention"]

        self.pre_norm = FusedLayerNorm(dim_in)
        self.split_layer = EmbeddingSplitLayers(
            layers,
            dim_in,
            dim_qk,
            heads,
            mha_instance,
            mha_instance_kwargs,
        )
        self.norm_out = FusedLayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward2(dim_in, dim_feedforward, dropout)

    def forward(self, x):
        y = self.pre_norm(x)
        y = self.split_layer(y)
        y = self.dropout(y) + x

        z = self.norm_out(y)
        z = self.feed_forward(z) + y

        return z


class DecoderMHALayer(nn.Module):
    def __init__(
        self,
        dim_in: int = 128,
        dim_qk=16,
        dim_feedforward: int = 512,
        heads: list = ["learnable"] * 8,
        mha_instance=MHAMixer,
        mha_instance_kwargs={},
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm_in_1 = FusedLayerNorm(dim_in)
        self.norm_in_2_q = FusedLayerNorm(dim_in)
        self.norm_in_2_kv = FusedLayerNorm(dim_in)
        self.norm_out = FusedLayerNorm(dim_in)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.self_attention = mha_instance(dim_in, dim_qk, heads, **mha_instance_kwargs)

        # enc_attention works only with learnable heads
        heads = "learnable"
        self.enc_attention = mha_instance(dim_in, dim_qk, heads, **mha_instance_kwargs)

        self.feed_forward = FeedForward2(dim_in, dim_feedforward, dropout)

    def forward(self, x):
        decoder_input, encoder_output = x

        y = self.norm_in_1(decoder_input)
        y, _ = self.self_attention(y, y, y)
        y = self.dropout_1(y) + decoder_input

        q = self.norm_in_2_q(y)
        kv = self.norm_in_2_kv(encoder_output)
        z, _ = self.enc_attention(q, kv, kv)
        z = self.dropout_2(z) + y

        w = self.norm_out(z)
        w = self.feed_forward(w) + z

        return w, encoder_output


class DecoderOmniLayer(nn.Module):
    def __init__(
        self,
        dim_in: int = 128,
        dim_qk=16,
        dim_feedforward: int = 512,
        heads: list = "learnable",
        mha_instance=MHAMixer,
        mha_instance_kwargs={},
        dropout: float = 0.1,
        layers=["attention"],
    ):
        super().__init__()

        self.norm_in_1 = FusedLayerNorm(dim_in)
        self.norm_in_2_q = FusedLayerNorm(dim_in)
        self.norm_in_2_kv = FusedLayerNorm(dim_in)
        self.norm_out = FusedLayerNorm(dim_in)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.split_layer = EmbeddingSplitLayers(
            layers,
            dim_in,
            dim_qk,
            heads,
            mha_instance,
            mha_instance_kwargs,
        )

        # enc_attention works only with learnable heads
        heads = "learnable"
        self.enc_attention = mha_instance(dim_in, dim_qk, heads, **mha_instance_kwargs)

        self.feed_forward = FeedForward2(dim_in, dim_feedforward, dropout)

    def forward(self, x):
        decoder_input, encoder_output = x

        y = self.norm_in_1(encoder_output)
        y = self.split_layer(y)
        y = self.dropout_1(y) + encoder_output

        q = self.norm_in_2_q(decoder_input)
        kv = self.norm_in_2_kv(y)
        z, _ = self.enc_attention(q, kv, kv)
        z = self.dropout_2(z) + decoder_input

        w = self.norm_out(z)
        w = self.feed_forward(w) + z

        return w, encoder_output


class FFTEncoder(nn.Module):  # Feed forward transformer
    def __init__(
        self,
        num_layers: int = 2,
        dim_in: int = 128,
        dim_qk=16,
        dim_feedforward: int = 512,
        heads: list = ["learnable"] * 8,
        mha_instance=MHAMixer,
        dropout: float = 0.1,
        seq_len=100,
        position_encoding=None,  # use positional in decoder only
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MHALayer(
                    dim_in,
                    dim_qk,
                    dim_feedforward,
                    heads,
                    mha_instance,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dim_in = dim_in
        self.seq_len = seq_len

        if position_encoding is not None:
            self.pe = position_encoding(seq_len=seq_len, dim_model=dim_in)
        else:
            self.pe = None

        self.position_encoding = position_encoding

    def forward(self, x, mask=None):
        # Use positional in decoder only
        if self.pe is not None:
            x = self.pe(x)

        attentions = []
        for layer in self.layers:
            x, attention = layer(x, x, x, mask)
            attentions.append(attention)

        return x, attentions


class FFTDecoder(nn.Module):  # Feed forward transformer
    def __init__(
        self,
        num_layers: int = 2,
        dim_in: int = 128,
        dim_qk=16,
        dim_feedforward: int = 512,
        heads: list = ["learnable"] * 8,
        mha_instance=MHAMixer,
        dropout: float = 0.1,
        seq_len=100,
        position_encoding=None,  # use positional in decoder only
    ):
        super().__init__()

        self.mha_one = nn.ModuleList(
            [
                MHALayer(
                    dim_in,
                    dim_qk,
                    dim_feedforward,
                    heads,
                    mha_instance,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.mha_two = nn.ModuleList(
            [
                MHALayer(
                    dim_in,
                    dim_qk,
                    dim_feedforward,
                    heads,
                    mha_instance,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.dim_in = dim_in
        self.seq_len = seq_len

        if position_encoding is not None:
            self.pe = position_encoding(seq_len=seq_len, dim_model=dim_in)
        else:
            self.pe = None

    def forward(self, t, enc_seq, mask=None):
        # Use positional in decoder only

        if self.pe is not None:
            enc_seq = self.pe(enc_seq)

        attentions = []
        for mha_one, mha_two in zip(self.mha_one, self.mha_two):
            out, attention = mha_one(t, t, t, mask)
            t, attention = mha_two(out, enc_seq, enc_seq, mask)
            attentions.append(attention)
        return t, attentions
