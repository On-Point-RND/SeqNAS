import torch
from torch import nn
from .basic_ops import init_attentions, available_attentions, KQVProduct
from ...search_spaces.basic_ops import LayerChoice, RepeatParallel
from apex.normalization import FusedLayerNorm


class AttentionSelector(nn.Module):
    def __init__(self, dim_in, dim_qk, heads, *, return_att=True, **kwargs):
        super().__init__()
        self.mixture_of_heads = nn.ModuleList(
            [
                LayerChoice(init_attentions(heads, dim_in, dim_qk, return_att=False))
                for _ in range(len(heads) - 1)
            ]
        )

        self.project = nn.Linear((len(heads) - 1) * dim_qk, dim_in)
        self.return_att = return_att

    def forward(self, query, key, value, mask=None):
        heads_outputs = []
        for mixed_head in self.mixture_of_heads:
            v = mixed_head(query, key, value, mask)
            if isinstance(v, tuple):
                print("TUPLE")
                heads_outputs.append(v[0])
            else:
                heads_outputs.append(v)

        if self.return_att:
            return self.project(torch.cat(heads_outputs, dim=-1)), [None]

        return self.project(torch.cat(heads_outputs, dim=-1))


class MixtureOfHeads(nn.Module):
    """Represents layer with random number of attention heads"""

    def __init__(self, dim_in, dim_qk, heads, *, return_att=True, n_heads=1, **kwargs):
        """Initializes MixtureOfHeads

        :param dim_in: input dimensions
        :type dim_in:
        """
        super().__init__()
        if isinstance(heads, list):
            head = LayerChoice(init_attentions(heads, dim_in, dim_qk, return_att=False))
        else:
            head = init_attentions([heads], dim_in, dim_qk, return_att=False)[0]

        self.heads = RepeatParallel(head, n_heads)
        self.project = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for n in n_heads:
            self.norm[str(n)] = FusedLayerNorm(n * dim_qk)
            self.project[str(n)] = nn.Linear(n * dim_qk, dim_in)

        self.return_att = return_att

    def forward(self, query, key, value, mask=None):
        heads_outputs = self.heads(query, key, value, mask)
        n = self.heads.num_reps
        x = self.norm[str(n)](heads_outputs)
        if self.return_att:
            return self.project[str(n)](x), [None]

        return self.project[str(n)](x)
