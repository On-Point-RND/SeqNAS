"""
A wrapper for an RNN cell to make it searchebale.
"""
import torch
import torch.nn as nn
import numpy as np
from .path_features import get_feature_vector, get_paths_limited_by_length
from .model import RecipeModel
from .recipe_generator import RecipeGenerator
from .utils import repackage_hidden
from .. import register_model


@register_model("SearchableRNN")
class SearchableRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings_hidden,
        cat_cardinalities,
        continious,
        seq_len,
        num_embeddings_hidden="auto",
        bidirectional=False,
        dense_dim=1,
        hidden_tuple_size=[2],
        intermediate_vertices=[7],
        min_intermediate_num=2,
        main_operations=[
            "linear",
            "blend",
            "elementwise_prod",
            "elementwise_sum",
        ],
        main_weights=[3.0, 1.0, 1.0, 1.0],
        activations=[
            "activation_tanh",
            "activation_sigm",
            "activation_leaky_relu",
        ],
        activation_weights=[1.0, 1.0, 1.0],
        linear_connections=[2, 3],
        linear_connections_weights=[4, 1],
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embeddings_hidden = embeddings_hidden
        self.num_embeddings_hidden = num_embeddings_hidden
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.dense_dim = dense_dim
        self.ntokens = [-1] * len(continious) + [size for _, size in cat_cardinalities]

        self.generator = RecipeGenerator(
            hidden_tuple_size,
            intermediate_vertices,
            min_intermediate_num,
            main_operations,
            main_weights,
            activations,
            activation_weights,
            linear_connections,
            linear_connections_weights,
        )

        if num_embeddings_hidden == "auto":
            if cat_cardinalities:
                self.num_embeddings_hidden = int(
                    np.mean([size for _, size in cat_cardinalities])
                ) * len(continious)
            else:
                raise ValueError(
                    "Can't automatically determine embedding size for numerical features. "
                    "Specify explicitly `num_embeddings_hidden`."
                )
        else:
            self.num_embeddings_hidden = self.num_embeddings_hidden * len(continious)

        self.max_path_length = 3
        self.path_features = get_paths_limited_by_length(self.max_path_length)
        self.current_recipe = None
        self.model = None
        self.hidden = None

    def _sample_recipe(self):
        good_recipe = False
        recipe = None
        while not good_recipe:
            recipe, good_recipe = self.generator.generate_random_recipe()
        return recipe

    def sample_random(self):
        self.current_recipe = self._sample_recipe()

    def get_arch(self):
        return self.current_recipe

    def set_arch(self, arch):
        self.current_recipe = arch
        self.model = RecipeModel(
            recipes=[self.current_recipe],
            bidirectional=self.bidirectional,
            ntokens=self.ntokens,
            emb_dim=self.embeddings_hidden,
            num_emb_dim=self.num_embeddings_hidden,
            nhid=self.hidden_size,
            dense_dim=self.dense_dim,
            out_dim=self.output_size,
            verbose=False,
        )

    def forward(self, batch):
        batch = torch.stack([x.T for x in list(batch.values())]).T
        batch = batch.permute(1, 0, 2)
        if self.hidden is None:
            self.hidden = self.model.init_hidden(batch.shape[1])
        output, self.hidden = self.model(batch, self.hidden, return_h=False)
        output = output[-1]
        output = self.model.decoder(output)
        self.hidden = repackage_hidden(self.hidden)
        return {"preds": output}

    def set_final_and_clean(self):
        return

    def reset_weights(self):
        return

    def _get_weights(self):
        return self.current_recipe

    def get_feature_vector(self, arch):
        features = get_feature_vector([arch], self.path_features)
        features = features.tolist()[0]
        return features
