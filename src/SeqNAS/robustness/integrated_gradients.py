import copy
import torch
from tqdm import tqdm
from .common import _data_to_device

import warnings
from typing import Tuple, Callable


def get_grads(model, model_input, target_label_idx, corrupted_input):
    model.zero_grad()
    with torch.autograd.set_grad_enabled(True):
        out = model(model_input)["preds"]
        # we want to keep only target value
        out = torch.gather(out, 1, target_label_idx.reshape(len(out), 1))
        assert (
            out[0].numel() == 1
        ), "Cannot take gradient with respect to multiple outputs."
        # computes grdients F(x_pt) for a batch
        grads = torch.autograd.grad(
            torch.unbind(out), corrupted_input, allow_unused=True
        )
    # return a tuple of gradients with respect to inputs
    return grads


def integrated_gradients(
    model,
    linear_names,
    categorical_names,
    data,
    baseline,
    alphas,
    device,
    single_tensor_input=False,
    embeddings_name="categorical_features",
    target_name="target",
):
    """
    The integrated_gradients function computes the importance of each feature in a model's prediction.
    It computes the sum of gradients for each feature over all possible values of that feature, and then averages them.
    The result is a measure of how much each feature contributes to the final prediction across all possible values.


    :param model:  The model to be explained
    :param linear_names: Specify the names of the linear features
    :param categorical_names: Specify the names of the categorical features
    :param data: A batch of data
    :param baseline: A scalar
    :param alphas: A range of different input values
    :param device: Specify the device that should be used
    :param single_tensor_input=False: Indicate that the model input is a dictionary of tensors, as opposed to a single tensor
    :param embeddings_name: Specify the name of the embedding layer in the model
    :param target_name: Specify the name of the target tensor in a batch dictionary
    :return: Two dictionaries, one with the initial gradients and one with the scaled gradients
    """

    grads_initial = dict()
    grads_scaled = dict()
    all_names = linear_names + categorical_names
    if single_tensor_input:
        all_names = ["input"]
        if len(linear_names + categorical_names) > 0:
            warnings.warn(
                f"Single_tensor_input set True. Feature names are not going to be used"
            )

    for feature_name in tqdm(all_names):
        grads_initial[feature_name], grads_scaled[feature_name] = 0, 0

        for i, batch in enumerate(data):
            batch = _data_to_device(batch, device)

            if feature_name in linear_names:
                feature_tensor = batch["model_input"][feature_name]
                scale = feature_tensor - baseline

            if single_tensor_input:
                feature_tensor = batch["model_input"]
                scale = feature_tensor - baseline

            if feature_name in categorical_names:
                embedding = copy.deepcopy(
                    getattr(model, "categorical_features")[feature_name].weight
                )
                baseline_tensor = torch.full_like(embedding, baseline)
                scale = embedding - baseline_tensor

            for alpha in alphas:
                if feature_name in linear_names:
                    corrupted_input = alpha * feature_tensor + (1 - alpha) * baseline
                    corrupted_input.requires_grad_()
                    batch["model_input"][feature_name] = corrupted_input
                    grads = get_grads(
                        model,
                        batch["model_input"],
                        batch[target_name],
                        corrupted_input,
                    )[0]
                    ig_grads = grads.mean(0)
                    ig_grads_scaled = (scale * grads).mean(0)

                if single_tensor_input:
                    corrupted_input = alpha * feature_tensor + (1 - alpha) * baseline
                    corrupted_input.requires_grad_()
                    batch["model_input"] = corrupted_input
                    grads = get_grads(
                        model,
                        batch["model_input"],
                        batch[target_name],
                        corrupted_input,
                    )[0]
                    ig_grads = grads.mean(0)
                    ig_grads_scaled = (scale * grads).mean(0)

                if feature_name in categorical_names:
                    corrupted_embedding = torch.nn.Parameter(
                        alpha * embedding + (1 - alpha) * baseline_tensor
                    )
                    getattr(model, "categorical_features")[
                        feature_name
                    ].weight = corrupted_embedding
                    grads = get_grads(
                        model,
                        batch["model_input"],
                        batch[target_name],
                        corrupted_embedding,
                    )[0]

                    # mean along batch dim
                    ig_grads = grads.mean(-1)
                    ig_grads_scaled = (scale * grads).mean(-1)

                    # embeddings a dictonary with embedding weights
                    getattr(model, embeddings_name)[feature_name].weight = embedding

                grads_initial[feature_name] = +ig_grads
                grads_scaled[feature_name] = +ig_grads_scaled

    return grads_initial, grads_scaled
