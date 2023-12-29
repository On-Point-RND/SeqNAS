import torch
from tqdm import tqdm
from .common import _data_to_device

import warnings


def pertrubation(
    model,
    criterion,
    linear_names,
    categorical_names,
    data,
    baseline,
    alphas,
    device,
    single_tensor_input,
    embeddings_name="categorical_features",
):
    """
    The pertrubation function is used to calculate the perturbation loss (score) for each feature.
    It takes in a model, criterion, list of linear features and categorical features as well as a data loader and baseline value.
    The function returns the pertrubation loss for each feature over all alphas.

    :param model: Pass the model to be evaluated
    :param criterion: Loss function
    :param linear_names: Specify the names of the linear features
    :param categorical_names: Specify which features are categorical
    :param data: Data loader
    :param baseline: A scalar
    :param alphas: Define the range of values to be used in the pertrubation function
    :param device: Specify the device on which the model is run cpu or gpu
    :param single_tensor_input: Determine if the model input is a single tensor or a dictionary of tensors
    :param embeddings_name: Specify the name of the embeddings in your model
    :param : Determine the number of epochs to train the model
    :return: A dictionary with the name of each feature and a list of loss values for each perturbation
    """

    history = dict()
    all_names = categorical_names + linear_names

    if single_tensor_input:
        all_names = ["input"]
        if len(linear_names + categorical_names) > 0:
            warnings.warn(
                f"Single_tensor_input set True. Feature names are not going to be used"
            )

    for feature_name in tqdm(all_names):
        history[feature_name] = [0.0] * len(alphas)
        with torch.no_grad():
            for batch in data:
                batch = _data_to_device(batch, device)
                if feature_name in linear_names:
                    featue_tensor = batch["model_input"][feature_name]

                if single_tensor_input:
                    featue_tensor = batch["model_input"]

                if feature_name in categorical_names:
                    embedding = getattr(model, embeddings_name)[feature_name].weight

                out_base = model(batch["model_input"])
                base_loss = criterion(out_base, batch)
                for idx, alpha in enumerate(alphas):
                    if feature_name in linear_names:
                        batch["model_input"][feature_name] = (
                            alpha * featue_tensor + (1 - alpha) * baseline
                        )
                        out_corrupted = model(batch["model_input"])

                    if single_tensor_input:
                        batch["model_input"] = (
                            alpha * featue_tensor + (1 - alpha) * baseline
                        )
                        out_corrupted = model(batch["model_input"])

                    if feature_name in categorical_names:
                        baseline_tensor = torch.full_like(embedding, baseline)
                        corrupted_embedding = (
                            alpha * embedding + (1 - alpha) * baseline_tensor
                        )
                        getattr(model, embeddings_name)[
                            feature_name
                        ].weight = torch.nn.Parameter(corrupted_embedding)
                        out_corrupted = model(batch["model_input"])
                        getattr(model, embeddings_name)[feature_name].weight = embedding

                    history[feature_name][idx] += base_loss - criterion(
                        out_corrupted, batch
                    )

    return history
