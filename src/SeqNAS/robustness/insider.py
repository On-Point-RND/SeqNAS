from .common import get_alphas
from .integrated_gradients import integrated_gradients
from .pertruabations import pertrubation

import os
import json
import torch
import numpy as np
from typing import Tuple
import warnings


def get_feature_names(model, linear_features_instance, cargorical_features_instance):
    linear_names = []
    categorical_names = []

    if hasattr(model, linear_features_instance):
        linear_names = model.linear_features.keys()
    if hasattr(model, cargorical_features_instance):
        categorical_names = model.categorical_features.keys()

    return list(linear_names), list(categorical_names)


class Insider:
    def __init__(
        self,
        model: torch.nn.Module,
        save_path="./",
        single_tensor_input=False,
        linear_feature_names=[],
        categorical_feature=[],
        inherit_features_from_model=True,
        cargorical_features_instance="categorical_features",
        linear_features_instance="linear_features",
    ):
        """

        :param model:torch.nn.Module: Store the model that is used
        :param save_path='./': Specify the path where you want to save results
        :param single_tensor_input=False: Indicate whether the model expects a single tensor input or multiple inputs
        :param linear_feature_names=[]: Specify the names of the linear features
        :param categorical_feature=[]: Pass the names of the categorical features
        :param inherit_features_from_model=True: Inherit the feature names from the model
        :param cargorical_features_instance='categorical_features': Specify the name of the instance in which to look for categorical features
        :param linear_features_instance='linear_features': Specify the name of the instance in which to look for linear features
        """

        if len(linear_feature_names + categorical_feature) > 0 and single_tensor_input:
            warnings.warn(
                f"Single_tensor_input set True. Feature names are not going to be used"
            )

        self.model = model
        self.save_path = save_path
        self.linear_names, self.categorical_names = [], []

        if inherit_features_from_model:
            self.linear_names, self.categorical_names = get_feature_names(
                model, linear_features_instance, cargorical_features_instance
            )
        else:
            self.linear_names = linear_feature_names
            self.categorical_names = cargorical_features_instance

        self.single_tensor_input = single_tensor_input

        self.cargorical_features_instance = cargorical_features_instance

    def ig_grads(
        self,
        data,
        target_name,
        baseline_min=0,
        baseline__max=1,
        steps=10,
        device="cpu",
    ):
        """
        The ig_grads function computes the integrated gradients for a given data point.
        The function takes as input:
            - A model,
            - The names of the linear features in the model,
            - The names of categorical features in the model, and
            - A data point (a list or numpy array) to compute integrated gradients on.
            - A number of intermediate steps to integrate

             It returns a dictionary with keys corresponding to each feature name and values corresponding to their respective IG scores.

        :param self: Access the class attributes
        :param data: Pass the data to be explained
        :param target_name: Specify the target variable that we want to explain
        :param baseline_min=0: Set the minimum value for the baseline
        :param baseline__max=1: Set the maximum value of the baseline
        :param steps=10: Define how many steps we want to take between the baseline and the maximum value
        :param device='cpu': Specify the device to use for the computation
        :return: A dictionary of the perturbation scores for each feature in the model and importance scores
        """

        self.alphas = get_alphas(baseline_min, baseline__max)
        self.model.to(device)
        grads_initial, grads_scaled = integrated_gradients(
            self.model,
            self.linear_names,
            self.categorical_names,
            data,
            baseline=0,
            alphas=self.alphas,
            device=device,
            single_tensor_input=self.single_tensor_input,
            embeddings_name=self.cargorical_features_instance,
            target_name=target_name,
        )

        for k in grads_initial:
            grads_initial[k] = grads_initial[k].detach().cpu().tolist()
            grads_scaled[k] = grads_scaled[k].detach().cpu().tolist()
        self._save_results(grads_scaled, "pertrubation_scores_scaled.json")
        self._save_results(grads_initial, "pertrubation_scores_initial.json")

        importance = dict()
        for k in grads_initial:
            importance[k] = sum([abs(x) for x in grads_scaled[k]])

        return grads_scaled, importance

    def pt(
        self,
        data,
        criterion,
        baseline_min=0,
        baseline__max=1,
        steps=10,
        device="cpu",
    ):
        """
        The pt function is used to calculate the pertrubation scores for each feature.
        The pertrubation score is defined as the difference between the model's prediction
        with and without perturbations in that feature.

        The function takes
        - A data point,
        - A criterion (loss) function,
        - Baseline (zero or noisy vector) min/max values and number of steps to take
        between them.
        - It returns a dictionary with keys corresponding to features names
        and values corresponding to lists of pertrubation scores for each step. As well as feature importances.


        :param data: Get the input tensor to the model
        :param criterion: Calculate the loss of the model
        :param baseline_min=0: The minimum value of the baseline
        :param baseline__max=1: The maximum value of the baseline
        :param steps=10: Specify the number of intermediate steps to be done
        :return: A dictionary with the perturbation score for each feature. As well as feature importances.

        """

        self.alphas = get_alphas(baseline_min, baseline__max, steps)

        self.model.to(device)
        result = pertrubation(
            self.model,
            criterion,
            self.linear_names,
            self.categorical_names,
            data,
            baseline=0,
            alphas=self.alphas,
            device=device,
            single_tensor_input=self.single_tensor_input,
            embeddings_name=self.cargorical_features_instance,
        )
        for k in result:
            result[k] = [x.detach().cpu().tolist() for x in result[k]]
        self._save_results(result, "pertrubation_scores.json")

        importance = dict()
        for k in result:
            importance[k] = sum(abs(x) for x in result[k])

        return result, importance

    def _save_results(self, result, name):
        path = os.path.join(self.save_path, name)
        with open(path, "w") as fp:
            json.dump(result, fp)
