import os
import torch
from copy import deepcopy


class Ensembler:
    def __init__(
        self,
        model,
        num_models=4,
        mode="mean",
        problem_type="cls",
        augmentations=[],
    ):
        """

        Basic Ensmebler class, wraps a PyTorch model to mimic models behavior.
        Can be used instead of a PyTorch model almost anywhere.
        Use the next method to train the next model


        :param model: Pass the model that is being trained
        :param num_models=4: Specify the number of models that will be used in the ensemble
        :param mode='mean': Specify the way to combine the predictions of all models. Boosting is available too but it has not been tested!
        :param problem_type='cls': Specify the problem type, only classification is supported at the moment
        :param augmentations=[]: Pass a list of augmentations to the model

        """

        assert mode in [
            "mean",
            "boost",
        ], f"Unknow mode {mode} is provided"
        assert problem_type in [
            "reg",
            "cls",
        ], f"Unknow problem type {problem_type} is provided"
        self.mode = mode
        self.models = self.init_models(model, num_models)
        self.model_to_train = self.models[0]
        self.trained = []
        self.traning = True
        self.problem_type = problem_type
        self.all_trained = False
        self.augmentations = augmentations
        self.uq_mode = True

    def set_uq_mode(self, mode=True):
        """
        The set_uq_mode function sets the uncertainty quantification mode for a given
        model. Is set is True then augmentation will be used during infernce time.
        Often, random augmentations increase ensembles robustness.

            Parameters:

                m : Model object or string representation of model name (e.g., 'myModel')

            Returns: None


        :param mode=True: Set the uq mode to true

        """
        self.uq_mode = mode

    def _aug(self, x):
        if self.augmentations is not None:
            if (not self.uq_mode) and (not self.traning):
                return x
            else:
                x = self.augmentations(x)
        return x

    def init_models(self, model, num_models):
        models = []
        for _ in range(num_models):
            models.append(deepcopy(model))
        return models

    def reset_weights(self, model):
        @torch.no_grad()
        def apply(m):
            for name, child in m.named_children():
                if hasattr(child, "_parameters"):
                    for param_name in child._parameters:
                        # print(name, param_name)
                        if len(child._parameters[param_name].shape) < 2:
                            torch.nn.init.normal_(child._parameters[param_name].data)
                        else:
                            torch.nn.init.xavier_uniform_(
                                child._parameters[param_name].data
                            )
                reset_parameters = getattr(child, "reset_parameters", None)
                if callable(reset_parameters):
                    child.reset_parameters()
                else:
                    apply(child)

        apply(model)
        return model

    def loadweights(self, model, weights_path):
        if os.path.exists(weights_path):
            print("Loading saved model from", weights_path)
            weights = torch.load(weights_path)["state_dict"]
            models = []
            for idx in range(len(weights)):
                models.append(model.load_state_dict(weights[str[idx]]))
            self.trained = models
            self.all_trained = True
        else:
            raise Exception(f"Path to a model does not exist {weights_path}")

    def to(self, device):
        self.model_to_train.to(device)
        for model in self.trained:
            model.to(device)

    def eval(self):
        self.traning = False
        self.model_to_train.eval()
        for model in self.trained:
            model.eval()

    def train(self):
        self.traning = True
        self.model_to_train.train()
        for model in self.trained:
            model.train()

    def _boost(self, x):
        if not self.all_trained:
            preds_main = self.model_to_train(x)["preds"]
            preds = 0
        else:
            preds_main = 0

        with torch.no_grad():
            first = True
            for model in self.trained:
                if first:
                    preds = model(x)["preds"].detach()
                    first = False
                else:
                    preds = preds + model(x)["preds"]

        return (preds + preds_main) / (len(self.trained) + 1)

    def _mean(self, x):
        if not self.traning:
            if not self.all_trained:
                x_aug = self._aug(x)
                preds = self.model_to_train(x_aug)["preds"]
            else:
                preds = 0
            with torch.no_grad():
                for model in self.trained:
                    x_aug = self._aug(x)
                    preds = preds + model(x_aug)["preds"].detach()

            preds = (preds) / (len(self.trained) + 1)
        else:
            if self.all_trained:
                with torch.no_grad():
                    for model in self.trained:
                        x_aug = self._aug(x)
                        preds = preds + model(x_aug)["preds"].detach()
            else:
                x_aug = self._aug(x)
                preds = self.model_to_train(x_aug)["preds"]
        return preds / (len(self.trained) + 1)

    def __call__(self, x):
        if self.mode == "boost":
            out = self._boost(x)
        if self.mode == "mean":
            out = self._mean(x)

        return self.make_output(out)

    def next(self):
        """
        The next function switches to the next model in the list of models.
        If there are no more models, it sets all_trained to True.

        """
        print("Switching to the next model")
        model = self.models.pop(0)
        self.trained.append(model)
        if len(self.models) > 0:
            self.model_to_train = self.reset_weights(self.models[0])
        else:
            self.all_trained = True
            print("All the models were used")

    def state_dict(self):
        state = dict()
        for idx, model in enumerate(self.trained):
            state[f"model_{str(idx)}"] = model.state_dict()
            return state

    def parameters(self):
        return self.model_to_train.parameters()

    def make_output(self, outputs):
        return {
            "preds": outputs,
            "entropy": self.get_enropy(outputs),
            "max_prob": self.max_prob(outputs),
        }

    def set_all_trained(self):
        self.all_trained = True

    def get_enropy(self, outputs):
        """
        The get_enropy function takes in the output of a neural network, and returns
        the entropy of that distribution.

        """
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        return -(outputs * outputs.log()).sum(1)

    def max_prob(self, outputs):
        """
        The max_prob function takes in a tensor of logits and returns the index with the highest probability.
        """
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        assert (
            len(outputs.shape) == 2
        ), f"output shape should be 2, provided shape is {outputs.shape}"
        max_prob, _ = torch.max(outputs, 1)
        return max_prob
