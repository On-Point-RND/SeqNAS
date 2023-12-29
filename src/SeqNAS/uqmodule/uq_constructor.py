# The solution is implemented on the basis of this work
# On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty
# https://arxiv.org/abs/2102.11409


import torch
import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import SoftmaxLikelihood

from uqmodule.dkl import DKL, GP, initial_values


class DUE:
    def __init__(
        self,
        problem_type="cls",  # reg or cls (regression or classification)
        n_inducing_points=10,  # a  number of points
        kernel="RBF",  # GP kernel
        num_outputs=1,  # num classes or 1 for eregression
        dataset_size=1,
        batch_size=1,
        device="cpu",
    ):
        assert problem_type in [
            "cls",
            "reg",
        ], f"Uknown problem type {problem_type}. Known types are cls and reg"
        self.problem_type = problem_type
        self.n_inducing_points = n_inducing_points
        self.kernel = kernel
        self.num_outputs = num_outputs
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.device = device
        self.training = True
        self.samples = 16

    def make_model_and_loss(self, feature_extractor, dataloader):
        initial_inducing_points, initial_lengthscale = initial_values(
            dataloader,
            feature_extractor,
            self.n_inducing_points,
            self.batch_size,
            self.device,
        )

        gp = GP(
            num_outputs=self.num_outputs,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel,
        )

        self.model = DKL(feature_extractor, gp)

        if self.problem_type == "reg":
            likelihood = GaussianLikelihood()
            elbo_fn = VariationalELBO(
                likelihood, self.model.gp, num_data=self.dataset_size
            )

        elif self.problem_type == "cls":
            likelihood = SoftmaxLikelihood(
                num_classes=self.num_outputs, mixing_weights=False
            )
            elbo_fn = VariationalELBO(
                likelihood, self.model.gp, num_data=self.dataset_size
            )

        # liklihood should be on the same device as model outputs
        likelihood.to(self.device)
        self.likelihood = likelihood
        self.loss_fn = lambda x, y: -elbo_fn(x, y)

    def __call__(self, x):
        if self.training:
            return self.model(x)
        else:
            return self.inference(x)

    def inference(self, x):
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.samples):
            if self.problem_type == "reg":
                return self._infer_regression(x, self.model)
            if self.problem_type == "cls":
                return self._infer_classification(x, self.model)

    def _infer_regression(self, x):
        pred = self.model(x)
        ol = self.likelihood(pred)
        output = ol.mean
        output_std = ol.stddev
        return {"preds": output, "uncertianity": output_std}

    def _infer_classification(self, x, model):
        y_pred = model(x)
        # Sample softmax values independently for classification at test time
        y_pred = y_pred.to_data_independent_dist()
        # The mean here is over likelihood samples
        output = self.likelihood(y_pred).probs.mean(0)
        uncertainty = -(output * output.log()).sum(1)
        return {"preds": output, "uncertianity": uncertainty}

    def to(self, device):
        self.model.to(device)

    def train(self):
        self.training = True
        self.model.train()

    def eval(self):
        self.training = False
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()
