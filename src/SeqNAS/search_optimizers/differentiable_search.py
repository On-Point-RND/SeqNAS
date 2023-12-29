import torch
import torch.nn as nn
from . import register_search_method


@register_search_method("DiffSearcher")
class DiffSearcher:
    def __init__(
        self,
        model,
        trainer,
        dataloaders,
        scoring_metric="LossMetric",  # Metric to choose best model
        logger=None,
    ):
        """
        Diff searcher performs differentiable architechture search using a supernet.
        First we train a supernet and edge values (operator importances).
        Then we select N best edges (operations) and train the final architechture.

        Two trainers are used to train the supernet.

        Args:
            model: Searchable omnimodel
            trainer: Trainer to train the supernet and edge importance
            dataloaders: Pass the dataloaders for the search phase (training a supernet)
            scoring_metric: A metric to choose the best model
            logger: a logger instance
        """

        self.model = model
        self.logger = logger

        self.search_trainer = trainer
        self.search_trainer.set_dataloaders(dataloaders)
        self.scoring_metric = scoring_metric

    def search(self, jobs_queue=None, results_queue=None, barrier=None, arches=None):
        """
        Tune hyperparameters with differential method
        """
        main_parameters = self.model.parameters()
        model_arch = self.model.get_arch()

        alpha_params = {name: nn.Parameter(model_arch[name]) for name in model_arch}

        self.model.set_arch(alpha_params)

        self.search_trainer.set_model(
            self.model,
            param_groups={
                "main": main_parameters,
                "arch": list(alpha_params.values()),
            },
        )
        self.search_trainer.train()

    def log_arch(self, arch):
        """
        The log_arch function is a helper function that logs the architecture of the model.
        It is called in every training loop and it logs all layer names, their value.
        This information can be used to debug any problems with your model's architecture,
        track searching performance and mainly to recover best found architechture.
        """
        for key in arch:
            if isinstance(arch[key], torch.Tensor):
                values = list(arch[key][0].detach().cpu().numpy())
            else:
                values = list(arch[key])
            values = [round(v, 3) for v in values]
            self.logger.log(f"layer: {key}")
            self.logger.log(f"values: {values}")
            values = {str(i): values[i] for i in range(len(values))}
            self.logger.log_scalars(f"Arch/Layer_{key}", values)
