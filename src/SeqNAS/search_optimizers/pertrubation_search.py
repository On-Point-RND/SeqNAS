import copy
import torch
from . import register_search_method


@register_search_method("PTSearcher")
class PTSearcher:
    def __init__(
        self,
        model,
        trainer,
        dataloaders,
        scoring_metric="LossMetric",  # Metric to choose best model
        logger=None,
    ):
        """
        Pertrubatoin searcher trains a supernet first and then scores edges (operations)
        by removing each edge (operation) and computing validation scores without it.

        Args:
            model:
                Initial omnimodel (supernet)
            trainer:
                Trainer object
            dataloaders:
                Data loaders for training the supernet
            scoring_metric:
                A metric to choose the best model
        """
        self.model = model
        self.logger = logger

        self.trainer = trainer
        self.train_epochs = self.trainer.num_epochs

        self.supernet_loaders = dataloaders
        self.scoring_metric = scoring_metric

    def search(self, jobs_queue=None, results_queue=None, barrier=None, arches=None):
        """
        The search function is the core of the architecture search. It takes in a supernet and trains it on
        the training data, then uses this trained supernet to generate a set of weighted architectures which are
        then evaluated on validation data. The best scoring architecture is returned along with its score.
        """
        supernet_score = self.train_supernet()
        self.logger.log(f"SUPER NET {self.scoring_metric}: {supernet_score}")
        weighted_arch = self.search_acrh(supernet_score)
        self.log_arch(weighted_arch)

    def train_supernet(self):
        """

        The train_supernet function trains the supernet for a number of epochs equal to self.num_epochs.


        """
        self.trainer.trainer_name = "supernet"
        self.trainer.set_dataloaders(self.supernet_loaders)
        main_parameters = self.model.parameters()
        self.trainer.set_model(
            self.model,
            param_groups={
                "main": main_parameters,
            },
        )
        self.trainer.train()
        self.supernet_score = self.trainer.get_best_epoch_metrics(self.scoring_metric)[
            self.scoring_metric
        ]

        return self.supernet_score

    def search_acrh(self, supernet_score):
        """
        The search_acrh function takes in a supernet_score with all the edges (operations), which is the validation loss (score) of the
        supernet. It then iterates through each operation and each index within that operation to find
        the best possible architecture for that specific operation. The search_acrh function returns a
        dictionary with all of these values.
        """
        self.trainer.set_dataloaders(self.supernet_loaders)
        cached_arch = copy.deepcopy(self.model.get_arch())
        weighted_ach = copy.deepcopy(cached_arch)

        for operation in weighted_ach:
            for idx in range(len(weighted_ach[operation])):
                weighted_ach[operation][0][idx] = -100

        for operation in cached_arch:
            for idx in range(len(cached_arch[operation][0])):
                cached_arch[operation][0][idx] = 0

                self.model.set_arch(cached_arch)
                self.trainer.model = self.model
                self.trainer.model.cuda(self.trainer.device)
                self.trainer._init_optimizers({"main": self.model.parameters()})
                (
                    self.trainer.metrics,
                    self.trainer.metrics_history,
                    self.trainer.loss_func,
                ) = self.trainer._init_metrics_and_loss(
                    self.trainer.init_metrics, self.trainer.criterion
                )
                self.trainer.last_complete_epoch = -1

                self.trainer._iterate_one_epoch("validation")
                self.trainer._compute_metrics("validation")
                self.trainer.reset_metrics()

                arch_eval_score = self.trainer.get_best_epoch_metrics(
                    self.scoring_metric
                )[self.scoring_metric]
                self.logger.log(
                    f"{self.scoring_metric} drop with {operation} {idx} {arch_eval_score}"
                )

                # ASSUMING SCORING FUNCTION IS LOSS
                operation_weight = arch_eval_score - supernet_score

                cached_arch[operation][0][idx] = 1
                weighted_ach[operation][0][idx] = operation_weight

            self.logger.log(weighted_ach[operation])

        return weighted_ach

    def final_train(self, weighted_arch):
        """
        The final_train function trains the final model on the training set.
        It uses a weighted arch from the search phase and trains it for num_epochs epochs.
        """
        # USE THE SAME SUPER NET TRAINER
        self.set_num_epochs(search=False)
        self.trainer.trainer_name = "tune_final"
        self.trainer.log_arch = False
        self.trainer.set_dataloaders(self.train_loaders)
        self.model.set_arch(weighted_arch)
        self.model.set_final_and_clean()
        if self.train_from_scratch:
            self.model.reset_weights()

        self.trainer.set_model(
            self.model,
            param_groups={
                "main": self.model.parameters(),
            },
        )

        self.trainer.train()

        current_score = self.trainer.get_best_epoch_metrics(self.scoring_metric)[
            self.scoring_metric
        ]

        return current_score

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
