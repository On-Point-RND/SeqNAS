import torch
import numpy as np
import matplotlib.pyplot as plt


# from uqmodule.fc_resnet import FCResNet
from ..uqmodule.uq_constructor import DUE
from ..trainers.trainer import Trainer

from .metrics import get_metrics_and_loss
from .pipeline import prepare_experiment
from ..datasets.timeseries_dataset import TimeSeriesInMemoryDataset


def run(
    ModelInstance,
    env_config,
    exp_cfg,
    experiment_name,
    DatasetInstance=TimeSeriesInMemoryDataset,
):
    """
    Train model with passed parameters
    """

    TARGET = exp_cfg.dataset.TARGET
    device = env_config.HARDWARE.GPU

    _, metrics = get_metrics_and_loss(
        exp_cfg.trainer.loss, exp_cfg.trainer.metrics, TARGET
    )

    criterion = None
    logger, feature_extractor, _, loaders = prepare_experiment(
        ModelInstance,
        DatasetInstance,
        env_config,
        exp_cfg,
        metrics,
        criterion,
        experiment_name,
    )

    due = DUE(
        problem_type="cls",  # reg or cls (regression or classification)
        n_inducing_points=exp_cfg.model.output_size,  # same as number of classes
        num_outputs=exp_cfg.model.output_size,  # num classes or 1 for eregression
        dataset_size=exp_cfg.dataset.size,
        batch_size=exp_cfg.dataset.batch_size,
        device=device,
    )

    due.make_model_and_loss(feature_extractor, loaders["train"])

    def criterion(outputs, batch):
        return due.loss_fn(outputs, batch["target"])

    optimizer = {
        "main": (
            "ADAM",
            {
                "lr": exp_cfg.trainer.lr,
                "weight_decay": exp_cfg.trainer.weight_decay,
            },
        )
    }

    trainer = Trainer(
        criterion,
        metrics=metrics,
        optimizers=optimizer,
        phases=["train", "validation"],
        num_epochs=exp_cfg.trainer.epochs,
        device=device,
        logger=logger,
        metrics_on_train=False,
    )

    trainer.set_model(due, {"main": due.parameters()})
    trainer.set_dataloaders(loaders)
    trainer.train()

    history = trainer.get_history()
    for k in history:
        for i, value in enumerate(history[k]):
            logger.log_metrics("Validation scores", {k: value}, i)
        print(k, max(history[k]))
