import os
from .metrics import get_metrics_and_loss
from .pipeline import prepare_experiment
from ..datasets.timeseries_dataset import TimeSeriesInMemoryDataset

from ..uqmodule.ensembler import Ensembler
from ..uqmodule.eval_ensemles import eval_ensemble
from ..models.modules.augmenter import Augmenter
from ..models.modules.augmentations import (
    ts_mul_noise,
    random_window_transform_big,
    random_window_transform_small,
    zero_noise,
    block_permute,
    total_permute,
    zero_noise,
)


def validate(model, trainer, logger):
    trainer.reset_metrics("validation")
    trainer.set_model(model, {"main": model.parameters()})
    trainer._iterate_one_epoch(phase="validation")
    metrics = trainer.get_last_epoch_metrics("validation")
    metric_string = " | ".join([f"{key}:{value:.3f}" for key, value in metrics.items()])
    trainer._log(f"{'VALIDATION'.upper()} {metric_string}")


def run(
    ModelInstance,
    env_config,
    exp_cfg,
    experiment_name,
    DatasetInstance=TimeSeriesInMemoryDataset,
    number_of_models=1,
):
    """
    Train model with passed parameters
    """

    TARGET = exp_cfg.dataset.TARGET
    device = env_config.HARDWARE.GPU
    # scoring_metric = exp_cfg.trainer.scoring_metric

    criterion, metrics = get_metrics_and_loss(
        exp_cfg.trainer.loss,
        exp_cfg.trainer.metrics + ["max_prob", "entropy", "confidence"],
        TARGET,
    )

    logger, model, trainer, loaders = prepare_experiment(
        ModelInstance,
        DatasetInstance,
        env_config,
        exp_cfg,
        metrics,
        criterion,
        experiment_name,
        optimizer={
            "main": (
                "ADAM",
                {
                    "lr": exp_cfg.trainer.lr,
                    "weight_decay": exp_cfg.trainer.weight_decay,
                },
            )
        },
    )
    trainer.set_dataloaders(loaders)

    augs_normal = Augmenter(
        cat_features=exp_cfg.dataset.categorical,
        real_features=exp_cfg.dataset.continious,
        cat_transforms=[block_permute],
        real_transforms=[random_window_transform_small],
        seq_len=exp_cfg.dataset.seq_len,
    )

    augs_ood_light = Augmenter(
        cat_features=exp_cfg.dataset.categorical,
        real_features=exp_cfg.dataset.continious,
        cat_transforms=[block_permute, block_permute],
        real_transforms=[random_window_transform_big, block_permute],
        seq_len=exp_cfg.dataset.seq_len,
    )

    augs_ood = Augmenter(
        cat_features=exp_cfg.dataset.categorical,
        real_features=exp_cfg.dataset.continious,
        cat_transforms=[zero_noise, block_permute, total_permute],
        real_transforms=[total_permute, zero_noise],
        seq_len=exp_cfg.dataset.seq_len,
    )

    ensemble = Ensembler(
        model, number_of_models, problem_type="cls", augmentations=augs_normal
    )

    for i in range(number_of_models):
        trainer.set_model(ensemble, {"main": ensemble.parameters()})
        trainer.train()
        ensemble.next()

    ensemble.set_all_trained()

    history = trainer.get_history()

    trainer._log("#### In data validation scores UQ mode True")
    for k in history:
        for i, value in enumerate(history[k]):
            logger.log_metrics("Validation scores", {k: value}, i)
        print(k, max(history[k]))

    trainer._log("#### In data validation scores UQ mode False")
    ensemble.set_uq_mode(False)
    validate(ensemble, trainer, logger)

    ensemble.augmentations = augs_ood
    ensemble.set_uq_mode(True)
    trainer._log("#### OOD data validation scores UQ mode True")
    validate(ensemble, trainer, logger)

    experiement_path = os.path.join(env_config.EXPERIMENT.DIR, experiment_name)
    augmentations = ((augs_normal, 0), (augs_ood, 1), (augs_ood_light, 1))
    eval_ensemble(
        ensemble, loaders["validation"], device, augmentations, experiement_path
    )
