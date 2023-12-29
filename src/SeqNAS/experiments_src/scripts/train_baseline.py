from SeqNAS.experiments_src.pipeline import get_dataset
from SeqNAS.utils.config import patch_env_config
from SeqNAS.utils.distribute import setup
from SeqNAS.experiments_src import METRIC_REGISTRY
from SeqNAS.utils.optuna import (
    Objective,
    ParamRepeatPruner,
    save_best_checkpoint,
    save_trials_dataframe,
)

import os
import subprocess
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf
import torch.multiprocessing as mp

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main(
    rank,
    exp_name,
    env_config,
    exp_config,
    hyp_config,
    model_name,
    n_trials,
):
    """

    :param rank:
    :param env_config:
    :param exp_config:
    :param model_name: model name for baseline training. Mutually exclusive with arches_path
    """
    setup(rank, env_config.HARDWARE.GPU)
    if hyp_config.sampler_method == "TPESampler":
        sampler = TPESampler(multivariate=True, group=True, constant_liar=True)
    elif hyp_config.sampler_method == "RandomSampler":
        sampler = RandomSampler()
    elif hyp_config.sampler_method == "CmaEsSampler":
        sampler = CmaEsSampler()
    else:
        raise Exception(
            f"Not supported sampler: {hyp_config.sampler_method}; "
            f"Currenctly supports: [TPESampler, RandomSampler, CmaEsSampler]"
        )

    study = optuna.load_study(
        study_name="baseline",
        storage="mysql://root@localhost/baseline",
        sampler=sampler,
    )
    pruner = ParamRepeatPruner(study)
    study.optimize(
        Objective(
            rank,
            exp_name,
            env_config,
            exp_config,
            hyp_config.hyperparams,
            model_name,
            pruner,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )


if __name__ == "__main__":
    """
    Train baseline model with hyperparameter tuning
    Run from repository root. Example:

    python examples/main_examples/train_baseline.py
    --experiment_name=alpha
    --data_path=/data/alpha/alpha.csv.zip
    --model_name=EncoderDecoderModel
    --epochs_count=5
    --worker_count=4
    --gpu_num=0,1

    """

    parser = ArgumentParser()
    parser.add_argument(
        "--exp_cfg",
        type=str,
        help="path to the exepriment config",
    )
    parser.add_argument(
        "--env_cfg",
        type=str,
        help="path to the environment config",
    )
    parser.add_argument(
        "--hpo_cfg",
        help="HPO config for optuna",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--name",
        dest="name",
        help="name of experiment as it will be saved on disk, "
        "experiment_name field in exp_cfg or exp_cfg file name by default",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        help="path to [.csv, .csv.zip, .parquet] data, "
        "by default read from exp_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="registered model name",
        type=str,
        default="EncoderDecoderModel",
    )
    parser.add_argument(
        "--n_trials",
        dest="n_trials",
        help="Number of trials per GPU to select hyperparams",
        type=int,
    )
    parser.add_argument(
        "--use_amp",
        dest="use_amp",
        help="use mixed precision",
        action="store_true",
    )
    parser.add_argument(
        "--epochs_count",
        dest="epochs_count",
        help="Count of epoches per arch",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        help="Batch size",
        type=int,
    )
    parser.add_argument(
        "--gpu_num",
        dest="gpu_num",
        help="numbers of gpus. train_baselines.py supports multigpu; "
        "example for single GPU: --gpu_num 0;"
        "example for 4 GPUS: --gpu_num 0,1,2,3",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--worker_count",
        dest="worker_count",
        help="number of cpus for data loading per GPU. Advised using less than cpu count",
        type=int,
        default=None,
    )
    options = parser.parse_args()

    env_config = OmegaConf.load(options.env_cfg)
    exp_config = OmegaConf.load(options.exp_cfg)
    hyp_config = OmegaConf.load(options.hpo_cfg)

    name = Path(options.exp_cfg).stem
    if "experiment_name" in exp_config:
        name = exp_config.experiment_name
    name = options.name or name

    exp_config["experiment_name"] = name

    if options.data_path is not None:
        exp_config.dataset.data_path = options.data_path
    if options.model_name is not None:
        exp_config.model_name = options.model_name
    if options.epochs_count is not None:
        exp_config.trainer.epochs = options.epochs_count
    if options.batch_size is not None:
        exp_config.dataset.batch_size = options.batch_size

    env_config = patch_env_config(
        env_config, gpu_num=options.gpu_num, worker_count=options.worker_count
    )

    n_trials = options.n_trials

    # number of processes for multi gpu. process/gpu
    nprocs = 1
    mp.set_start_method(
        "spawn"
    )  # set start method to 'spawn' BEFORE instantiating the queue

    if len(options.gpu_num) > 1:
        nprocs = len(options.gpu_num.split(","))

    os.environ["USE_DDP"] = str(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = env_config.HARDWARE.GPU
    if nprocs == 1:
        env_config.HARDWARE.GPU = "0"

    # create dataset before training
    dataset = get_dataset(exp_config.dataset.dataset_type, exp_config)
    dataset.create_dataset()
    model_name = options.model_name

    exp_name = os.path.join(
        exp_config.experiment_name, "baseline", model_name, f"{datetime.now():%F_%T}"
    )

    # create db and study
    subprocess.run(
        'service mysql start && mysql -e "CREATE DATABASE IF NOT EXISTS baseline"',
        check=True,
        shell=True,
    )
    try:
        optuna.delete_study(
            study_name="baseline", storage="mysql://root@localhost/baseline"
        )
    except KeyError:
        pass

    direction = "maximize"
    if not METRIC_REGISTRY[exp_config.trainer.scoring_metric][
        "metric_class"
    ].higher_is_better:
        direction = "minimize"

    study = optuna.create_study(
        storage="mysql://root@localhost/baseline",
        study_name="baseline",
        direction=direction,
    )

    processes = []
    for rank in range(nprocs):
        p = mp.Process(
            target=main,
            args=(
                rank,
                exp_name,
                env_config,
                exp_config,
                hyp_config,
                model_name,
                n_trials,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # save trials as dataframe to analyse and print param importance
    study = optuna.load_study(
        study_name="baseline",
        storage="mysql://root@localhost/baseline",
    )
    save_trials_dataframe(
        study,
        filename=os.path.join(env_config.EXPERIMENT.DIR, exp_name, "study_res.csv"),
    )
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(
            os.path.join(env_config.EXPERIMENT.DIR, exp_name, "param_importance.png")
        )
    except ValueError as e:
        print(e)

    # save best checkpoint
    save_best_checkpoint(
        checkpoint_tmp_dir=os.path.join(
            env_config.EXPERIMENT.DIR, exp_name, "tmp_checkpoints"
        ),
        scoring_metric=exp_config.trainer.scoring_metric,
    )
    # delete db and study
    optuna.delete_study(
        study_name="baseline", storage="mysql://root@localhost/baseline"
    )
    subprocess.run('mysql -e "DROP DATABASE baseline"', check=True, shell=True)
