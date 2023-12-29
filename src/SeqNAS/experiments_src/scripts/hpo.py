from SeqNAS.experiments_src.pipeline import get_dataset
from SeqNAS.utils.config import patch_env_config
from SeqNAS.utils.distribute import setup
from SeqNAS.utils.misc import print_arch
from SeqNAS.experiments_src import METRIC_REGISTRY
from SeqNAS.utils.optuna import (
    Objective,
    ParamRepeatPruner,
    save_best_checkpoint,
    save_trials_dataframe,
)

import os
import json
import subprocess
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import torch.multiprocessing as mp
from omegaconf import OmegaConf as omg
from pathlib import Path

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_top_k_models(arches_path, scoring_metric, train_top_k):
    higher_is_better = True
    if not METRIC_REGISTRY[scoring_metric]["metric_class"].higher_is_better:
        higher_is_better = False

    arches_files = os.listdir(arches_path)
    if len(arches_files) < train_top_k:
        to_train_arches_idxs = list(range(len(arches_files)))
        print(
            "WARNING! train_top_k is higher than number of searched arches. "
            "All arches will be trained"
        )
    else:
        scores = []
        for arch_file_name in arches_files:
            arch_path_abs = os.path.join(arches_path, arch_file_name)
            with open(arch_path_abs, "r") as f:
                arch_data = json.load(f)
            scores.append(arch_data["scoring_metric"])
        to_train_arches_idxs = np.array(scores).argsort()
        if higher_is_better:
            to_train_arches_idxs = to_train_arches_idxs[::-1]
        to_train_arches_idxs = to_train_arches_idxs[:train_top_k].tolist()
    return [arches_files[i] for i in to_train_arches_idxs]


def main(
    rank,
    exp_name,
    env_config,
    exp_config,
    hyp_config,
    arch,
    n_trials,
):
    """

    :param rank:
    :param env_config:
    :param exp_config:
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
        study_name="hpo",
        storage="mysql://root@localhost/hpo",
        sampler=sampler,
    )
    pruner = ParamRepeatPruner(study)
    objective = Objective(
        rank,
        exp_name,
        env_config,
        exp_config,
        hyp_config.hyperparams,
        exp_config.model_name,
        pruner,
        arch,
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)


if __name__ == "__main__":
    """
    Hyper-parameter optimization script.
    Train top_k final models with hyperparameter tuning after search
    Run from repository root. Example:

    python -m SeqNAS hpo
    --experiment_path=path_to_exp_dir
    --train_top_k=3
    --n_trials=10
    --epochs_count=10
    --worker_count=4
    --gpu_num=0,1

    """

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        help="path to folder with experiment results, used for final training. "
        "If this parameter is used --experiment_name, --data_path, "
        "--model_name will be skipped.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--env_cfg",
        help="use specific env config instead the one from experiment",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hpo_cfg",
        help="HPO config for optuna",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_top_k",
        help="number of best epoches to train "
        "set this parameter with experiment_path",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_trials",
        help="Number of trials per GPU to select hyperparams",
        type=int,
        required=True,
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
        default=None,
    )
    parser.add_argument(
        "--gpu_num",
        dest="gpu_num",
        help="numbers of gpus. hpo.py supports multigpu; "
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

    env_config_path = options.env_cfg or Path(options.experiment_path) / "config.yaml"
    exp_config_path = Path(options.experiment_path) / "exp_cfg.yaml"
    env_config = omg.load(env_config_path)
    exp_config = omg.load(exp_config_path)
    hyp_config = omg.load(options.hpo_cfg)

    if options.epochs_count:
        exp_config.trainer.epochs = options.epochs_count
    exp_config.trainer.use_amp = options.use_amp

    arches_path = os.path.join(options.experiment_path, "arches")
    env_config = patch_env_config(
        env_config, gpu_num=options.gpu_num, worker_count=options.worker_count
    )

    n_trials = options.n_trials
    train_top_k = options.train_top_k

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

    exp_name = os.path.join(
        exp_config.experiment_name,
        "hpo",
        exp_config.model_name,
        f"{datetime.now():%F_%T}",
    )

    # create db
    subprocess.run(
        'service mysql start && mysql -e "CREATE DATABASE IF NOT EXISTS hpo"',
        check=True,
        shell=True,
    )

    top_k_file_names = get_top_k_models(
        arches_path, exp_config.trainer.scoring_metric, train_top_k
    )
    for top_i, arch_name in enumerate(top_k_file_names):
        arch_path_abs = os.path.join(arches_path, arch_name)
        with open(arch_path_abs, "r") as f:
            best_arch_data = json.load(f)

        print(f"-----BEST MODEL TOP {top_i + 1}-----:")
        print(f"model: {best_arch_data['model']}")
        print(f"n_params: {best_arch_data['params']}")
        print(
            f"scoring_metric {exp_config.trainer.scoring_metric}: {best_arch_data['scoring_metric']}"
        )
        print_arch(best_arch_data["arch"])

        try:
            optuna.delete_study(study_name="hpo", storage="mysql://root@localhost/hpo")
        except KeyError:
            pass

        direction = "maximize"
        if not METRIC_REGISTRY[exp_config.trainer.scoring_metric][
            "metric_class"
        ].higher_is_better:
            direction = "minimize"

        study = optuna.create_study(
            storage="mysql://root@localhost/hpo",
            study_name="hpo",
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
                    best_arch_data["arch"],
                    n_trials,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # save trials as dataframe to analyse and print param importance
        study = optuna.load_study(
            study_name="hpo",
            storage="mysql://root@localhost/hpo",
        )

        log_dir = f"top_{top_i}"
        full_log_path = os.path.join(env_config.EXPERIMENT.DIR, exp_name, log_dir)
        os.makedirs(full_log_path, exist_ok=True)

        with open(os.path.join(full_log_path, "arch_from_search_logs.json"), "w") as f:
            json.dump(best_arch_data, f)

        save_trials_dataframe(
            study,
            filename=os.path.join(full_log_path, "study_res.csv"),
        )
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(os.path.join(full_log_path, "param_importance.png"))
        except ValueError as e:
            print(e)

        # save best checkpoint
        save_best_checkpoint(
            checkpoint_tmp_dir=os.path.join(
                env_config.EXPERIMENT.DIR, exp_name, "tmp_checkpoints"
            ),
            scoring_metric=exp_config.trainer.scoring_metric,
            ckpt_dir=full_log_path,
        )

        optuna.delete_study(study_name="hpo", storage="mysql://root@localhost/hpo")

    # delete db
    subprocess.run('mysql -e "DROP DATABASE hpo"', check=True, shell=True)
