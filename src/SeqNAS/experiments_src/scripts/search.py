from SeqNAS.experiments_src.pipeline import prepare_experiment
from SeqNAS.experiments_src.pipeline import get_dataset
from SeqNAS.utils.config import patch_env_config
from SeqNAS.utils.distribute import setup
from SeqNAS.utils.randomness import seed_everything

from SeqNAS.search_optimizers import SEARCH_METHOD_REGISTRY
from SeqNAS.search_optimizers.base_searcher import RandomSearcher

import os
import json
from datetime import datetime
from argparse import ArgumentParser
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from omegaconf import OmegaConf
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main(
    rank,
    name,
    env_config,
    exp_config,
    jobs_queue,
    results_queue,
    barrier,
    seed,
    search_space,
    use_distillation,
):
    # setup the process groups
    setup(rank, env_config.HARDWARE.GPU)
    seed_everything(seed + rank, avoid_benchmark_noise=True)

    dataset_type = exp_config.dataset.dataset_type
    scoring_metric = exp_config.trainer.scoring_metric
    metrics = exp_config.trainer.metrics
    loss = exp_config.trainer.loss

    scheduler = None
    if "scheduler" in exp_config.trainer:
        scheduler = {
            "main": (
                exp_config.trainer.scheduler,
                dict(exp_config.trainer.sched_params),
            )
        }

    if "optimizer" not in exp_config.trainer:
        optimizer = {"main": ("FUSEDADAM", {"lr": 3e-4, "weight_decay": 1e-5})}
    else:
        optimizer = {
            "main": (
                exp_config.trainer.optimizer,
                dict(exp_config.trainer.optim_params),
            )
        }

    logger, model, trainer, loaders = prepare_experiment(
        ModelInstance=exp_config.model_name,
        DatasetInstance=dataset_type,
        env_config=env_config,
        exp_cfg=exp_config,
        metrics=metrics,
        citerion=loss,
        experiement_name=name,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    searcher = SEARCH_METHOD_REGISTRY[exp_config.search_method.name](
        model=model,
        trainer=trainer,
        dataloaders=loaders,
        scoring_metric=scoring_metric,
        logger=logger,
        **exp_config.search_method.params,
    )

    if search_space is not None:
        searcher.set_space(search_space)

    if use_distillation:
        searcher.enable_distillation()

    searcher.search(jobs_queue=jobs_queue, results_queue=results_queue, barrier=barrier)


if __name__ == "__main__":
    """
    Main Search script. Example:

    python examples/main_examples/search.py
    --experiment_name=vtb_random
    --epochs_count=10
    --worker_count=0
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
        help="path to [.csv, .csv.zip, .parquet] data, by default read from exp_cfg",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="registered model name, overwrites the one in exp_cfg",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--epochs_count",
        dest="epochs_count",
        help="Count of epoches per arch, overwrites the number in exp_cfg",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        help="batch size per GPU, overwrites the one in exp_cfg",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gpu_num",
        dest="gpu_num",
        help="numbers of gpus. Project supports multi gpu and ddp (for diffnas and ptb); "
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
    parser.add_argument(
        "--search_space",
        dest="search_space_fname",
        type=str,
        default=None,
        help="JSON-file containing architectures",
    )
    parser.add_argument(
        "--use_distillation",
        dest="use_distillation",
        help="use knowledge distillation during search",
        action="store_true",
    )
    options = parser.parse_args()

    env_config = OmegaConf.load(options.env_cfg)
    exp_config = OmegaConf.load(options.exp_cfg)

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

    assert exp_config.search_method.name in SEARCH_METHOD_REGISTRY, (
        f"Not supported search method, currently supports: "
        f"{list(SEARCH_METHOD_REGISTRY.keys())}"
    )

    name = os.path.join(
        name,
        exp_config.search_method.name,
        exp_config.model_name,
        f"{datetime.now():%F_%T}",
    )

    # number of processes for multi gpu (process per gpu)
    nprocs = 1
    if len(env_config.HARDWARE.GPU) > 1:
        nprocs = len(env_config.HARDWARE.GPU.split(","))

    mp.set_start_method(
        "spawn"
    )  # set start method to 'spawn' BEFORE instantiating the queue

    if exp_config.search_method.name in ["Hyperband", "RandomSearcher", "Bananas"]:
        # TODO add support of results_queue to all methods
        jobs_queue = Queue()
        results_queue = None
        if exp_config.search_method.name == "Hyperband":
            results_queue = mp.Queue()

        barrier = None
        if nprocs > 1:
            barrier = mp.Barrier(nprocs)

        # use multi gpu mode
        os.environ["USE_DDP"] = str(0)
    elif exp_config.search_method.name in ["DiffSearcher", "PTSearcher"]:
        jobs_queue = None
        results_queue = None
        barrier = None
        # use ddp mode
        os.environ["USE_DDP"] = str(1)
    else:
        raise Exception("Specify multi-gpu or ddp for your search method in code above")

    if not issubclass(
        SEARCH_METHOD_REGISTRY[exp_config.search_method.name], RandomSearcher
    ):
        raise ValueError(
            f"Knowledge distillation is not supported for {exp_config.search_method.name}"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = env_config.HARDWARE.GPU

    if nprocs == 1:
        env_config.HARDWARE.GPU = "0"

    # create dataset before training
    dataset = get_dataset(exp_config.dataset.dataset_type, exp_config)
    dataset.create_dataset()

    search_space = None
    if "search_space" in options:
        with open(options.search_space, "r") as f:
            search_space = json.load(f)

    processes = []
    for rank in range(nprocs):
        p = mp.Process(
            target=main,
            args=(
                rank,
                name,
                env_config,
                exp_config,
                jobs_queue,
                results_queue,
                barrier,
                options.seed,
                search_space,
                options.use_distillation,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
