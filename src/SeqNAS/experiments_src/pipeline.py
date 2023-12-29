from ..models.modules.layers import Augmentation
from ..trainers.trainer import Trainer
from ..nash_logging.common import LoggerUnited
from ..models import MODEL_REGISTRY
from ..datasets import DATASET_REGISTRY
from ..utils.distribute import pytorch_worker_info

import copy
import inspect
import omegaconf
from pathlib import Path
from omegaconf import OmegaConf

"""
A wrapper to combine general epxeriments logic.

"""


def prepare_experiment(
    ModelInstance,
    DatasetInstance,
    env_config,
    exp_cfg,
    metrics,
    citerion,
    experiement_name="Test",
    optimizer=None,  # default value
    scheduler=None,
):
    """
    General function to create model, dataset, trainer and searcher. Answer to the question "how to create"
    is inside config objects, which are dictionaries in specific format, see "Configs description"
    in the framework documentation for more information.

    :param ModelInstance: class of model, which commonly represent the searchspace
    :type ModelInstance: python class variable
    :param DatasetInstance: class of dataset
    :type DatasetInstance: python class variable
    :param env_config: dictionary with environment configuration
    :type env_config: omegaconf.dictconfig.DictConfig
    :param exp_cfg: dictionary with experiment configuration
    :type exp_cfg: omegaconf.dictconfig.DictConfig
    :param metrics: list with metrics, commonly is returned by get_metrics_and_loss. Pay attention to that loss and metrics must take model output and target like dictionary, see examples for more information
    :type metrics: list of tuples (metric_name, metric_function, average_type)
    :param citerion: loss function. Pay attention to that loss and metrics must take model output and target like dictionary, see examples for more information
    :type citerion: function
    :param experiement_name: name of your experiment, defaults to "Test"
    :type experiement_name: str, optional
    :param optimizer: dictionary with information about optimizer to each param group, defaults to None
    :type optimizer: dict {param_group_name: (optimizer config)}, optional
    :param scheduler: dictionary with information about LR schedulers to each param group, defaults to None
    :type scheduler: dict {param_group_name: (scheduler config)}, optional

    :return: logger, model, trainer, loaders
    :rtype: tuple
    """

    new_conf = copy.deepcopy(env_config)
    new_conf.EXPERIMENT.DIR = str(Path(new_conf.EXPERIMENT.DIR) / experiement_name)
    dataset = get_dataset(DatasetInstance, exp_cfg)
    logger = get_logger(new_conf)
    OmegaConf.save(exp_cfg, Path(new_conf.EXPERIMENT.DIR) / "exp_cfg.yaml")

    # dataset creates only single main process
    is_main_process = False
    rank, world_size, _, _ = pytorch_worker_info()

    if rank == 0:
        is_main_process = True

    is_dataset_created = dataset.is_created()
    if world_size == 1 and not is_dataset_created:
        dataset.create_dataset()

    elif world_size > 1 and not is_dataset_created:
        logger.log(
            "Dataset is tried to be created in multi-gpu settings, "
            "create dataset before launching prepare_experiment",
            type=LoggerUnited.ERROR_MSG,
        )
        raise Exception

    dataset.load_dataset()

    if is_main_process:
        dataset_info = dataset.print_report()
        logger.log(dataset_info)

    model = get_model(ModelInstance, dataset, exp_cfg)
    trainer = get_trainer(
        exp_cfg,
        citerion,
        metrics,
        logger,
        optimizer,
        scheduler,
        device=new_conf.HARDWARE.GPU,
        class_weights=dataset.get_class_weights(),
    )

    loaders = get_loaders(dataset, exp_cfg, new_conf)

    return logger, model, trainer, loaders


def get_dataset(DatasetInstance, exp_cfg):
    """
    Function to create the dataset with specific configs. DatasetInstance must take exactly that parameters which take
    the TimeSeriesInMemoryDataset, see documentation and examples for more information.

    :param DatasetInstance: class of dataset or str - dataset_type
    :type DatasetInstance: python class variable
    :param exp_cfg: dictionary with configuration of dataset
    :type exp_cfg: omegaconf.dictconfig.DictConfig

    :return: dataset
    :rtype: DatasetInstance
    """
    if type(DatasetInstance) is str:
        if DatasetInstance not in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset doesn't exist or wasn't registered: {DatasetInstance}"
            )
        DatasetInstance = DATASET_REGISTRY[DatasetInstance]

    return DatasetInstance(
        data_path=exp_cfg.dataset.data_path,
        **exp_cfg.dataset.dataset_params,
    )


def get_model(ModelInstance, dataset, exp_cfg):
    """
    Function to create model with any configs, each parameter of model, whose name is in exp_cfg would initialize with exp_cfg[name] value.

    :param ModelInstance: class of model or model_name (str)
    :type ModelInstance: python class variable
    :param dataset: any object with "cardinalities" member, which represents dataset cardinalities
    :type dataset: instance with "cardinalities" member
    :param exp_cfg: dictionary with all needful configs to model
    :type exp_cfg: omegaconf.dictconfig.DictConfig

    :return: model, which commonly represents the searchspace
    :rtype: ModelInstance
    """
    if type(ModelInstance) is str:
        if ModelInstance not in MODEL_REGISTRY:
            raise ValueError(
                f"Model doesn't exist or wasn't registered: {ModelInstance}"
            )
        ModelInstance = MODEL_REGISTRY[ModelInstance]

    params_dict = get_params_dict(
        inspect.getargspec(ModelInstance.__init__).args, exp_cfg, {}
    )
    for _key in ["continious", "seq_len"]:
        if _key not in params_dict.keys():
            if hasattr(dataset, _key):
                params_dict[_key] = getattr(dataset, _key)
            else:
                print(f"Config and dataset has no attribute: {_key}")

    model = ModelInstance(
        **params_dict,
        cat_cardinalities=dataset.cardinalities,
    )  # list of tuples : (name , size)

    if hasattr(model, "run_replacement"):
        model.run_replacement()

    return model


def get_trainer(
    exp_cfg,
    criterion,
    metrics,
    logger,
    optimizer=None,
    scheduler=None,
    device="cpu",
    class_weights=None,
    **kwargs,
):
    """
    Function to get the trainer.

    :param exp_cfg: dictionary with all needful information about trainer
    :type exp_cfg: omegaconf.dictconfig.DictConfig
    :param criterion: loss function. Pay attention to that loss and metrics must take model output and target like dictionary, see examples for more information
    :type criterion: function
    :param metrics: list with metrics, commonly is returned by get_metrics_and_loss. Pay attention to that loss and metrics must take model output and target like dictionary, see examples for more information
    :type metrics: list of tuples (metric_name, metric_function, average_type)
    :param logger: logger to log a train process
    :type logger: nash_logging.common.LoggerUnited
    :param optimizer: dictionary with information about optimizer to each param group, defaults to None
    :type optimizer: dict {param_group_name: (optimizer config)}, optional
    :param scheduler: dictionary with information about LR scheduler to each param group, defaults to None
    :type scheduler: dict {param_group_name: (scheduler config)}, optional
    :param device: to what model and tensors would be delivered, defaults to "cpu"
    :type device: str or int, optional

    :return: trainer
    :rtype: trainers.trainer.Trainer
    """
    if optimizer is None:
        optimizer = {
            "main": (
                "ADAM",
                {
                    "lr": exp_cfg.trainer.optim_params.lr,
                    "weight_decay": exp_cfg.trainer.optim_params.weight_decay,
                },
            )
        }

    if exp_cfg.search_method.name in ["DiffSearcher"]:
        phases = ["train_main", "train_alpha", "validation"]
        optimizer["arch"] = optimizer["main"]
        dataset_opt_link = {"train_main": "main", "train_alpha": "arch"}
        log_arch = False
    elif exp_cfg.search_method.name in ["PTSearcher"]:
        phases = ["train", "validation"]
        dataset_opt_link = None
        log_arch = False
    elif exp_cfg.search_method.name in ["Hyperband", "RandomSearcher", "Bananas"]:
        phases = ["train", "validation"]
        dataset_opt_link = None
        log_arch = True
    else:
        raise Exception("Specify parameters for your search method in code above")

    iters_per_epoch = None
    if "iters_per_epoch" in exp_cfg.trainer:
        iters_per_epoch = exp_cfg.trainer.iters_per_epoch

    trainer = Trainer(
        criterion,
        metrics,
        scoring_metric=exp_cfg.trainer.scoring_metric,
        optimizers=optimizer,
        schedulers=scheduler,
        phases=phases,
        num_outputs=exp_cfg.model.output_size,
        num_epochs=exp_cfg.trainer.epochs,
        iters_per_epoch=iters_per_epoch,
        device=device,
        logger=logger,
        dataset_opt_link=dataset_opt_link,
        use_amp=exp_cfg.trainer.get("use_amp", False),
        class_weights=class_weights,
        log_arch=log_arch,
        **kwargs,
    )

    return trainer


def get_loaders(dataset, exp_cfg, env_cfg):
    """
    Function to get data loaders

    :param dataset: dataset instance
    :param exp_cfg: experiment config
    :param env_cfg: environment config
    :return: loaders
    """
    if exp_cfg.search_method.name in ["DiffSearcher"]:
        train_loader_one = dataset.get_data_loader(
            batch_size=exp_cfg.dataset.batch_size,
            workers=env_cfg.HARDWARE.WORKERS,
            train=True,
        )

        train_loader_two = dataset.get_data_loader(
            batch_size=exp_cfg.dataset.batch_size,
            workers=env_cfg.HARDWARE.WORKERS,
            train=True,
        )

        val_loader = dataset.get_data_loader(
            batch_size=exp_cfg.dataset.batch_size,
            workers=env_cfg.HARDWARE.WORKERS,
            train=False,
        )
        loaders = {
            "train_main": train_loader_one,
            "train_alpha": train_loader_two,
            "validation": val_loader,
        }
    else:
        train_loader = dataset.get_data_loader(
            batch_size=exp_cfg.dataset.batch_size,
            workers=env_cfg.HARDWARE.WORKERS,
            train=True,
        )
        val_loader = dataset.get_data_loader(
            batch_size=exp_cfg.dataset.batch_size,
            workers=env_cfg.HARDWARE.WORKERS,
            train=False,
        )
        loaders = {"train": train_loader, "validation": val_loader}
    return loaders


def get_logger(cfg):
    """
    Function to get logger

    :param cfg: configs to logger
    :type cfg: omegaconf.dictconfig.DictConfig

    :return: logger
    :rtype: nash_logging.common.UnitedLogger
    """
    return LoggerUnited(cfg, online_logger="tensorboard")


# Service function that SHOULDN`T BE CALLED manually
def get_params_dict(model_params, exp_cfg, d):
    for key in exp_cfg:
        if isinstance(exp_cfg[key], omegaconf.dictconfig.DictConfig):
            get_params_dict(model_params, exp_cfg[key], d)
        elif key in model_params:
            d[key] = exp_cfg[key]
            if key == "augmentations":
                d[key] = AUGM[exp_cfg[key]]
    return d


AUGM = {"Augmentation": Augmentation, "None": None}
