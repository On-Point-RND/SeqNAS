import pprint

from omegaconf import DictConfig, OmegaConf

from .logger import Logger
from .checkpoint import get_checkpoint_folder
from ..utils.distribute import pytorch_worker_info


def print_cfg(cfg, logger):
    """
    Supports printing both DictConfig and also the AttrDict config

    """
    logger.log("Training with config:", Logger.INFO_MSG, only_main_rank=True)
    if isinstance(cfg, DictConfig):
        if hasattr(cfg, "pretty"):
            # Backward compatibility
            logger.log(cfg.pretty(), Logger.INFO_MSG, only_main_rank=True)
        else:
            # Newest version of OmegaConf
            logger.log(OmegaConf.to_yaml(cfg), Logger.INFO_MSG, only_main_rank=True)
    else:
        logger.log(pprint.pformat(cfg), Logger.INFO_MSG, only_main_rank=True)


def save_cfg(cfg, logger):
    """
    Saves cfg to experiment folder

    :param cfg: Environment config.yaml file
    :param logger: logger
    """
    rank, _, _, _ = pytorch_worker_info()
    checkpoint_folder = get_checkpoint_folder(cfg)
    logger.log(
        f"Saving config to {checkpoint_folder}/config.yaml:",
        Logger.INFO_MSG,
        only_main_rank=True,
    )
    if isinstance(cfg, DictConfig) and rank == 0:
        OmegaConf.save(config=cfg, f=f"{checkpoint_folder}/config.yaml")
    else:
        logger.log(
            f"Config is not of DictConfig type", Logger.ERROR_MSG, only_main_rank=True
        )
