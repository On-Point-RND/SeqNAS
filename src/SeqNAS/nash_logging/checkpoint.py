import os
from glob import glob
import shutil
from pathlib import Path

from iopath.common.file_io import g_pathmgr
from .logger import Logger
from .io_utils import makedir
from ..utils.distribute import pytorch_worker_info


def get_checkpoint_folder(config):
    """
    Check, create and return the checkpoint folder. User can specify their own
    checkpoint directory otherwise the default "." is used.
    """
    odir = config.EXPERIMENT.DIR

    makedir(odir)
    assert g_pathmgr.exists(
        config.EXPERIMENT.DIR
    ), f"Please specify config.CHECKPOINT.DIR parameter. Invalid: {config.CHECKPOINT.DIR}"
    return odir


def save_source_files(config, logger):
    rank, _, _, _ = pytorch_worker_info()
    checkpoint_folder = get_checkpoint_folder(config)
    logger.log(
        f"Saving source files to {checkpoint_folder}/project_source/",
        Logger.INFO_MSG,
        only_main_rank=True,
    )
    assert g_pathmgr.exists(
        config.PROJECT.ROOT
    ), f"Please specify config.PROJECT.ROOT parameter. Invalid: {config.PROJECT.ROOT}"
    dirs_to_walk = [
        os.path.join(config.PROJECT.ROOT, x) for x in config.PROJECT.DEFAULT_DIRS
    ]
    py_files = glob(os.path.join(config.PROJECT.ROOT, "*.py"))
    for dw in dirs_to_walk:
        py_files.extend(
            [y for x in os.walk(dw) for y in glob(os.path.join(x[0], "*.py"))]
        )
    for pyf in py_files:
        src = pyf
        dst = os.path.join(
            f"{checkpoint_folder}/project_source/",
            f'{pyf.replace(config.PROJECT.ROOT, "")}',
        )
        dst_dir = Path(dst).parents[0]
        if rank == 0:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
