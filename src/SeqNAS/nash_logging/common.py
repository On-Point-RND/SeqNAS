import torch
import shutil
from pathlib import Path
from contextlib import contextmanager

from .logger import Logger
from .config import print_cfg, save_cfg
from .checkpoint import save_source_files
from .checkpoint import get_checkpoint_folder
from .tensorboard_utils import get_tensorboard_logger
from ..utils.distribute import pytorch_worker_info

online_loggers = {"tensorboard": get_tensorboard_logger}


class LoggerUnited(Logger):
    """
    Main Logger that used in library.
    Can work in multiprocessing mode.
    """

    def __init__(self, config, online_logger=None):
        """
        Function creates experiment directory and save main config files to it

        :param config: Environment config.yaml file
        :param online_logger: tensorboard logger
        """
        self.checkpoint_folder = get_checkpoint_folder(config)
        super().__init__(__name__, output_dir=self.checkpoint_folder)

        if online_logger is not None:
            self.online_logger = online_loggers[online_logger](config)
        self.use_online = False if online_logger is None else True
        self._runs_stack = []

        self.log_gpu_stats()
        self.print_gpu_memory_usage()
        print_cfg(config, self)
        save_cfg(config, self)
        save_source_files(config, self)

    def get_exp_root_dir(self):
        return Path(self.checkpoint_folder)

    def set_run(self, run_name=None):
        if self.use_online:
            self.online_logger.set_run(run_name)

    @contextmanager
    def toggle_run(self, run_name=None):
        if self.use_online:
            with self.online_logger.toggle_run(run_name):
                yield
        else:
            yield

    def on_update(
        self,
        iteration,
        loss,
        log_frequency,
        batch_time,
        max_iteration,
    ):
        """
        Use to log training process to online logger.

        :param iteration: iteration from which log started
        :type iteration: int
        :param loss: loss to log
        :type loss: float
        :param log_frequency: frequency (in epochs) to log (can be None)
        :type log_frequency: int
        :param batch_time: time execution for batches (can be None)
        :type batch_time: list of floats
        :param max_iteration: Max iteration which must be logged (can be None)
        :type max_iteration: int
        """

        if self.use_online:
            self.online_logger.on_update(
                iteration=iteration,
                loss=loss,
                log_frequency=log_frequency,
                batch_time=batch_time,
                max_iteration=max_iteration,
            )

    def log_metrics(self, tab="Train", metrics={}, phase_idx=0):
        """
        Log metrics to online logger.

        :param tab: table name
        :type tab: string
        :param metrics: metrics to log in format {name: values}
        :type metrics: dict
        :param phase_idx: number of step
        :type phase_idx: int
        """

        if self.use_online:
            rank, _, _, _ = pytorch_worker_info()
            if self.use_ddp and rank != 0:
                return
            self.online_logger.log_metrics(tab, metrics, phase_idx, rank)

    def log_scalars(self, tab="Train", scalars={}, phase_idx=0):
        """
        Log scalars to online logger.

        :param tab: table name
        :type tab: string
        :param scalars: scalars to log in format {name: values}
        :type scalars: dict
        :param phase_idx: number of step
        :type phase_idx: int
        """

        if self.use_online:
            self.online_logger.log_scalars(tab, scalars, phase_idx)

    def save_checkpoint(self, state, is_best, filename=None, only_main_rank=True):
        """
        Save state of model with label (best or no).

        :param state: model.state_dict
        :type state: dict
        :param is_best: is model best
        :type is_best: bool
        :param filename: filename to save checkpoint
        :type filename: str
        """
        rank, _, _, _ = pytorch_worker_info()
        if only_main_rank and rank != 0:
            return

        if filename is None:
            filename = Path(self.checkpoint_folder) / "checkpoint.pth.tar"

        try:
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, filename.parent / "model_best.pth.tar")
        except Exception as e:
            print(f"Unable to save checkpoint to '{filename}' at this time due to {e}")

    # Wrappers on tensorboard functions
    def add_histogramm(self, values=None, phase_idx=0, name="histogram"):
        if self.use_online:
            self.online_logger.add_histogramm(
                values=values, phase_idx=phase_idx, name=name
            )

    def add_embedding(self, emb, tag, phase_idx):
        if self.use_online:
            self.online_logger.add_embedding(emb, tag, phase_idx)

    def add_graph(self, model, input):
        if self.use_online:
            self.online_logger.add_graph(model, input)

    def add_images(self, tag, image):
        if self.use_online:
            self.online_logger.add_images(tag, image)
