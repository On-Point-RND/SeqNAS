import os
import atexit
import functools
import logging
import subprocess
import sys

from iopath.common.file_io import g_pathmgr
from .io_utils import makedir
from ..utils.distribute import pytorch_worker_info
import torch


class Logger:
    """
    Base logger class
    """

    INFO_MSG = 1
    DEBUG_MSG = 2
    WARNING_MSG = 3
    ERROR_MSG = 4

    def __init__(self, name: str, output_dir: str = None):
        """
        Init Logger

        :param name: logger name
        :param output_dir: experiment output dir
        """
        rank, world_size, _, _ = pytorch_worker_info()
        if world_size > 1:
            self.use_ddp = int(os.environ.get("USE_DDP", "1"))
            if not self.use_ddp:
                self.multi_gpu = True
        else:
            self.use_ddp = False
            self.multi_gpu = False
        self.setup_logging(name, output_dir)

    def init_two_loggers(self, name: str, log_filename: str):
        """
        Function initializes 2 loggers: for writing to std and to log file

        :param name: logger name
        :param log_filename: logger file name
        """
        logger_stdout = logging.getLogger(f"{name}_stdout")
        logger_stdout.setLevel(logging.DEBUG)

        # create formatter
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
        formatter = logging.Formatter(FORMAT)

        # setup the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger_stdout.addHandler(console_handler)
        logger_stdout.propagate = False

        # we log to file as well if user wants
        logger_filewriter = None
        if log_filename:
            logger_filewriter = logging.getLogger(f"{name}_filewriter")
            logger_filewriter.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger_filewriter.addHandler(file_handler)
            logger_filewriter.propagate = False
        return logger_stdout, logger_filewriter

    def setup_logging(self, name, output_dir=None):
        """
        Setup various logging streams: stdout and file handlers.
        For file handlers, we only setup for the master gpu.
        """
        # get the filename if we want to log to the file as well
        log_filename = None
        rank, world_size, _, _ = pytorch_worker_info()
        if output_dir and rank == 0:
            makedir(output_dir)
        if self.use_ddp:
            if rank == 0:
                log_filename = f"{output_dir}/log_{rank}.txt"
        else:
            log_filename = f"{output_dir}/log_{rank}.txt"

        self.output_dir = output_dir
        logger_stdout = None
        logger_filewriter = None
        if self.use_ddp:
            if rank == 0:
                logger_stdout, logger_filewriter = self.init_two_loggers(
                    name, log_filename
                )
        else:
            logger_stdout, logger_filewriter = self.init_two_loggers(name, log_filename)

        self.logger_stdout = logger_stdout
        self.logger_filewriter = logger_filewriter

    def log(
        self,
        message: str,
        type: int = None,
        only_main_rank: bool = False,
        stdout: bool = True,
        filewrite: bool = True,
    ):
        """
        Logs message to stdout/file

        :param message: msg to log
        :param type: INFO/DEBUG/WARNING/ERROR
        :param only_main_rank: log process with only rank=0
        :param stdout: True if write to stdout
        :param filewrite: True if write to file
        """
        rank, world_size, _, _ = pytorch_worker_info()
        if only_main_rank and rank != 0:
            return
        if self.use_ddp and rank != 0:
            return

        if type == Logger.INFO_MSG or type is None:
            if stdout:
                self.logger_stdout.info(message)
            if filewrite:
                self.logger_filewriter.info(message)
        elif type == Logger.DEBUG_MSG:
            if stdout:
                self.logger_stdout.debug(message)
            if filewrite:
                self.logger_filewriter.debug(message)
        elif type == Logger.WARNING_MSG:
            if stdout:
                self.logger_stdout.warning(message)
            if filewrite:
                self.logger_filewriter.warning(message)
        elif type == Logger.ERROR_MSG:
            if stdout:
                self.logger_stdout.error(message)
            if filewrite:
                self.logger_filewriter.error(message)

    # cache the opened file object, so that different calls to `setup_logger`
    # with the same file name can safely write to the same file.
    @functools.lru_cache(maxsize=None)
    def _cached_log_stream(self, filename):
        # we tune the buffering value so that the logs are updated
        # frequently.
        log_buffer_kb = 10 * 1024  # 10KB
        io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
        atexit.register(io.close)
        return io

    def shutdown_logging(self):
        """
        After training is done, we ensure to shut down all the logger streams.
        """
        logging.info("Shutting down loggers...")
        handlers = logging.root.handlers
        for handler in handlers:
            handler.close()

    def log_gpu_stats(self):
        """
        Log nvidia-smi snapshot. Useful to capture the configuration of gpus.
        """
        try:
            self.log(
                subprocess.check_output(["nvidia-smi"]).decode("utf-8"),
                Logger.INFO_MSG,
                only_main_rank=True,
            )
        except FileNotFoundError:
            self.log(
                "Failed to find the 'nvidia-smi' executable for printing GPU stats",
                Logger.ERROR_MSG,
                only_main_rank=True,
            )
        except subprocess.CalledProcessError as e:
            self.log(
                f"nvidia-smi returned non zero error code: {e.returncode}",
                Logger.ERROR_MSG,
                only_main_rank=True,
            )

    def print_gpu_memory_usage(self):
        """
        Parse the nvidia-smi output and extract the memory used stats.
        Not recommended to use.

        """
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        all_values, count, out_dict = [], 0, {}
        for item in out_list:
            if " MiB" in item:
                out_dict[f"GPU {count}"] = item.strip()
                all_values.append(int(item.split(" ")[0]))
                count += 1
        self.log(
            f"Memory usage stats:\n"
            f"Per GPU mem used: {out_dict}\n"
            f"nMax memory used: {max(all_values)}",
            Logger.INFO_MSG,
            only_main_rank=True,
        )

    def _prepare_tree_for_saving(self, name: str, subdir: str = None):
        """
        Prepare full path for saving files

        :param name: file name
        :param subdir: subdirectory
        """
        rank, _, _, _ = pytorch_worker_info()
        if self.use_ddp and rank != 0:
            return None

        if subdir is not None:
            out_folder = os.path.join(self.output_dir, subdir)
            out_file = os.path.join(out_folder, name)
            os.makedirs(out_folder, exist_ok=True)
        else:
            out_file = os.path.join(self.output_dir, name)

        return out_file

    def save_custom_txt(
        self, content: str = "", name: str = "name.txt", subdir: str = None
    ):
        """
        Allow to save custom txt file

        :param content: msg
        :param name: file name
        :param subdir: subdirectory
        """
        out_file = self._prepare_tree_for_saving(name, subdir)
        if out_file is None:
            return
        with open(out_file, "w") as f:
            f.write(content)

    def save_custom_torch(self, content: str, name: str, subdir: str = None):
        """
        Allow to save custom torch file

        :param content: msg
        :param name: file name
        :param subdir: subdirectory
        """

        out_file = self._prepare_tree_for_saving(name, subdir)
        if out_file is None:
            return
        torch.save(content, out_file)
