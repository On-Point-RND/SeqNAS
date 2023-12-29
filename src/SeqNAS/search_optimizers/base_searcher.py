from ..utils.distribute import pytorch_worker_info
from ..nash_logging.common import LoggerUnited
from ..experiments_src import METRIC_REGISTRY
from . import register_search_method

import os
import re
import json
import copy
import pickle
import traceback
import numpy as np
from queue import Queue
from queue import Empty
from dict_hash import sha256
from tqdm.autonotebook import tqdm


@register_search_method("RandomSearcher")
class RandomSearcher:
    def __init__(
        self,
        model,
        trainer,
        dataloaders,
        scoring_metric="loss",  # Metric to choose best model
        arches_count=3,
        logger=None,
        max_search_space=1e6,  # After sampling this number of equal architecture search will be ended
    ):
        """
        Parent instance for most searchers in library,
        have many useful functions but haven`t search() method, so can`t be used as independent searcher.

        :param model: model which represents the searchspace
        :type model: omnimodel
        :param trainer: trainer to model
        :type trainer: trainers.trainer.Trainer
        :param dataloaders: dictionary of two loaders: validation and train
        :type dataloaders: dict
        :param scoring_metric: name of metric by which architecctures would be evaluated
        :type scoring_metric: string
        :param arches_count: number of architectures to search
        :type arches_count: int
        :param logger: logger to searcher
        :type logger: nash_logging.common.UnitedLogger
        :param max_search_space: service parameter which stop seaarch if nuber of equal architectures which was sampled will be more than it
        :type max_search_space: int

        """

        self.model = model
        self.arches_count = arches_count
        self.logging = False if logger is None else True
        self.logger = logger
        self.space = None
        self.arch = None
        self.weights = None
        self.max_search_space = max_search_space

        self.trainer = trainer
        self.trainer.set_dataloaders(dataloaders)
        self.scoring_metric = scoring_metric
        self.computed_arches = dict()
        self.computed_arches_scores = dict()
        self.best_arch = None

        if not METRIC_REGISTRY[scoring_metric]["metric_class"].higher_is_better:
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

        self._distill = False
        self._distill_w = 0.0
        self._distill_n_skip = 0
        self._distill_n_teachers = 0

    def enable_distillation(self, weight=0.2, n_skip=30, n_teachers=3):
        """
        Enable knowledge distillation during search.

        :param weight: distillation loss weight (will be truncated to [0, 1] range)
        :type weight: float
        :param n_skip: start distillation when at least `n_skip` predictions will be obtained
        :type n_skip: int
        :param n_teachers: number of teachers to distill with
        :type n_teachers: int
        """
        assert n_skip > 0
        assert n_teachers <= n_skip
        self.trainer.save_train_preds = True
        self._distill = True
        self._distill_w = weight
        self._distill_n_skip = n_skip
        self._distill_n_teachers = n_teachers

        rank, world_size, _, _ = pytorch_worker_info()
        type_module = "[SEARCHER]"
        process = f"PROCESS {rank}"
        spaces = 10

        if self.logging and rank == 0:
            self.logger.log(
                message=f"{type_module:<{spaces}} - {process:<{spaces}}: knowledge distillation on"
            )

    def get_arch(self):
        """
        Sample random architecture from self.model.
        """
        end = False
        repeated = 0
        if self.space is None:
            self.model.sample_random()
            current_arch = self.model.get_arch()
            arch_hash = self._hash_arch(current_arch)

            while arch_hash in self.computed_arches:
                repeated += 1
                if repeated > self.max_search_space:
                    self.logger.log("No more unique arches - quitting")
                    end = True
                    break
                self.model.sample_random()
                current_arch = self.model.get_arch()
                arch_hash = self._hash_arch(current_arch)
            return current_arch, arch_hash, end

        else:
            arch_hash = np.random.choice(self.hashs)
            while arch_hash in self.computed_arches:
                repeated += 1
                if repeated > self.max_search_space:
                    self.logger.log("No more unique arches - quitting")
                    end = True
                    break
                arch_hash = np.random.choice(self.hashs)
            arch = self.space[arch_hash]
            if "arch" in arch:
                arch = arch["arch"]
            return arch, arch_hash, end

    def set_space(self, space):
        """
        Set searchspace from which will be sampled random architectures

        :param space: architectures to random sample
        :type space: list of architectures
        """
        self.space = space
        self.hashs = list(space.keys())

    def set_computed_arches(self, arches_path):
        self.computed_arches = dict()
        self.computed_arches_scores = dict()

        arches_files = os.listdir(arches_path)
        for arch_file_name in arches_files:
            arch_path_abs = os.path.join(arches_path, arch_file_name)
            with open(arch_path_abs, "r") as f:
                arch_data = json.load(f)
            arch = arch_data["arch"]
            score = arch_data["scoring_metric"]
            self.computed_arches[self._hash_arch(arch)] = arch
            self.computed_arches_scores[self._hash_arch(arch)] = score

    def log_arch(self, score, arch):
        if self.logging:
            name = f"score_{score:.4f}_{len(self.computed_arches)}.txt"
            self.logger.save_custom_txt(content=str(arch), name=name, subdir="arches")

    def log_model_scores(self, best=False):
        if best:
            tab = "Validation SearchBest"
        else:
            tab = "Validation Search"
        if self.logging:
            metrics = self.trainer.get_best_epoch_metrics(self.scoring_metric)
            self.logger.log_metrics(
                tab=tab,
                metrics=metrics,
                phase_idx=len(self.computed_arches),
            )

    def set_dataloaders(self):
        self.trainer.set_dataloaders()

    # Convert found architechture into a unique key
    def _hash_arch(self, arch):
        if isinstance(arch, dict):
            return sha256(arch)
        else:
            raise Exception("Arch structure is not a dictionary")

    def track_best_model(self, score, arch):
        sign = 1
        if not METRIC_REGISTRY[self.scoring_metric]["metric_class"].higher_is_better:
            sign = -1

        if sign * score > sign * self.best_score:
            self.best_score = score
            self.best_arch = arch
            self.log_model_scores(best=True)

        rank, world_size, _, _ = pytorch_worker_info()
        type_module = "[SEARCHER]"
        process = f"PROCESS {rank}"
        spaces = 10

        if self.logging:
            self.logger.log(
                message=f"{type_module:<{spaces}} - {process:<{spaces}}: TRAINED MODEL SCORE {score:.9f} - BEST SCORE: {self.best_score:.9f} "
                f"- METRIC: {self.scoring_metric} - COMPUTED: {len(list(self.computed_arches.keys()))} ARCHES"
            )

    def _setup_distill(self):
        pred_dir = self.logger.get_exp_root_dir() / "predictions"
        preds = list(pred_dir.glob("*"))
        if len(preds) < self._distill_n_skip:
            return

        prog = re.compile("^score_(-?\d*\.?\d*)_")
        preds = [it.name for it in preds]
        preds.sort(key=lambda name: float(prog.findall(name)[0]))
        self.trainer.setup_distillation(
            True, preds[-self._distill_n_teachers :], self._distill_w
        )

    def train_one_arch(
        self,
        arch,
        *,
        n_epochs=None,
        save_ckpt_to=None,
        resume_from_ckpt=None,
        train_run_name=None,
    ):
        if self._distill:
            self._setup_distill()

        self.model.set_arch(arch)
        self.model.reset_weights()
        self.trainer.set_model(copy.deepcopy(self.model))

        saved_num_epochs = self.trainer.num_epochs
        if n_epochs is not None:
            self.trainer.num_epochs = n_epochs
        if resume_from_ckpt is not None:
            self.trainer.load_checkpoint(resume_from_ckpt)

        train_run_name = train_run_name or str(self._hash_arch(arch))
        with self.logger.toggle_run(train_run_name):
            self.trainer.train()

        if save_ckpt_to is not None:
            self.trainer.save_checkpoint(save_ckpt_to, only_main_rank=False)

        self.trainer.num_epochs = saved_num_epochs

        current_score = self.trainer.get_best_epoch_metrics(self.scoring_metric)[
            self.scoring_metric
        ]
        self.computed_arches_scores[self._hash_arch(arch)] = current_score
        self.computed_arches[self._hash_arch(arch)] = arch

        self.track_best_model(current_score, arch)
        self.log_model_scores()

        return current_score

    def sample_n_arches(self, N):
        new_sampled = dict()
        with tqdm(total=N) as pbar:
            while len(new_sampled) < N:
                arch, arch_hash, end = self.get_arch()
                if end:
                    break
                if arch_hash not in new_sampled:
                    new_sampled[arch_hash] = arch
                pbar.update(1)

        return new_sampled

    def get_model(self):
        """
        Return best model with trained weights.
        """
        if self.arch is not None:
            self.model.set_arch(self.arch)
            print("Architecture was set successfully")
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
            print("Weights was set successfully")
        self.model.set_final_and_clean()
        return self.model

    def search(self, jobs_queue=None, results_queue=None, barrier=None, arches=None):
        """
        Start search.
        :param barrier: barrier for multigpu and DDP
        :param jobs_queue: use multiprocessing queue for multi_gpu and DDP and simple queue for single-gpu mode
        :param arches: arches that will be trained
        :return:
        """
        # TODO add support of results_queue
        assert (
            results_queue is None
        ), "Currently results_queue isn't supported; in todo list"

        failed_arches = []
        rank, world_size, _, _ = pytorch_worker_info()
        type_module = "[SEARCHER]"
        process = f"PROCESS {rank}"
        spaces = 10

        if rank == 0:
            # process with rank 0 is main. it sends arches to the other processes
            arches_count = self.arches_count if arches is None else len(arches)
            self.logger.log(
                f"{type_module:<{spaces}} - {process:<{spaces}}: Generating {arches_count} arches",
                LoggerUnited.INFO_MSG,
                only_main_rank=True,
            )
            if arches is None:
                arches = self.sample_n_arches(arches_count)
            for arch in arches.values():
                jobs_queue.put(pickle.dumps(arch))

        # waits arches will be generated
        if world_size > 1:
            assert barrier is not None, "Pass barrier to pipeline func"
            barrier.wait()

        if world_size > 1:
            total_arches = jobs_queue.qsize()
            barrier.wait()
        else:
            total_arches = jobs_queue.qsize()

        while jobs_queue.qsize() != 0:
            try:
                current_arch = jobs_queue.get(timeout=60)
            except Empty:
                break
            current_arch = pickle.loads(current_arch)
            try:
                self.train_one_arch(current_arch)
                self.logger.log(
                    f"{type_module:<{spaces}} - {process:<{spaces}}: LEFT {jobs_queue.qsize()} "
                    f"out of {total_arches}; computed: {total_arches - jobs_queue.qsize()}",
                    LoggerUnited.INFO_MSG,
                    only_main_rank=False,
                    filewrite=False,
                )
            except Exception as e:
                self.logger.log(
                    f"{type_module:<{spaces}} - {process:<{spaces}}: Failed to train architecture "
                    f"{sha256(current_arch)}, error: {e}. Skipping",
                    LoggerUnited.ERROR_MSG,
                    only_main_rank=False,
                )
                failed_arches.append(current_arch)
                traceback.print_exc()
                self.logger.save_custom_txt(
                    json.dumps(failed_arches), f"failed_arches_{rank}.json"
                )
            finally:
                try:
                    self.trainer.free_model_mem()
                except Exception as e:
                    self.logger.log(
                        f"{type_module:<{spaces}} - {process:<{spaces}}: Failed to free memory "
                        f"{sha256(current_arch)}, error: {e}",
                        LoggerUnited.ERROR_MSG,
                        only_main_rank=False,
                    )
                self.trainer.reset_best_score()

        return self.best_arch
