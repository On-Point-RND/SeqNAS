import numpy as np
import time
import shutil
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import multiprocessing as mp
from threading import BrokenBarrierError
from queue import Empty
from dict_hash import sha256
import traceback

from . import register_search_method
from .base_searcher import RandomSearcher
from ..utils.distribute import pytorch_worker_info
from ..nash_logging.common import LoggerUnited
from ..experiments_src import METRIC_REGISTRY


@dataclass
class Job:
    arch: dict
    ckpt: Path
    target_num_epochs: Optional[int] = None
    score: float = np.nan

    def __lt__(self, other):
        return self.score < other.score


@register_search_method("Hyperband")
class Hyperband(RandomSearcher):
    def __init__(
        self,
        model,
        trainer,
        dataloaders,
        max_total_epochs,
        max_epochs_per_arch=10,
        epochs_growth_factor=3,
        scoring_metric="accuracy",  # Metric to choose best model
        logger=None,
        max_search_space=1e6,  # After sampling this number of equal architecture search will be ended
        name="default",
        ckpt_path: Optional[Path] = None,
    ):
        """

        :param model: model which represents the searchspace
        :type model: omnimodel
        :param trainer: trainer to model
        :type trainer: trainers.trainer.Trainer
        :param dataloaders: dictionary of two loaders: validation and train
        :type dataloaders: dict
        :param max_total_epochs: max total epochs to spend on training during the search
        :type max_total_epochs: int
        :param max_epochs_per_arch: maximum number of train epochs to one architecture in hyperband, defaults to 1e6
        :type max_epochs_per_arch: int, optional
        :param epochs_growth_factor: the factor # of epochs grows and number of arches decreases during Successive Halvings
        :type epochs_growth_factor: int, optional
        :param scoring_metric: name of metric by which architecctures would be evaluated
        :type scoring_metric: string
        :param logger: logger to searcher
        :type logger: nash_logging.common.UnitedLogger
        :param max_search_space: service parameter which stop seaarch if nuber of equal architectures which was sampled will be more than it
        :type max_search_space: int
        :param name: hyperband name, need to create folder with temporal results, defaults to 'default'
        :type name: str, optional

        """
        super().__init__(
            model,
            trainer,
            dataloaders,
            scoring_metric,
            logger=logger,
            max_search_space=max_search_space,
        )

        rank, _, _, _ = pytorch_worker_info()

        spaces = 10
        type_module = "[SEARCHER]"
        process = f"PROCESS {rank}"
        self._log_prefix = f"{type_module:<{spaces}} - {process:<{spaces}}: "

        # self.forlog = []
        self.answers = []
        if ckpt_path is None:
            ckpt_path = Path(self.logger.checkpoint_folder) / "hyperband_tmp"

        self.path = ckpt_path or Path("_".join(("hyperband", name, str(time.time()))))
        if not self.path.exists():
            self.path.mkdir()

        assert logger is not None
        assert scoring_metric != "loss"

        self.model = model
        self.max_epochs_per_arch = max_epochs_per_arch
        self.max_total_epochs = max_total_epochs
        self.epochs_growth_factor = epochs_growth_factor

        self.max_sh_steps = (
            math.ceil(
                math.log(self.max_epochs_per_arch) / math.log(self.epochs_growth_factor)
            )
            + 1
        )
        # print(f"maximum SH steps: {max_sh_steps}")
        self.epochs_sched = self.epochs_growth_factor ** np.arange(self.max_sh_steps)
        self.epochs_sched[-1] = np.minimum(
            self.epochs_sched[-1], self.max_epochs_per_arch
        )
        # print(f"epochs schedule: {self.epochs_sched}")
        self.base_arches_to_retrain = (
            self.epochs_growth_factor ** np.arange(self.max_sh_steps)[::-1]
        )
        # print(f"base arches to retrain per step: {self.base_arches_to_retrain}")

        self.logger = logger
        self.dataloaders = dataloaders
        self.trainer = trainer
        self.trainer.set_dataloaders(self.dataloaders)
        self.scoring_metric = scoring_metric

        self.jobs_queue = mp.Queue()
        self.results_queue = mp.Queue()
        self.barrier = None

        self.logger.log(
            f"Total restrictions: {self.max_total_epochs} epochs",
            logger.INFO_MSG,
            only_main_rank=True,
        )

    def _sh_comp_budget(self, sh_steps: int) -> int:
        """Actual number of epochs for SH.

        Returns amount of resources (epochs) required to perform SH with `sh_steps`
        that ends up with one arch fully trained for `self.max_epochs_per_arch` epochs.

        :param hbo: Hyperband options
        :param sh_steps: number of SH steps
        :return: computational budget of the SH
        """

        epochs_to_retrain = np.diff(self.epochs_sched[-sh_steps:], prepend=0)
        # print(f"epochs to retrain per step: {epochs_to_retrain}")
        arches_to_retrain = self.base_arches_to_retrain[-sh_steps:]
        # print(f"arches to retrain per step: {arches_to_retrain}")
        resources_consumed = epochs_to_retrain * arches_to_retrain
        # print(f"resources consumed per step: {resources_consumed}")
        total_resources = resources_consumed.sum()
        # print(f"total resources consumed: {total_resources}")
        return total_resources

    def _hyperbandlet_schedule(
        self, n_halvings: int, remaining_epochs: int
    ) -> Tuple[np.ndarray, int]:
        """Creates SH schedule for one Hyperbandlet

        Hyperbandlet is a modification of Hyperband algorithm that has exactly `n_halvings` Successive
        Halvigs ending with a random search one. The function tries to make a balanced schedule of SHs i.e.
        such that all SHs have almost identical theoretical budget. Theoretical budget is a number of epochs
        required to perform a SH if no checkpointing is performed and the last arch is trained for the full
        number of epochs without clipping to fit the constraint. If the last remaining arch is trained for $n$
        epochs, then the theoretical budget of $-i$'th (python array indexing, SHs are sorted by the number of
        steps in descending order) SH equals $in$.

        To balance SHs the function multiplies the number of arches required for the first step of SH by some
        factor to match the theoretical budget of the largest SH (SH with the largest number of steps).
        Then the function calculates the computational budgets of all SHs and scales the SHs (numbers of arches
        for the first steps) to fit the `remaining_epochs` overall computational resources. If the largest SH
        has the coefficient less than 1, then such Hyperbandlet is assumed to be impossible and an array of zeros
        is returned. Else, the rounded towards zero coefficients are returned.

        :param hbo: Hyperband options
        :param n_halvings: number of SHs in Hyperbandlet
        :param remaining_epochs: remaining computational budget
        :return: SH scaling factors and consumed resources (epochs)
        """

        coeffs = n_halvings / np.arange(
            n_halvings, 0, -1
        )  # balance theoretical budgets
        # print(f"coefficients that balance theoretical budget: {coeffs}")
        comp_budgets = np.array(
            [self._sh_comp_budget(i) for i in range(n_halvings, 0, -1)]
        )
        # print(f"computations budgets: {comp_budgets}")
        coeffs *= remaining_epochs / (coeffs @ comp_budgets)
        # print(f"coefficients scaled to math the computational budget: {coeffs}")
        if coeffs[0] < 1:
            return np.zeros(n_halvings, dtype=int), 0
        coeffs = np.floor(coeffs).astype(int)
        # print(f"final coefficients: {coeffs}")
        epochs_consumed = int(coeffs @ comp_budgets)
        return coeffs, epochs_consumed

    def _hb_schedule(self) -> np.ndarray:
        """Create Hyperband schedule.

        Computes factors of Successive Halvings to closely match the computational budget by iterative
        addition of hyperbandlets starting from the one with the largest number of SHs in it.

        :param hbo: Hyperband options
        :return: Successive Halving factors
        """

        sh_coeffs = np.zeros(self.max_sh_steps, dtype=int)
        comp_budget_remaining = self.max_total_epochs
        for n_halvings in range(self.max_sh_steps, 0, -1):
            c, budget_consumed = self._hyperbandlet_schedule(
                n_halvings, comp_budget_remaining
            )
            sh_coeffs[-n_halvings:] += c
            comp_budget_remaining -= budget_consumed

        hb_report = "\n\n____xXx____HYPERBAND____xXx____SCHEDULE____xXx____"

        n = 1
        for i, factor in enumerate(sh_coeffs):
            if factor == 0:
                continue
            hb_report += f"\n\nSuccessive Halving {n}:"
            hb_report += "\n\tarches:\t" + "\t".join(
                str(i) for i in self.base_arches_to_retrain[i:] * factor
            )
            hb_report += "\n\tepochs:\t" + "\t".join(
                str(i) for i in self.epochs_sched[i:]
            )
            n += 1

        hb_report += (
            f"\n\nTotal epochs: {self.max_total_epochs - comp_budget_remaining}\n"
        )
        self.logger.log(
            self._log_prefix + hb_report,
            LoggerUnited.INFO_MSG,
            only_main_rank=True,
            filewrite=False,
        )

        return sh_coeffs

    def _successive_halving(self, n_steps: int, jobs: List[Job]):
        n = len(jobs)
        for step in range(n_steps, 0, -1):
            self.logger.log(
                self._log_prefix + f"filling the queue with {len(jobs)} jobs",
                LoggerUnited.DEBUG_MSG,
                only_main_rank=False,
                filewrite=False,
            )

            time.sleep(0.15)  # for synchronization

            for job in jobs:
                job.target_num_epochs = int(self.epochs_sched[-step])
                self.jobs_queue.put(job)

            self.train_r()
            jobs = [self.results_queue.get() for _ in range(n)]
            jobs = list(filter(lambda job: not math.isnan(job.score), jobs))
            n //= self.epochs_growth_factor
            jobs = self.top_n(jobs, n)

        ans = self.top_n(jobs, 1)[0]
        self.logger.log_metrics(
            tab="Test", metrics={self.scoring_metric: ans.score}, phase_idx=1
        )
        return ans

    def train_r(self, jobs: Optional[List[Job]] = None):
        """
        Train each architecture from the `jobs` list for target number of epochs
        """

        rank, _, _, _ = pytorch_worker_info()
        if rank > 0:
            assert self.barrier is not None

        if jobs is not None:
            jobs_it = iter(jobs)

        done = False
        while not done:
            # wait for queue to be filled
            if self.barrier is not None:
                try:
                    self.logger.log(
                        self._log_prefix + "on barrier",
                        LoggerUnited.DEBUG_MSG,
                        only_main_rank=False,
                        filewrite=False,
                    )
                    self.barrier.wait()
                except BrokenBarrierError:
                    return

            while True:
                if jobs is None:
                    try:
                        self.logger.log(
                            self._log_prefix + "trying to read from the queue",
                            LoggerUnited.DEBUG_MSG,
                            only_main_rank=False,
                            filewrite=False,
                        )
                        job = self.jobs_queue.get(block=True, timeout=0.1)
                    except Empty:
                        self.logger.log(
                            self._log_prefix
                            + "empty queue, waiting for a new workload",
                            LoggerUnited.DEBUG_MSG,
                            only_main_rank=False,
                            filewrite=False,
                        )
                        break

                else:
                    try:
                        job = next(jobs_it)
                    except StopIteration:
                        break

                try:
                    ckpt = None
                    if job.ckpt.exists():
                        ckpt = job.ckpt

                    self.logger.log(
                        self._log_prefix + f"training arch {sha256(job.arch)} "
                        f"for {job.target_num_epochs} epochs",
                        LoggerUnited.INFO_MSG,
                        only_main_rank=False,
                        filewrite=False,
                    )
                    score = self.train_one_arch(
                        job.arch,
                        n_epochs=job.target_num_epochs,
                        resume_from_ckpt=ckpt,
                        save_ckpt_to=job.ckpt,
                    )
                    self.logger.log(
                        self._log_prefix + f"finished training {sha256(job.arch)}",
                        LoggerUnited.INFO_MSG,
                        only_main_rank=False,
                        filewrite=False,
                    )
                    job.score = score

                except Exception as e:
                    self.logger.log(
                        self._log_prefix + f"Failed to train architecture "
                        f"{sha256(job.arch)}, error: {e}. Skipping",
                        LoggerUnited.ERROR_MSG,
                        only_main_rank=False,
                    )
                    traceback.print_exc()

                finally:
                    try:
                        self.trainer.free_model_mem()
                    except Exception as e:
                        type_module = "[SEARCHER]"
                        process = f"PROCESS {rank}"
                        spaces = 10
                        self.logger.log(
                            f"{type_module:<{spaces}} - {process:<{spaces}}: Failed to free memory "
                            f"{sha256(job.arch)}, error: {e}",
                            LoggerUnited.ERROR_MSG,
                            only_main_rank=False,
                        )
                    self.trainer.reset_best_score()

                if jobs is None:
                    self.results_queue.put(job)

            # run loop once if running in a main process to generate new jobs
            done = rank == 0

    def top_n(self, jobs: List[Job], n):
        jobs = [job for job in jobs if not math.isnan(job.score)]
        jobs = sorted(jobs)
        if not METRIC_REGISTRY[self.scoring_metric]["metric_class"].higher_is_better:
            return jobs[:n]
        return jobs[-n:]

    def get_jobs(self, n):
        """
        Generate n random architecture to search
        """
        jobs = []
        for _ in range(n):
            current_arch, arch_hash, end = self.get_arch()
            if end:
                break

            self.computed_arches[arch_hash] = 1
            path = self.path / str(arch_hash)
            jobs.append(Job(current_arch, path))
        return jobs

    def restricted_R(self):
        """
        Main function of algorythm, realize all search and return best architecture
        """
        hb_sched = self._hb_schedule()
        for i, sh_factor in enumerate(hb_sched):
            if sh_factor == 0:
                continue
            jobs = self.get_jobs(sh_factor * self.base_arches_to_retrain[i])
            if len(jobs) == 0:
                break
            ans = self._successive_halving(self.max_sh_steps - i, jobs)
            self.logger.log(
                f"{i} Successive Halving has been finished. Score: {ans.score}"
            )
            self.answers.append(ans)

        return self.top_n(self.answers, 1)[0]

    def search(self, jobs_queue=None, results_queue=None, barrier=None, arches=None):
        """
        Start search.
        """
        assert arches is None, "Parameter arches isn't used in Hyperband"
        start = time.time()
        self.answers = []

        if jobs_queue is not None:
            self.jobs_queue = jobs_queue
        if results_queue is not None:
            self.results_queue = results_queue
        self.barrier = barrier

        rank, _, _, _ = pytorch_worker_info()

        if rank > 0:
            # subprocesses just wait for the jobs
            self.train_r()
            return

        # main process fills the queue and process the results
        ans = self.restricted_R()

        if self.barrier is not None:
            self.barrier.abort()

        self.logger.log(
            "Real time arrival: " + str((time.time() - start)) + " seconds",
            self.logger.INFO_MSG,
        )
        self.logger.log("Best score: " + str(ans.score), self.logger.INFO_MSG)
        # self.logger.shutdown_logging()
        self.ckpt = torch.load(ans.ckpt)
        self.weights = self.ckpt["model_state"]
        self.arch = ans.arch
        shutil.rmtree(self.path)
        return ans
