import os

from .. import register_search_method
from ..base_searcher import RandomSearcher
from ...utils.distribute import pytorch_worker_info
from ...nash_logging.common import LoggerUnited
from ...experiments_src import METRIC_REGISTRY

import copy
import torch
import random
import shutil
import numpy as np
from queue import Queue
from omegaconf import OmegaConf

from .mutations import MUTATIONS
from .acquisition_func import (
    independent_thompson_sampling,
    expected_improvement,
    greedy_sampling,
)
from .ensemble import CatBoost_Ensemble

acquisition_functions = {
    "ITS": independent_thompson_sampling,
    "EI": expected_improvement,
    "GR": greedy_sampling,
}

bananas_default_config = {
    "initial_step": 100,
    "candidates_to_seed": 100,
    "candidates_per_step": 10,
    "predictor_objective": "MAE",
    "acquisition_function": "ITS",
    "predictor_ensembles": 5,
    "predictor_iters": 10,
    "predictor_lr": 0.01,
    "candidate_generation": {
        "type": "random",
    },
}


@register_search_method("Bananas")
class Bananas(RandomSearcher):
    def __init__(
        self,
        model,
        trainer,
        dataloaders,
        scoring_metric="loss",
        num_trials=3,
        logger=None,
        bananas_config=None,
        computed_arches_dir=None,
        max_search_space=1e4,
    ):
        """
        Bananas searcher. This searcher performs the following search strategy:

        1. Train a number of randomly sampled models - use initial_step in search method to specify a number of initial architectures;
            2. Convert architectures into features using get_feature_vector;
            3. Train a meta model (Catboost is used by default) to predict architecture scores;
            4. Use meta model scores to score new random article cultures - "candidates_to_seed" is the number of randomly sampled architectures to score and candidates_per_step is the number of selected architectures;
            5. To select architectures to train and to maintain good exploration VS exploitation ratio, we can use uncertainty estimation, also known as acquisition_function,  there are two options for it at the moment Independent Thompson Sampling and Expected Improvement;
            6. Then we train new selected architectures and use their scores to update our Meta model.
        7. Steps 2 - 6 are repeated for "num_trials".


        :param model: An omni model class which supports random sampling
        :param trainer: Model trainer class
        :param dataloaders: Dataloaders for training and validation
        :param scoring_metric: A metric to select best models
        :param num_trials=3: Define the number of times we want to perform a search cycle
        :param logger=None: Pass a logger object to the class
        :param bananas_config=None: Pass a dictionary of hyperparameters to the class
        :param max_search_space=1e4: Limit the number of architectures that are evaluated
        """
        super().__init__(
            model,
            trainer,
            dataloaders,
            scoring_metric,
            logger=logger,
            max_search_space=max_search_space,
        )

        self.num_trials = num_trials
        if bananas_config is None:
            self.cfg = bananas_default_config
        else:
            if isinstance(bananas_config, str):
                self.cfg = OmegaConf.load(bananas_config)
            else:
                self.cfg = bananas_config

        self.predictor = CatBoost_Ensemble(
            self.cfg["predictor_ensembles"],
            self.cfg["predictor_iters"],
            predictor_objective=self.cfg["predictor_objective"],
            predictor_lr=self.cfg["predictor_lr"],
        )

        self.acquisition_func = acquisition_functions[self.cfg["acquisition_function"]]

        self.benchmarks = []
        self.arches_dir = os.path.join(self.logger.checkpoint_folder, "arches")
        self.n_initial_steps = self.cfg["initial_step"]

        assert hasattr(model.__class__, "get_feature_vector") and callable(
            getattr(model.__class__, "get_feature_vector")
        ), self.logger.log(
            "Model has no get_feature_vector method",
            LoggerUnited.ERROR_MSG,
            only_main_rank=True,
        )

        self.computed_arches_dir = computed_arches_dir
        if computed_arches_dir is not None:
            self.n_initial_steps = 0

    def _get_evaluated_archs(self):
        """
        Store archs and corresponding losses for the set of evaluated architectures.

        Parameters
        ----------
        archs:
            Architectures
        losses:
            Losses obtained as a result of evaluation
        """

        evaluated_set = {"features": [], "scores": []}

        for arch_hash in self.computed_arches.keys():
            feature_vector = self.model.get_feature_vector(
                self.computed_arches[arch_hash]
            )
            score = self.computed_arches_scores[arch_hash]
            evaluated_set["features"].append(feature_vector)
            evaluated_set["scores"].append(score)

        evaluated_set["features"] = torch.tensor(evaluated_set["features"])
        evaluated_set["scores"] = torch.tensor(evaluated_set["scores"])
        return evaluated_set

    def _initial_step(self, jobs_queue, barrier=None, N=None, seed=None, arches=None):
        """
        Train uniformly chosen random architectures and store results in the object.

        Parameters
        ----------
        N:
            Number of architectures.
        seed:
            Random seed for generation process.
        recipes:
            Here is an option to pass exact list of recipes that should be evaluated
        """
        rank, world_size, _, _ = pytorch_worker_info()
        assert N is not None or arches is not None, self.logger.log(
            "Please specify number of archs or explicitly pass recipes.",
            LoggerUnited.ERROR_MSG,
            only_main_rank=True,
        )
        if rank == 0:
            arches = (
                self._generate_candidates(N, seed, random_gen=True)
                if arches is None
                else arches
            )
        super().search(jobs_queue=jobs_queue, barrier=barrier, arches=arches)
        if world_size > 1:
            barrier.wait()
        self.set_computed_arches(self.arches_dir)

    def search(self, jobs_queue=None, results_queue=None, barrier=None, arches=None):
        # TODO add support of results_queue
        assert (
            results_queue is None
        ), "Currently results_queue isn't supported; in todo list"

        rank, world_size, _, _ = pytorch_worker_info()
        type_module = "[SEARCHER]"
        process = f"PROCESS {rank}"
        spaces = 10

        if world_size > 1:
            assert barrier is not None, self.logger.log(
                "Pass barrier to search func",
                LoggerUnited.ERROR_MSG,
                only_main_rank=True,
            )

        if rank == 0 and self.computed_arches_dir is not None:
            shutil.copytree(self.computed_arches_dir, self.arches_dir)
            self.set_computed_arches(self.arches_dir)

        if world_size > 1:
            barrier.wait()

        if self.n_initial_steps != 0:
            self.logger.log(
                f"{type_module:<{spaces}} - {process:<{spaces}}: Initial step starts ...",
                LoggerUnited.INFO_MSG,
                only_main_rank=True,
            )
            self._initial_step(
                jobs_queue, barrier, N=self.n_initial_steps, arches=arches
            )
            self.logger.log(
                f"{type_module:<{spaces}} - {process:<{spaces}}: Initial step finished",
                LoggerUnited.INFO_MSG,
                only_main_rank=True,
            )

        for i in range(self.num_trials):
            self.logger.log(
                f"{type_module:<{spaces}} - {process:<{spaces}}: Step {i} starts",
                LoggerUnited.INFO_MSG,
                only_main_rank=True,
            )
            self._step(jobs_queue, barrier)
            self.logger.log(
                f"{type_module:<{spaces}} - {process:<{spaces}}: Step {i} finished",
                LoggerUnited.INFO_MSG,
                only_main_rank=True,
            )

    def _step(self, jobs_queue, barrier):
        """
        Perform NAS step:
        1) update meta-predictor.
        2) generate new candidate architectures.
        3) select the most promising ones.
        4) run training for these architectures.
        """
        rank, world_size, _, _ = pytorch_worker_info()
        type_module = "[SEARCHER]"
        process = f"PROCESS {rank}"
        spaces = 10

        self._train_ensemble(barrier)
        self.logger.log(
            f"{type_module:<{spaces}} - {process:<{spaces}}: Training ensemble finished",
            LoggerUnited.INFO_MSG,
            only_main_rank=True,
        )

        selected_arches = self._select_new_arches(barrier)
        super().search(jobs_queue=jobs_queue, barrier=barrier, arches=selected_arches)
        self.set_computed_arches(self.arches_dir)

    def _train_ensemble(self, barrier):
        """
        Retrain meta predictor using current set of evaluated architectures.
        """
        rank, world_size, _, _ = pytorch_worker_info()
        if rank == 0:
            evaluated_set = self._get_evaluated_archs()
            self.predictor.reset()
            self.predictor.train(
                evaluated_set["features"],
                evaluated_set["scores"],
                epochs=self.cfg["predictor_iters"],
            )
        if world_size > 1:
            barrier.wait()

    def _select_new_arches(self, barrier):
        rank, world_size, _, _ = pytorch_worker_info()
        type_module = "[SEARCHER]"
        process = f"PROCESS {rank}"
        spaces = 10

        new_arches = dict()
        if rank == 0:
            new_features = []
            # TODO change
            generated_arches = self._generate_candidates(
                self.cfg["candidates_to_seed"], random_gen=False
            )
            arches_names = list(generated_arches.keys())

            for arch_hash in arches_names:
                feature_vector = self.model.get_feature_vector(
                    generated_arches[arch_hash]
                )
                new_features.append(feature_vector)

            new_features = torch.tensor(new_features)
            old_features = self._get_evaluated_archs()["features"]
            phi = self.acquisition_func(new_features, self.predictor, old_features)

            if not METRIC_REGISTRY[self.scoring_metric][
                "metric_class"
            ].higher_is_better:
                ac_scores = np.argsort(phi)
            else:
                ac_scores = np.argsort(phi)[::-1]

            selected_names = [arches_names[i] for i in ac_scores][
                : self.cfg["candidates_per_step"]
            ]
            self.logger.log(
                f"{type_module:<{spaces}} - {process:<{spaces}}: Mean new selected arches score: "
                f"{np.mean(list(sorted(phi, reverse=True))[: self.cfg['candidates_per_step']]):.5f}",
                LoggerUnited.INFO_MSG,
                only_main_rank=True,
            )
            for name in selected_names:
                new_arches[name] = generated_arches[name]
        if world_size > 1:
            barrier.wait()
        return new_arches

    def _generate_candidates(self, N, seed=None, random_gen=False):
        """
        Generate candidate archs either by sampling uniformly from the search space or by applying mutations.

        Parameters
        ----------
        N:
            Number of generated architectures
        generation_type:
            Type of candidate generation. Can be 'random' or 'mutation'.
            If 'random', cell archs will be sampled from the search space uniformly.
            If 'mutation', base mutations will be applied to the best evaluated archs.
        seed:
            Random seed

        Returns
        -------
        ndarray:
            Ndarray of generated architectures
        """

        if seed is not None:
            np.random.seed(seed)
        if self.cfg["candidate_generation"]["type"] == "random" or random_gen:
            new_sampled = self.sample_n_arches(N)

        elif self.cfg["candidate_generation"]["type"] == "mutation":
            new_sampled = self._mutate_from_n_top(N)

        else:
            raise NotImplementedError("No such generation method is defined")

        return new_sampled

    def _mutate_from_n_top(self, N):
        """
        At the moment mutation works only if there is only one mutable object.

        """
        if self.logging:
            self.logger.log("MUTATING ~ !@#@!#%$#@^%$^!#! ")

        new_sampled = dict()

        sorted_arch_ids = sorted(
            self.computed_arches_scores.items(), key=lambda kv: kv[1]
        )

        top_part = self.cfg["candidate_generation"].get("top_part", N)
        top_archs_keys = sorted_arch_ids[: int(len(sorted_arch_ids) * top_part)]

        assert (
            len(top_archs_keys) > 0
        ), "There is no sufficient number of validated archs to generate candidates by mutations"

        while len(new_sampled) < N:
            arch_key_to_mutate, _ = random.choice(top_archs_keys)
            new_arch = copy.copy(self.computed_arches[arch_key_to_mutate])
            probas = self.cfg["candidate_generation"].get(
                "probas", [0.25 for _ in range(len(MUTATIONS))]
            )
            for m, proba in zip(
                MUTATIONS,
                probas,
            ):
                self.computed_arches[arch_key_to_mutate], _ = m(
                    self.computed_arches[arch_key_to_mutate], p=proba
                )

            arch_hash = self._hash_arch(new_arch)
            if not arch_hash in self.computed_arches and not arch_hash in new_sampled:
                new_sampled[arch_hash] = new_arch

        return new_sampled
