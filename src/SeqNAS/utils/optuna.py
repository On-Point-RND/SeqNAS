from ..experiments_src.pipeline import get_dataset, get_model, get_logger, get_trainer
from ..experiments_src import METRIC_REGISTRY
from ..nash_logging.logger import Logger
from . import randomness

import os
import copy
import time
import torch
import shutil
import optuna
import numpy as np
from pathlib import Path
from dict_hash import sha256
from omegaconf import OmegaConf
from collections import defaultdict
from optuna.trial import TrialState
from typing import Dict, List, Optional


class Objective:
    def __init__(
        self,
        rank,
        exp_name,
        env_config,
        exp_config,
        hyp_config,
        model_name,
        repeat_pruner,
        arch=None,
    ):
        self.rank = rank
        self.exp_name = exp_name
        self.env_config = env_config
        self.exp_config = exp_config
        self.hyp_config = hyp_config
        self.model_name = model_name
        self.arch = arch
        self.scoring_metric = self.exp_config.trainer.scoring_metric
        self.dataset_type = self.exp_config.dataset.dataset_type
        self.best_score = init_best_score(self.scoring_metric)
        self.repeat_pruner = repeat_pruner

        metrics = exp_config.trainer.metrics
        loss = exp_config.trainer.loss

        env_config.EXPERIMENT.DIR = str(Path(env_config.EXPERIMENT.DIR) / exp_name)
        self.logger = get_logger(env_config)
        OmegaConf.save(exp_config, Path(env_config.EXPERIMENT.DIR) / "exp_cfg.yaml")
        OmegaConf.save(hyp_config, Path(env_config.EXPERIMENT.DIR) / "hyp_config.yaml")

        checkpoint_dir = os.path.join(env_config.EXPERIMENT.DIR, "tmp_checkpoints")
        if rank == 0:
            os.mkdir(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

        dataset_type = exp_config.dataset.dataset_type
        dataset = get_dataset(dataset_type, exp_config)
        dataset.load_dataset()

        self.trainer = get_trainer(
            exp_config,
            loss,
            metrics,
            self.logger,
            optimizer=None,
            scheduler=None,
            device=env_config.HARDWARE.GPU,
            class_weights=dataset.get_class_weights(),
        )
        self.trainer.log_arch = False

    def get_suggestion(self, trial, param, name):
        if param.trial_method == "suggest_categorical":
            return trial.suggest_categorical(name, **param.trial_method_params)
        elif param.trial_method == "suggest_float":
            return trial.suggest_float(name, **param.trial_method_params)
        elif param.trial_method == "suggest_int":
            return trial.suggest_int(name, **param.trial_method_params)
        else:
            raise ValueError(
                f"not existing method: {param.trial_method}; "
                f"supports: [suggest_categorical, suggest_float, suggest_int]"
            )

    def get_train_params(self, trial):
        batch_size = self.get_suggestion(
            trial, self.hyp_config.train_params.batch_size, "batch_size"
        )
        optimizer_name = self.get_suggestion(
            trial, self.hyp_config.train_params.optimizer, "optimizer"
        )
        lr = self.get_suggestion(
            trial, self.hyp_config.train_params.optimizer_params.lr, "lr"
        )
        weight_decay = self.get_suggestion(
            trial,
            self.hyp_config.train_params.optimizer_params.weight_decay,
            "weight_decay",
        )

        optimizer_params = {}
        optimizer_params["lr"] = lr
        optimizer_params["weight_decay"] = weight_decay
        beta1 = self.get_suggestion(
            trial, self.hyp_config.train_params.optimizer_params.beta1, "beta1"
        )
        if optimizer_name == "FUSEDSGD":
            optimizer_params["momentum"] = beta1
        elif optimizer_name in ["FUSEDADAM", "FUSEDNOVOGRAD", "FUSEDLAMB"]:
            optimizer_params["betas"] = (beta1, 0.999)

        optimizer = {"main": (optimizer_name, optimizer_params)}
        model_params = {}
        for param_name in self.hyp_config.model_params.keys():
            param = self.get_suggestion(
                trial, self.hyp_config.model_params[param_name], param_name
            )
            model_params[param_name] = param
        return batch_size, optimizer, model_params

    def save_best_checkpoint(self, trial, filename, score):
        arch = None
        if hasattr(self.trainer.cpu_model_copy, "get_arch"):
            arch = self.trainer.cpu_model_copy.get_arch()
        model_state = self.trainer.best_model_state
        opt_state = {
            opt: self.trainer.optimizers[opt].state_dict()
            for opt in self.trainer.optimizers
        }
        sched_state = {
            sched: self.trainer.schedulers[sched].state_dict()
            for sched in self.trainer.schedulers
        }

        state = {
            "arch": arch,
            "model_state": model_state,
            "opt_state": opt_state,
            "sched_state": sched_state,
            "epoch": self.trainer.last_complete_epoch,
            "metrics_history": self.trainer.metrics_history,
            "randomness": randomness.get_global_state(),
            "params": trial.params,
            "score": score,
        }
        torch.save(state, filename)

    def __call__(self, trial):
        batch_size, optimizer, model_params = self.get_train_params(trial)
        self.repeat_pruner.check_params()

        self.exp_config.dataset.batch_size = batch_size
        for param_name in model_params.keys():
            self.exp_config.model[param_name] = model_params[param_name]
        self.trainer.initial_optimizers = optimizer

        dataset = get_dataset(self.dataset_type, self.exp_config)
        dataset.load_dataset()

        model = get_model(self.model_name, dataset, self.exp_config)
        if self.arch is not None:
            model.set_arch(self.arch)
            model.reset_weights()

        train_loader = dataset.get_data_loader(
            batch_size=self.exp_config.dataset.batch_size,
            workers=self.env_config.HARDWARE.WORKERS,
            train=True,
        )
        val_loader = dataset.get_data_loader(
            batch_size=self.exp_config.dataset.batch_size,
            workers=self.env_config.HARDWARE.WORKERS,
            train=False,
        )
        loaders = {"train": train_loader, "validation": val_loader}

        self.trainer.set_dataloaders(loaders)
        self.trainer.set_model(copy.deepcopy(model))

        self.exp_config.dataset.batch_size = batch_size

        # log options
        type_module = "[SEARCHER]"
        process = f"PROCESS {self.rank}"
        spaces = 10

        start_time = time.time()
        try:
            self.trainer.train()
        except RuntimeError as e:
            self.logger.log(
                message=f"{type_module:<{spaces}} - {process:<{spaces}}: " f"{e}",
                type=Logger.ERROR_MSG,
                stdout=False,
            )
            return None
        end_time = time.time()

        score = self.trainer.get_best_epoch_metrics(self.scoring_metric)[
            self.scoring_metric
        ]
        if new_score_is_better(score, self.best_score, self.scoring_metric):
            if self.arch is None:
                weights_path = os.path.join(
                    self.checkpoint_dir, f"model_best_{self.rank}.pth.tar"
                )
            else:
                weights_path = os.path.join(
                    self.checkpoint_dir, f"{self.rank}_{sha256(self.arch)}.pth.tar"
                )
            self.best_score = score
            self.save_best_checkpoint(trial, weights_path, score)
        self.trainer.free_model_mem()
        self.trainer.reset_best_score()

        self.logger.log(
            f"{type_module:<{spaces}} - {process:<{spaces}}: TRAINED {self.model_name}. "
            f"Score: {score:.5f}; Time: {end_time - start_time:.2f} sec; "
            f"Time/epoch: {(end_time - start_time) / self.exp_config.trainer.epochs:.2f}; "
            f"Params: {trial.params}",
            stdout=False,
        )
        return score


class ParamRepeatPruner:
    """Prunes reapeated trials, which means trials with the same paramters won't waste time/resources."""

    def __init__(
        self,
        study: optuna.study.Study,
        repeats_max: int = 0,
        should_compare_states: List[TrialState] = [TrialState.COMPLETE],
        compare_unfinished: bool = True,
    ):
        """
        Args:
            study (optuna.study.Study): Study of the trials.
            repeats_max (int, optional): Instead of prunning all of them (not repeating trials at all, repeats_max=0) you can choose to repeat them up to a certain number of times, useful if your optimization function is not deterministic and gives slightly different results for the same params. Defaults to 0.
            should_compare_states (List[TrialState], optional): By default it only skips the trial if the paremeters are equal to existing COMPLETE trials, so it repeats possible existing FAILed and PRUNED trials. If you also want to skip these trials then use [TrialState.COMPLETE,TrialState.FAIL,TrialState.PRUNED] for example. Defaults to [TrialState.COMPLETE].
            compare_unfinished (bool, optional): Unfinished trials (e.g. `RUNNING`) are treated like COMPLETE ones, if you don't want this behavior change this to False. Defaults to True.
        """
        self.should_compare_states = should_compare_states
        self.repeats_max = repeats_max
        self.repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.unfinished_repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.compare_unfinished = compare_unfinished
        self.study = study

    @property
    def study(self) -> Optional[optuna.study.Study]:
        return self._study

    @study.setter
    def study(self, study):
        self._study = study
        if self.study is not None:
            self.register_existing_trials()

    def register_existing_trials(self):
        """In case of studies with existing trials, it counts existing repeats"""
        trials = self.study.trials
        trial_n = len(trials)
        for trial_idx, trial_past in enumerate(self.study.trials[1:]):
            self.check_params(trial_past, False, -trial_n + trial_idx)

    def prune(self):
        self.check_params()

    def should_compare(self, state):
        return any(state == state_comp for state_comp in self.should_compare_states)

    def clean_unfinised_trials(self):
        trials = self.study.trials
        finished = []
        for key, value in self.unfinished_repeats.items():
            if self.should_compare(trials[key].state):
                for t in value:
                    self.repeats[key].append(t)
                finished.append(key)

        for f in finished:
            del self.unfinished_repeats[f]

    def check_params(
        self,
        trial: Optional[optuna.trial.BaseTrial] = None,
        prune_existing=True,
        ignore_last_trial: Optional[int] = None,
    ):
        if self.study is None:
            return
        trials = self.study.trials
        if trial is None:
            trial = trials[-1]
            ignore_last_trial = -1

        self.clean_unfinised_trials()

        self.repeated_idx = -1
        self.repeated_number = -1
        for idx_p, trial_past in enumerate(trials[:ignore_last_trial]):
            should_compare = self.should_compare(trial_past.state)
            should_compare |= (
                self.compare_unfinished and not trial_past.state.is_finished()
            )
            if should_compare and trial.params == trial_past.params:
                if not trial_past.state.is_finished():
                    self.unfinished_repeats[trial_past.number].append(trial.number)
                    continue
                self.repeated_idx = idx_p
                self.repeated_number = trial_past.number
                break

        if self.repeated_number > -1:
            self.repeats[self.repeated_number].append(trial.number)
        if len(self.repeats[self.repeated_number]) > self.repeats_max:
            if prune_existing:
                raise optuna.exceptions.TrialPruned()

        return self.repeated_number

    def get_value_of_repeats(
        self, repeated_number: int, func=lambda value_list: np.mean(value_list)
    ):
        if self.study is None:
            raise ValueError("No study registered.")
        trials = self.study.trials
        values = (
            trials[repeated_number].value,
            *(
                trials[tn].value
                for tn in self.repeats[repeated_number]
                if trials[tn].value is not None
            ),
        )
        return func(values)


def save_best_checkpoint(checkpoint_tmp_dir, scoring_metric, ckpt_dir=None):
    best_score = init_best_score(scoring_metric)
    best_ckpt_name = None
    new_ckpt_dir = ckpt_dir or os.path.join(
        os.path.dirname(checkpoint_tmp_dir), "checkpoints"
    )
    if not os.path.exists(new_ckpt_dir):
        os.mkdir(new_ckpt_dir)
    for ckpt_name in os.listdir(checkpoint_tmp_dir):
        state = torch.load(
            os.path.join(checkpoint_tmp_dir, ckpt_name),
            map_location=torch.device("cpu"),
        )
        score = state["score"]
        if new_score_is_better(score, best_score, scoring_metric):
            best_score = score
            best_ckpt_name = ckpt_name
    shutil.copy(
        os.path.join(checkpoint_tmp_dir, best_ckpt_name),
        os.path.join(new_ckpt_dir, best_ckpt_name),
    )
    shutil.rmtree(checkpoint_tmp_dir)
    os.remove(os.path.join(os.path.dirname(checkpoint_tmp_dir), "checkpoint.pth.tar"))
    os.remove(os.path.join(os.path.dirname(checkpoint_tmp_dir), "model_best.pth.tar"))


def save_trials_dataframe(study, filename):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)


def new_score_is_better(score, best_score, scoring_metric):
    if not METRIC_REGISTRY[scoring_metric]["metric_class"].higher_is_better:
        if score <= best_score:
            return True
        return False
    else:
        if score >= best_score:
            return True
        return False


def init_best_score(scoring_metric):
    best_score = -np.inf
    if not METRIC_REGISTRY[scoring_metric]["metric_class"].higher_is_better:
        best_score = np.inf
    return best_score
