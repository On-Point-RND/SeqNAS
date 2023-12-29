from collections.abc import Sequence
import copy
import os
import gc

from .optimizers import OPT
from ..nash_logging.logger import Logger
from ..experiments_src.metrics import init_loss, init_metric, is_clf_metric, LossMetric
from ..utils import randomness
from ..utils.distribute import pytorch_worker_info
from ..utils.misc import disp_arch
from ..experiments_src import METRIC_REGISTRY

import copy
import numpy as np
import time
import torch
import torch.nn.functional as F
import json
from dict_hash import sha256
from tqdm.autonotebook import tqdm
from torch.optim import lr_scheduler
from apex.parallel import DistributedDataParallel
from dataclasses import dataclass, field


class CyclicalLoader:
    """Cycles through pytorch dataloader specified number of steps."""

    def __init__(self, base_dataloader, iters_per_epoch=None):
        self.base_loader = base_dataloader
        assert iters_per_epoch is None or iters_per_epoch > 0
        self._len = iters_per_epoch or len(base_dataloader)
        self._iter = iter(self.base_loader)

    def __len__(self):
        return self._len

    def __iter__(self):
        self._total_iters = 0
        return self

    def __next__(self):
        if self._total_iters >= self._len:
            raise StopIteration

        try:
            item = next(self._iter)
        except StopIteration:
            self._iter = iter(self.base_loader)
            item = next(self._iter)
        self._total_iters += 1
        return item


@dataclass(order=True)
class Prediction:
    """Stores model predictions on the dataset sorted by the score"""

    score: float
    preds: torch.tensor = field(compare=False)


class Trainer:

    """
    1. Trainer performs trainig procedure on given datasets train, val, and other.
        Can be used more than two datasets for different optimization objectives;
    2. Initializes optimizers;
    3. Initializes metrics and loss average meters to track the progress over epochs;
    4. Calls logger if logger is provided.

    - Phases which require training should always start with "train", no idea how to do it better for now e.g.  "train_main", "train_alpha" - gradients will be computed for both parts, gradients won't be computed for "validation".

    - To link optimizers and dataset 'dataset_opt_link' is used. If dataset_opt_link=None all optimizers will compute gradients on all 'train_' sets. If only one train set and only one optimizer is used - then it's not a problem.


    .. highlight:: python
    .. code-block:: python
        trainer = Trainer(
            criterion,
            metrics,
            optimizers=optimizer,
            phases=["train_main", "train_alpha", "validation"],
            num_epochs=exp_cfg.trainer.epochs,
            device=env_config.HARDWARE.GPU,
            logger=logger,
            log_arch=True,
            dataset_opt_link={'train_main':'main','train_alpha':'arch},
        )

        loaders = {
            "train_main": train_loader_one,
            "train_alpha": train_loader_two,
            "validation": val_loader,
        }


        optimizer = {
            "main": (
                "ADAM",
                {
                    "lr": exp_cfg.trainer.lr,
                    "weight_decay": exp_cfg.trainer.weight_decay,
                },
            ),
            "arch": (
                "ADAM",
                {
                    "lr": 0.001,
                    "weight_decay": exp_cfg.trainer.weight_decay,
                },
            ),
        }



    :param criterion: loss function
    :param metrics: a list of metrics
    :param optimizers={"main": ("SGD", {"lr": 1e-3})}  # dict with element for each param group can be Adam or SGD, weight decay can be passed in the same manner
    :param schedulers: dict with learning rate schedulers for param groups. The
        structure is identical to the `optimizers` parameter. The `step()`
        method of the scheduler is called without any argumentsafter every
        iteration. If None (by default), a constant learning rate will be used.
    :param phases: Specify the phases in which the model is trained can be train, val and other optional datasets,
            different model parameters can be trained on different datasets, if dataset is goint to be used for updating parameters we need to
            use prefix 'train'. Model weights and a datasets can be linked in param_groups

    :param num_epochs=3: Specify the number of epochs to train for
    :param iters_per_epoch: Validation is performed every `iters_per_epoch`
        steps. By default len(dataloader) is used as a length of epoch.
    :param device: GPU to train on
    :param logger=False: Pass logger instance to log, if no logger has been passed then no logging will be perfomed
    :param log_arch=False: Log current architechture or not, can be used only with "omnimodels" which have get_arch method.
    :param log_arch_every_epoch=False: if set to True arch is logged after every epoch
    :param trainer_name: trainer name is used to name specific metrics, we can use different trainers within one experiment
        and need to split logs sometimes
    :param save_checkpoint=True: save the best model
    :param metrics_on_train=True: Compute metrics on the train or not.
            It is impossible to compute some of the metrics during train, for some models.
            Or sometimes we want to compute metrics only on validation set to speed up the process.
    :param resume_checkpoint: if set, the training process will be resumed from
        the given checkpoint
    :param use_amp=False: turn on automatic mixed precision
    :param class_weights: weights are forwarded to weighted cross entropy criterion
    :param save_train_preds: if set, the trainer will save model predcitions on the train set for further distillation.
    """

    def __init__(
        self,
        criterion,  # Loss function to train
        metrics=None,  # List of names of computed metrics
        scoring_metric="LossMetric",
        optimizers=None,  # dict with element for each param group
        schedulers=None,
        phases=("train",),  # Phases of training
        num_outputs=2,
        num_epochs=3,  # Number of epochs
        iters_per_epoch=None,
        device="0",
        logger=False,
        log_arch=True,  # If True, logger must be an instance of UnitedLogger
        log_arch_every_epoch=False,
        dataset_opt_link=None,
        dataset_sched_link=None,
        trainer_name="",
        save_checkpoint=True,
        metrics_on_train=True,
        resume_checkpoint=None,
        use_amp=False,
        class_weights=None,
        save_train_preds=False,
    ):
        # can be a list of optimizers / probably need a dictionary ??
        self.dataloaders = None
        self.criterion = criterion

        assert "validation" in phases
        self.phases = phases

        self.num_outputs = num_outputs
        if metrics is not None:
            self.init_metrics = metrics
        else:
            self.init_metrics = []
        self.scoring_metric = scoring_metric

        # if multigpu training rank==gpu_num
        rank, world_size, _, _ = pytorch_worker_info()

        self.multi_gpu = False
        if world_size > 1:
            self.device = rank
            self.use_ddp = int(os.environ.get("USE_DDP", "1"))
            if not self.use_ddp:
                self.multi_gpu = True
        else:
            self.device = int(device)
            self.use_ddp = False

        # TODO change to be a dictionary
        if optimizers is not None:
            self.initial_optimizers = optimizers
        else:
            self.initial_optimizers = {"main": ("SGD", {"lr": 1e-3})}

        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.initial_schedulers = schedulers
        self.logger = logger
        self.log_arch = log_arch
        self.log_arch_every_epoch = log_arch_every_epoch
        self.trainer_name = trainer_name
        self.dataset_opt_link = dataset_opt_link
        self.dataset_sched_link = dataset_sched_link
        self._save_checkpoint = save_checkpoint
        self.resume_checkpoint = resume_checkpoint
        self.metrics_on_train = metrics_on_train
        self.last_complete_epoch = -1
        self.use_amp = use_amp
        self.class_weights = class_weights
        self.save_train_preds = save_train_preds
        self.reset_best_score()

        # last computed loss used for backward
        self.loss = None
        self._preds = None
        self._score = None
        self.best_prediction = Prediction(-float("inf"), None)

        self.distill = False
        self.distill_predictions = []
        self.distill_predictions_fnames = []
        self.distill_weight = None

    def setup_distillation(self, on, predictions=None, distillation_weight=0.2):
        """Enable knowledge distillation during training.

        KD loss is the mean of KD losses of each prediction. Each loss is MSE loss between the model output
        and the output stored in one element of `predictions` list. KD loss has weight `distillation_weight`
        in training loss, and usual loss has weight `1 - distillation_weight`.

        :param on: enable or disable KD
        :type on: bool
        :param predictions: predictions to distill on. If filenames are specified, checkpoints will be searched
            in the `predictions` folder in the root of the experiment. These filenames will also be logged with
            architecture. If `Prediction`s are passed, they are used as is and nothing is logged in the `distill_predictions_fnames` field.
        :type predictions: sequence of str or sequence of Prediction
        :param distillation_weight: KD loss weight in training loss. Will be clipped to [0, 1].
        :type distillation_weight: float
        """
        self.distill = on
        self.distill_predictions = []
        self.distill_predictions_fnames = []
        self.distill_weight = None
        if on == False:
            return

        assert isinstance(predictions, Sequence)
        self.distill_weight = torch.clamp(torch.tensor(distillation_weight), 0, 1).to(
            self.device
        )
        for pred in predictions:
            if isinstance(pred, str):
                self.distill_predictions_fnames.append(pred)
                pred = torch.load(self.logger.get_exp_root_dir() / "predictions" / pred)

            self.distill_predictions.append(
                Prediction(
                    score=torch.tensor(pred.score),
                    preds=torch.tensor(pred.preds).to(self.device),
                )
            )

    def set_model(self, model, param_groups="default"):
        """
        The set_model function is used to set the model to train. It also initializes the optimizers for each group of parameters (default or specified by user).
        """

        # TODO: update optimizer prameters dict
        self.model = model
        self.cpu_model_copy = copy.deepcopy(model)
        torch.cuda.set_device(self.device)
        self.model.cuda(self.device)

        if self.use_ddp:
            self.model = DistributedDataParallel(self.model)

        if param_groups == "default":
            param_groups = {"main": self.model.parameters()}
        self._init_optimizers(param_groups)
        (
            self.metrics,
            self.metrics_history,
            self.loss_func,
        ) = self._init_metrics_and_loss(self.init_metrics, self.criterion)
        self.last_complete_epoch = -1

    def reset_best_score(self):
        self.best_model_state = None
        if not METRIC_REGISTRY[self.scoring_metric]["metric_class"].higher_is_better:
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def free_model_mem(self):
        del self.model, self.cpu_model_copy
        gc.collect()
        torch.cuda.empty_cache()

    def _num_used_parameters(self):
        arch = self.cpu_model_copy.get_arch()
        params_dict = {
            k: v.numel()
            for k, v in self.cpu_model_copy.named_parameters()
            if v.requires_grad
        }
        s = 0
        for k in params_dict.keys():
            arch_key, _, suffix = k.partition(".ops.")

            if len(suffix) == 0:
                s += params_dict[k]
                continue

            idx, _, _ = suffix.partition(".")
            if not idx.isdigit():
                s += params_dict[k]
                continue

            # chech if arch actually uses the parameter (its weight is positive)
            if arch_key in arch and arch[arch_key][int(idx)] > 0:
                s += params_dict[k]

        return s

    # See method usage in Random searcher
    def _init_optimizers(self, param_groups: dict):
        self.optimizers = dict()
        self.schedulers = dict()
        self.scaler_dict = dict()

        if not isinstance(param_groups, dict):
            param_groups = self._dictify(param_groups, initial=["main"])

        # TODO check that keys in opt_groups match keys in self.initial_optimizers
        for opt_group in param_groups:
            opt_name, opt_params = self.initial_optimizers[opt_group]
            opt_params["params_dict"] = param_groups[opt_group]
            optimizer = OPT[opt_name](**opt_params)

            if (
                self.initial_schedulers is not None
                and opt_group in self.initial_schedulers
            ):
                sched_name, sched_params = self.initial_schedulers[opt_group]
                sched = getattr(lr_scheduler, sched_name)(optimizer, **sched_params)
                self.schedulers[opt_group] = sched

            self.optimizers[opt_group] = optimizer
            self.scaler_dict[opt_group] = torch.cuda.amp.GradScaler(
                enabled=self.use_amp
            )

    def set_dataloaders(self, dataloaders):
        """
        The set_dataloaders function sets the dataloaders for the model.
        """
        self.dataloaders = {}
        for phase, loader in dataloaders.items():
            if phase == "train":
                self.dataloaders[phase] = CyclicalLoader(loader, self.iters_per_epoch)
            else:
                self.dataloaders[phase] = loader

    def _init_metrics_and_loss(self, metrics, criterion):
        """
        Function initializes metrics and loss function for all phases
        :param metrics: list of metric names
        :param criterion: name of loss function
        """
        metrics_dict = {}
        metrics_history = {}
        loss_params = None

        # init loss
        if criterion == "WeightCrossEntropyLoss":
            assert self.class_weights is not None, "class_weights is None"
            loss_params = {"weight": torch.tensor(self.class_weights).to(self.device)}
        loss = init_loss(criterion, loss_params)

        # init metrics history
        for phase in self.phases:
            metrics_history[phase] = {}
            for metric in metrics:
                metrics_history[phase][metric] = []
            if "LossMetric" not in metrics:
                # additional history for loss metric
                metrics_history[phase]["LossMetric"] = []

        # init metrics
        for phase in self.phases:
            metrics_dict[phase] = torch.nn.ModuleDict()
            for metric in metrics:
                metric_params = None
                if is_clf_metric(metric):
                    metric_params = {"num_classes": self.num_outputs}
                    if metric == "f1_macro":
                        metric_params["average"] = "macro"
                    elif metric == "f1_weighted":
                        metric_params["average"] = "weighted"

                metrics_dict[phase][metric] = init_metric(
                    metric, device=self.device, metric_params=metric_params
                )

            if "LossMetric" not in metrics:
                metrics_dict[phase]["LossMetric"] = LossMetric().to(self.device)

        return metrics_dict, metrics_history, loss

    def _update_metrics(self, preds, target, loss, phase):
        """
        The _update_metrics function updates the metrics dictionary.
        Updates metrics for all phases. The _update_metrics function
        is called in every iteration of training or testing.
        """
        for metric_name in self.metrics[phase].keys():
            if metric_name != "LossMetric":
                if is_clf_metric(metric_name):
                    preds = torch.nn.functional.softmax(preds, dim=1)
                self.metrics[phase][metric_name].update(preds, target)
            else:
                self.metrics[phase][metric_name].update(loss)

    def _compute_metrics(self, phase):
        """
        Function computes final metrics after epoch.
        """
        for metric_name in self.metrics[phase].keys():
            self.metrics_history[phase][metric_name].append(
                self.metrics[phase][metric_name].compute().cpu()
            )

    def _opt_zero_grad(self, phase):
        self._iter_and_call_optim("zero_grad", phase)

    def _opt_step(self, phase):
        self._iter_and_call_optim("step", phase)

    def _iter_and_call_optim(self, method_name, phase):
        if phase != "validation":
            if isinstance(self.optimizers, dict):
                if self.dataset_opt_link is None:
                    for name in self.optimizers:
                        if method_name == "step":
                            self.scaler_dict[name].scale(self.loss).backward()
                            self.scaler_dict[name].step(self.optimizers[name])
                            self.scaler_dict[name].update()
                        else:
                            method = getattr(self.optimizers[name], method_name)
                            method()
                else:
                    name = self.dataset_opt_link[phase]
                    if method_name == "step":
                        self.scaler_dict[name].scale(self.loss).backward()
                        self.scaler_dict[name].step(self.optimizers[name])
                        self.scaler_dict[name].update()
                    else:
                        method = getattr(self.optimizers[name], method_name)
                        method()

            else:
                method = getattr(self.optimizers, method_name)
                method()

    def _sched_step(self, phase):
        if self.dataset_sched_link is None:
            for sched in self.schedulers.values():
                sched.step()
        else:
            name = self.dataset_sched_link[phase]
            self.schedulers[name].step()

    def _log(self, message, type=None):
        if self.logger is not None:
            rank, _, _, _ = pytorch_worker_info()
            only_main_rank = True
            stdout = True
            if self.multi_gpu:
                only_main_rank = False
                stdout = False

            type_module = "[TRAINER]"
            process = f"PROCESS {rank}"
            spaces = 10
            message = f"{type_module:<{spaces}} - {process:<{spaces}}: " + message
            self.logger.log(
                message=message, type=type, only_main_rank=only_main_rank, stdout=stdout
            )

    def load_checkpoint(self, ckpt):
        """Load training state from checkpoint.

        :param ckpt: checkpoint
        :type ckpt: dict or path-like
        """

        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt)

        self.model.load_state_dict(ckpt["model_state"])
        for pg in ckpt["opt_state"]:
            self.optimizers[pg].load_state_dict(ckpt["opt_state"][pg])
        for pg in ckpt["sched_state"]:
            self.schedulers[pg].load_state_dict(ckpt["sched_state"][pg])
        for pg in ckpt["amp_scaler_state"]:
            self.scaler_dict[pg].load_state_dict(ckpt["amp_scaler_state"][pg])
        self.last_complete_epoch = ckpt["epoch"]
        self.metrics_history = ckpt["metrics_history"]
        randomness.set_global_state(ckpt["randomness"])

        if "distill_on" in ckpt:
            self.setup_distillation(
                ckpt["distill_on"], ckpt["distill_predictions"], ckpt["distill_weight"]
            )

        if "extra_data" in ckpt:
            return ckpt["extra_data"]

    def save_checkpoint(self, path=None, extra_data_to_save=None, only_main_rank=True):
        """
        Create checkpoint

        :param path: checkpoint filename, default is logger default
        :type path: bytes path-like
        :param extra_data_to_save: some data to save in checkpoint.
        :type extra_data_to_save: any
        :param only_main_rank: if True, only main process saves the checkpoint
        :type only_main_rank: bool
        """

        score = self.metrics_history["validation"][self.scoring_metric][-1]
        _, world_size, _, _ = pytorch_worker_info()
        model_state = self.model.state_dict()
        if self.use_ddp:
            new_model_state = {}
            for k, v in model_state.items():
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                new_model_state[name] = v
            model_state = new_model_state
        opt_state = {opt: self.optimizers[opt].state_dict() for opt in self.optimizers}
        sched_state = {
            sched: self.schedulers[sched].state_dict() for sched in self.schedulers
        }
        amp_scaler_state = {
            sc: self.scaler_dict[sc].state_dict() for sc in self.scaler_dict
        }

        state = {
            "model_state": model_state,
            "opt_state": opt_state,
            "sched_state": sched_state,
            "amp_scaler_state": amp_scaler_state,
            "epoch": self.last_complete_epoch,
            "metrics_history": self.metrics_history,
            "randomness": randomness.get_global_state(),
            "distill_on": self.distill,
            "distill_weight": self.distill_weight,
            "distill_predictions": self.distill_predictions,
        }
        if extra_data_to_save is not None:
            state["extra_data"] = extra_data_to_save

        if not METRIC_REGISTRY[self.scoring_metric]["metric_class"].higher_is_better:
            is_best = score <= self.best_score
        else:
            is_best = score >= self.best_score

        if is_best:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model_state)

        self.logger.save_checkpoint(state, is_best, path, only_main_rank=only_main_rank)

    def _dictify(self, object, initial=[]):
        # TODO check if object is iterable

        if isinstance(object, dict):
            return object
        else:
            # Assume inputs and targets go first
            labels = initial
            if len(object) - len(initial) > 0:
                # Numerate other by keys if keys were not provided
                labels += list(range(len(object) - len(labels)))
            return {k: v for (k, v) in zip(labels, object)}

    def _data_to_device(self, data, device):
        # recursivly set to device
        def to_device(obj):
            for key in obj:
                if isinstance(obj[key], dict):
                    to_device(obj[key])
                else:
                    if key not in ["index"]:
                        obj[key] = obj[key].to(device)

        to_device(data)

        return data

    def _process_batch(self, batch):
        batch = self._dictify(batch, initial=["model_input", "target"])
        batch = self._data_to_device(batch, self.device)
        return batch

    # def _iterate_dataset():
    def _log_arch(self, epoch=None, total_time=None):
        if self.log_arch:
            if hasattr(self.cpu_model_copy, "get_arch"):
                arch = self.cpu_model_copy.get_arch()

                # model is final and its architecture can't be logged
                if len(arch) == 1 and "empty" in arch and arch["empty"] == 0:
                    return
                n_par = "nd"

                try:
                    for key in arch:
                        if isinstance(arch[key], torch.Tensor):
                            values = list(arch[key][0].detach().cpu().numpy())
                        else:
                            values = arch[key]

                        if isinstance(values, list):
                            values = [round(v, 3) for v in values]
                        else:
                            values = round(values, 3)

                        self._log(f"layer: {key}")
                        self._log(f"values: {values}")

                    n_par = self._num_used_parameters()
                except Exception:
                    pass

                fname = f"arch_{self.cpu_model_copy.__class__.__name__}_{n_par}_params_{sha256(arch)}.json"
                if epoch is not None:
                    fname = f"epoch_{epoch}_" + fname

                best_score = self.get_best_epoch_metrics(
                    scoring_metric=self.scoring_metric, phase="validation"
                )
                for metric in best_score.keys():
                    best_score[metric] = best_score[metric].detach().cpu().item()

                time_per_epoch = None
                if total_time is not None and epoch is not None:
                    time_per_epoch = total_time / epoch

                final_dict = {
                    "arch": arch,
                    "model": self.cpu_model_copy.__class__.__name__,
                    "params": n_par,
                    "epoch": epoch,
                    "total_time": total_time,
                    "time_per_epoch": time_per_epoch,
                    "metrics": best_score,
                    "scoring_metric": best_score[self.scoring_metric],
                    "distillation": self.distill,
                }
                if self.distill:
                    final_dict[
                        "distill_predictions_fnames"
                    ] = self.distill_predictions_fnames.copy()

                content = json.dumps(final_dict)
                self.logger.save_custom_txt(content, fname, "arches")

                if self.save_train_preds:
                    pred_fname = f"{fname[:-5]}.pth.tar"
                    pred_fname = f"score_{self.best_prediction.score}_{pred_fname}"
                    self.logger.save_custom_torch(
                        self.best_prediction, pred_fname, "predictions"
                    )

    def _log_metrics(self, epoch):
        for phase in self.phases:
            metrics = self.get_last_epoch_metrics(phase=phase)
            if self.logger is not None:
                self.logger.log_metrics(f"{phase}", metrics, epoch)

    def _iterate_one_epoch(self, phase):
        # Each epoch has a training and validation phase

        rank, world_size, _, _ = pytorch_worker_info()
        show_tqdm = True
        if self.use_ddp:
            if rank != 0:
                show_tqdm = False
        if self.multi_gpu:
            show_tqdm = False

        if phase == "validation":
            self.model.eval()
        else:
            self.model.train()

        # Iterate over data.
        n_batches = len(self.dataloaders[phase])

        for batch in tqdm(
            self.dataloaders[phase], total=n_batches, disable=not show_tqdm
        ):
            batch = self._process_batch(batch)
            self._opt_zero_grad(phase)

            # forward
            # track history if only in train
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                with torch.set_grad_enabled(phase.startswith("train")):
                    time = batch["time"]
                    if hasattr(self.model, "accepts_time") and self.model.accepts_time:
                        preds = self.model(batch["model_input"], time)["preds"]
                    else:
                        preds = self.model(batch["model_input"])["preds"]
                    if phase == "train":
                        self._preds[batch["index"]] = (
                            preds.detach().cpu().to(self._preds.dtype)
                        )
                    target = batch["target"]
                    self.loss = self.loss_func(preds, target)
                    if self.distill:
                        distill_loss = torch.tensor(0.0, device=self.device)
                        n_teachers = torch.tensor(
                            len(self.distill_predictions), device=self.device
                        )
                        for teacher in self.distill_predictions:
                            teacher_logits = teacher.preds[batch["index"]]
                            valid = torch.isfinite(teacher_logits).all(-1)
                            teacher_logits = teacher_logits[valid]
                            student_logits = preds[valid]
                            distill_loss += (
                                0.5
                                * F.mse_loss(student_logits, teacher_logits)
                                / n_teachers
                            )

                        self.loss *= 1 - self.distill_weight
                        self.loss += self.distill_weight * distill_loss

                    self._update_metrics(preds, target, self.loss, phase)

            # backward + optimize only if in training phase
            if phase.startswith("train"):
                self._opt_step(phase)
                self._sched_step(phase)

        # use as a dummy input to compute a graph
        self.last_batch = batch

    def terminator_chek_if_nan_or_inf(self):
        """
        The terminator_chek_if_nan_or_inf function is a helper function that checks if the loss is nan or inf.
        If it is, then it terminates the training process and prints out an error message.
        """
        flag = False
        for phase in self.phases:
            loss = self.get_last_epoch_metrics(phase=phase)["LossMetric"]
            if not torch.isfinite(loss) or torch.isnan(loss):
                self.logger.log(
                    "TERMINATING NA or INF in loss encountered", type=Logger.ERROR_MSG
                )
                flag = True

        return flag

    def train(self):
        """
        Train model with properties which was set at __init__
        """

        if self.save_train_preds:
            (self.logger.get_exp_root_dir() / "predictions").mkdir(exist_ok=True)

        if self.resume_checkpoint:
            self.load_checkpoint(self.resume_checkpoint)

        try:
            n_par = self._num_used_parameters()
        except Exception as e:
            self._log(
                f"Can't count parameters due to the exception: {e}. "
                "Continue training.",
                type=Logger.WARNING_MSG,
            )
        else:
            self._log(f"Training model containing {n_par} parameters")

        bs = self.dataloaders["train"].base_loader.batch_size
        n_batches = len(self.dataloaders["train"].base_loader)
        start_time = time.time()
        for epoch in range(self.last_complete_epoch + 1, self.num_epochs):
            print_epoch = epoch + 1
            self._score = None
            self._preds = torch.full(
                ((n_batches + 1) * bs, self.num_outputs), torch.nan, dtype=torch.float32
            )
            for phase in self.phases:
                self._log(" Epoch {}/{}".format(print_epoch, self.num_epochs))
                phase_start_time = time.time()
                self._iterate_one_epoch(phase)
                phase_end_time = time.time()

                # add last metric value to history
                self._compute_metrics(phase)

                metrics = self.get_last_epoch_metrics(phase)
                metric_string = " | ".join(
                    [f"{key}:{value:.9f}" for key, value in metrics.items()]
                )

                self._log(
                    f"{phase.upper()} {metric_string} | time: {phase_end_time - phase_start_time:.2f}s"
                )
                if phase == "validation":
                    self._score = self.get_best_epoch_metrics(self.scoring_metric)[
                        self.scoring_metric
                    ].item()

            if self.save_train_preds:
                score_for_pred = self._score
                if not METRIC_REGISTRY[self.scoring_metric][
                    "metric_class"
                ].higher_is_better:
                    score_for_pred = -score_for_pred
                self.best_prediction = max(
                    self.best_prediction, Prediction(score_for_pred, self._preds)
                )

            self.last_complete_epoch = epoch
            self.save_checkpoint()
            self.reset_metrics()

            if self.terminator_chek_if_nan_or_inf():
                break

            if self.log_arch_every_epoch:
                self._log_arch(epoch)

            self._log_metrics(print_epoch)

        time_elapsed = time.time() - start_time

        if self.log_arch:
            self._log_arch(self.num_epochs, time_elapsed)

        self._log(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def get_history(self, phase="validation"):
        """
        Return history of all computing metrics
        """
        history = dict()
        for k in self.metrics_history[phase]:
            history[k] = self.metrics_history[phase][k]

        return history

    def get_last_epoch_metrics(self, phase="validation"):
        """
        Return last computed metric in hisrory
        """
        metrics = dict()
        for k in self.metrics_history[phase].keys():
            metrics[k] = self.metrics_history[phase][k][-1]

        return metrics

    def get_best_epoch_metrics(self, scoring_metric, phase="validation"):
        """
        Return Best metrics within all epoches.
        """
        arg_func = torch.argmax
        if not METRIC_REGISTRY[self.scoring_metric]["metric_class"].higher_is_better:
            arg_func = torch.argmin

        scoring_metric = [
            it.cpu() for it in self.metrics_history[phase][scoring_metric]
        ]
        best_ep_idx = arg_func(torch.stack(scoring_metric, dim=-1))

        metrics = dict()
        for k in self.metrics_history[phase]:
            metrics[k] = self.metrics_history[phase][k][best_ep_idx].cpu()

        return metrics

    def reset_metrics(self):
        """
        The reset_metrics function resets all history for the specified phase.
        For example, if you wanted to reset the training loss and accuracy history,
        you would call reset_metrics('training') after each epoch.
        """
        for phase in self.phases:
            for k in self.metrics[phase].keys():
                self.metrics[phase][k].reset()
