import os
from omegaconf import OmegaConf as omg


def patch_env_config(env_config, gpu_num, worker_count):
    if gpu_num is not None:
        env_config.HARDWARE.GPU = gpu_num
    if worker_count is not None:
        env_config.HARDWARE.WORKERS = worker_count

    return env_config


def patch_exp_config(
    exp_config,
    exp_name,
    model_name="",
    data_path="",
    use_amp=False,
    epochs_count=0,
    batch_size=0,
):
    # check experiment name
    if not exp_name:
        raise Exception(f"You didn't specify experiment_name in parameters")
    if exp_name not in exp_config.keys():
        raise Exception(f"Experiment name doesn't exist: {exp_name}")

    # select only needed subconfig
    exp_subconf = exp_config[exp_name]
    exp_subconf["experiment_name"] = exp_name
    exp_subconf["model_name"] = model_name

    # check data path
    if data_path:
        exp_subconf.dataset.data_path = data_path
    data_path = exp_subconf.dataset.data_path

    if not os.path.exists(data_path):
        raise Exception(
            f"Path doesn't exist: {data_path}; specify data_path in parameters"
        )

    # set mixed precision
    exp_subconf.trainer.use_amp = use_amp

    # set epochs_count
    if epochs_count:
        exp_subconf.trainer.epochs = epochs_count

    # set batch_size
    if batch_size:
        exp_subconf.dataset.batch_size = batch_size
    return exp_subconf
