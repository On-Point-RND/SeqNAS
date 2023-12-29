import numpy as np

from SeqNAS.experiments_src.pipeline import get_dataset, get_model
from SeqNAS.experiments_src.metrics import init_metric
from SeqNAS.utils.config import patch_env_config
from SeqNAS.utils.misc import data_to_device, print_arch, print_params

import os
import torch
import pandas as pd
from torch.nn.functional import softmax
from optparse import OptionParser
from tqdm.autonotebook import tqdm
from omegaconf import OmegaConf as omg

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def eval_one_model(loader, model, scoring_metric=None, dumps=None):
    df_dict = {"seq_id": [], "target": []}
    prediction = []
    num_classes = None
    # Iterate over data.
    n_batches = len(loader)
    for batch in tqdm(loader, total=n_batches):
        batch = data_to_device(batch, gpu_num)
        with torch.set_grad_enabled(False):
            if hasattr(model, "accepts_time") and model.accepts_time:
                preds = model(batch["model_input"], batch["time"])["preds"]
            else:
                preds = model(batch["model_input"])["preds"]
            pred_prob = softmax(preds, dim=1).detach().cpu()
            df_dict["seq_id"].extend(batch["index"].tolist())
            df_dict["target"].extend(batch["target"].tolist())
            prediction.append(pred_prob)
            if num_classes is None:
                num_classes = pred_prob.shape[1]

    prediction = np.vstack(prediction)
    df = pd.DataFrame.from_dict(df_dict)
    df = pd.concat([df, pd.DataFrame(prediction)], axis=1)
    df = df[df["seq_id"] != -1]
    if dumps is not None:
        if len(dumps) == 0:
            dumps["target"] = df.set_index("seq_id")["target"]
            dumps["preds"] = [df.set_index("seq_id").iloc[:, 1:]]
        else:
            dumps["preds"].append(df.set_index("seq_id").iloc[:, 1:])

    metric = init_metric(
        scoring_metric, device="cpu", metric_params={"num_classes": num_classes}
    )
    score = metric(
        torch.tensor(df.iloc[:, 2:].to_numpy(), device="cpu"),
        torch.tensor(df["target"].to_numpy(), device="cpu"),
    )
    print(f"Test Score: {score.item()}")


if __name__ == "__main__":
    """
    Model evaluation
    Run from repository root. Example:

    python examples/main_examples/eval_local_test.py
    --experiment_path=experiments/vtb_transactions_webdataset/train_final/EncoderDecoderModelRandom/2022-10-24_09:03:51
    --batch_size=256
    --gpu_num=0
    --worker_count=4

    Returns test scores
    """

    parser = OptionParser()
    parser.add_option(
        "--experiment_path",
        dest="experiment_path",
        help="path to folder with experiment results, used for final training.",
        type="str",
        default="",
    )
    parser.add_option(
        "--batch_size", dest="batch_size", help="batch size", type="int", default=0
    )
    parser.add_option(
        "--gpu_num",
        dest="gpu_num",
        help="number of gpu (only single)",
        type="int",
        default=0,
    )
    parser.add_option(
        "--worker_count",
        dest="worker_count",
        help="number of cpus for data loading. Advised using less than cpu count",
        type="int",
        default=1,
    )
    (options, args) = parser.parse_args()

    env_config_path = os.path.join(options.experiment_path, "config.yaml")
    exp_config_path = os.path.join(options.experiment_path, "exp_cfg.yaml")

    env_config = omg.load(env_config_path)
    exp_config = omg.load(exp_config_path)

    if options.batch_size:
        exp_config.dataset.batch_size = options.batch_size

    env_config = patch_env_config(
        env_config, gpu_num=options.gpu_num, worker_count=options.worker_count
    )

    gpu_num = env_config.HARDWARE.GPU
    model_name = exp_config.model_name
    checkpoint_path = os.path.join(options.experiment_path, "checkpoints")

    dataset_type = exp_config.dataset.dataset_type
    assert (
        dataset_type == "WebSequenceDataset"
    ), "Currently supports only WebSequenceDataset type of dataset"
    dataset = get_dataset(dataset_type, exp_config)
    dataset.create_dataset()
    dataset.load_dataset()
    dataset.print_report()

    loader = dataset.get_test_dataloader(
        batch_size=exp_config.dataset.batch_size,
        workers=env_config.HARDWARE.WORKERS,
        with_target_column=True,
    )

    dumps = {}
    scoring_metric = exp_config.trainer.scoring_metric

    for ckpt_name in os.listdir(checkpoint_path):
        state_path = os.path.join(checkpoint_path, ckpt_name)
        state = torch.load(state_path, map_location=torch.device("cpu"))
        hidden_size = state["params"]["hidden_size"]
        embeddings_hidden = state["params"]["embeddings_hidden"]
        dropout = state["params"].get("dropout", 0.0)
        num_embeddings_hidden = state["params"].get("num_embeddings_hidden", "auto")
        augmentations = state["params"].get("augmentations", None)
        exp_config.model.hidden_size = hidden_size
        exp_config.model.embeddings_hidden = embeddings_hidden
        exp_config.model.dropout = dropout
        exp_config.model.num_embeddings_hidden = num_embeddings_hidden
        exp_config.model.augmentations = augmentations
        model = get_model(model_name, dataset, exp_config)
        print_params(state["params"])
        if state["arch"] is not None:
            print_arch(state["arch"])
            model.set_arch(state["arch"])
            model.reset_weights()
        model.load_state_dict(state["model_state"])

        torch.cuda.set_device(gpu_num)
        model.cuda(gpu_num)
        model.eval()

        print(f"Val Score: {state['score']}")
        eval_one_model(loader, model, scoring_metric, dumps)

    metric = init_metric(
        scoring_metric,
        device="cpu",
        metric_params={"num_classes": dumps["preds"][0].shape[1]},
    )
    preds = pd.concat(
        dumps["preds"], axis=1, keys=np.arange(len(dumps["preds"])), join="inner"
    )
    preds = preds.groupby(axis=1, level=1).mean()
    pred = torch.tensor(preds.to_numpy(), device="cpu")
    tgt = torch.tensor(dumps["target"].to_numpy(), device="cpu")
    score = metric(pred, tgt)
    print(f"\nAveraged model score: {score}")
