from SeqNAS.experiments_src.pipeline import get_dataset, get_model
from SeqNAS.utils.config import patch_env_config
from SeqNAS.utils.misc import data_to_device, print_arch, print_params

import os
import torch
import pandas as pd
from pathlib import Path
from torch.nn.functional import softmax
from optparse import OptionParser
from tqdm.autonotebook import tqdm
from omegaconf import OmegaConf as omg


def eval_one_model(loader, model, save_path):
    df_dict = {"seq_id": [], "prediction": []}
    # Iterate over data.
    n_batches = len(loader)
    for batch in tqdm(loader, total=n_batches):
        batch = data_to_device(batch, gpu_num)
        with torch.set_grad_enabled(False):
            preds = model(batch["model_input"], batch["time"])["preds"]
            pred_prob = softmax(preds, dim=1)[:, 1].tolist()
            df_dict["seq_id"].extend(batch["index"].tolist())
            df_dict["prediction"].extend(pred_prob)

    df = pd.DataFrame.from_dict(df_dict)
    df = df[df["seq_id"] != -1]

    # prepare seq id
    seq_id_path = Path(
        os.path.join(
            os.path.dirname(exp_config.dataset.data_path), "data_web", "seq_id_test"
        )
    )
    seq_id_df = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in seq_id_path.glob("*.parquet")
    )
    df = pd.merge(seq_id_df, df, on="seq_id")
    df.drop("seq_id", axis=1, inplace=True)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    """
    Model evaluation
    Tested only for AMEX competition https://www.kaggle.com/competitions/amex-default-prediction
    Run from repository root. Example:

    python examples/main_examples/eval_kaggle.py
    --experiment_path=experiments/amex/train_final/EncoderDecoderModelRandom/2022-10-04_14:06:59
    --batch_size=256
    --gpu_num=0
    --worker_count=4

    Returns .csv file with results
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
    save_dir = os.path.join(options.experiment_path, "submissions")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

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
        with_target_column=False,
    )

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
        eval_one_model(
            loader, model, os.path.join(save_dir, f"{ckpt_name.split('.')[0]}.csv")
        )
