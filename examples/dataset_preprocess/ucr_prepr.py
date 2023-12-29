from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import tempfile
import zipfile
import time
import os

from sktime.datasets import load_from_arff_to_dataframe, load_from_tsfile


def seq2table(df: pd.DataFrame):
    new_df = pd.DataFrame()
    for _, row in df.iterrows():
        seq_dict = {}
        for i in range(df.shape[1] - 2):
            seq_dict[f"dim_{i}"] = row[f"dim_{i}"]
        seq_df = pd.DataFrame(data=seq_dict)
        seq_df["id"] = row["id"]
        seq_df["target"] = row["target"]
        seq_df["time"] = list(range(seq_df.shape[0]))
        new_df = pd.concat([new_df, seq_df], axis=0)
    return new_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ucr_zip",
        type=str,
        help="path to UCR zipped dataset",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="path to folder where dataset will be saved",
    )
    options = parser.parse_args()

    if not os.path.exists(options.save_folder):
        os.mkdir(options.save_folder)

    start_time = time.time()
    chunk_size = 100
    with zipfile.ZipFile(options.ucr_zip) as myzip:
        with tempfile.TemporaryDirectory() as tempdir:
            myzip.extractall(tempdir)
            dataset_files = list(Path(tempdir).glob("**/*.ts"))
            load_dataset_func = load_from_tsfile
            if len(dataset_files) == 0:
                dataset_files = list(Path(tempdir).glob("**/*.arff"))
                load_dataset_func = load_from_arff_to_dataframe
                if len(dataset_files) == 0:
                    raise Exception("Format not supported; must be in [.arff, .ts]")

            for file_path in dataset_files:
                X, y = load_dataset_func(file_path)
                X["id"] = X.index
                X["target"] = y.astype(str)
                new_df = pd.DataFrame()

                tasks = []
                with Pool() as pool:
                    chunk_num = 0
                    while chunk_num * chunk_size < X.shape[0]:
                        tasks.append(
                            pool.apply_async(
                                seq2table,
                                (
                                    X.iloc[
                                        chunk_num
                                        * chunk_size : (chunk_num + 1)
                                        * chunk_size,
                                        :,
                                    ],
                                ),
                            )
                        )
                        chunk_num += 1

                    while True:
                        all_finished = True

                        for i, task in enumerate(tasks):
                            if task.ready():
                                new_df = pd.concat([new_df, task.get()])
                            else:
                                all_finished = False

                        if all_finished:
                            break

                    new_file_name = file_path.name.split(".")[0] + ".parquet"
                    save_path = Path(options.save_folder) / new_file_name
                    new_df.to_parquet(
                        save_path,
                        index=False,
                    )

    finished_time = time.time()
    print(f"Total time: {finished_time - start_time}")
