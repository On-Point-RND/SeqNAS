import os
import json
import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import SubsetRandomSampler


class LocalDataset(Dataset):
    def __init__(
        self,
        indices,
        target_index,
        data,
        max_len,
        train=True,
    ):
        self.indices = indices
        self.target_index = target_index
        self.max_len = max_len
        self.train = train
        self.data = data

    def __getitem__(self, index):
        idx = self.indices[index]

        record = dict()
        path = self.data[idx]["path"]
        series = np.load(path, allow_pickle=True)
        s_len = len(series)
        if s_len < self.max_len:
            r = self._pad_zeros(
                self.max_len,
                s_len,
                torch.tensor(series, dtype=torch.float),
            )
        else:
            r = torch.tensor(series, dtype=torch.float)

        if self.train:
            r = self._sample_from_max(self.max_len, s_len, r)
        else:
            r = self._sample_from_start(self.max_len, s_len, r)

        record[self.target_index] = self.data[idx][self.target_index]
        record["model_input"] = r
        return record

    def __len__(self):
        return len(self.indices)

    def _pad_zeros(self, max_len, size, tensor):
        diff = max_len - size
        if diff >= 0:
            zeros = torch.zeros(diff)
            tensor = torch.cat([tensor, zeros], 0)
            return tensor
        else:
            return tensor

    def _sample_from_max(self, max_len, size, tensor):
        diff = size - max_len
        if diff > 0:
            start = random.randint(0, diff - 1)
            end = start + max_len
            tensor = tensor[start:end]
        return tensor

    def _sample_from_start(self, max_len, size, tensor):
        diff = size - max_len
        if diff > 0:
            start = diff
            end = start + max_len
            tensor = tensor[start:end]

        return tensor


class SpeechDataset:
    def __init__(
        self,
        main_index,
        val_portion,
        time_index=None,
        target_index="target",
        categorical=None,
        continious=None,
        path="",
        store_series_path="",
        overwrite_stored_series=True,
        max_len=1000,
        min_len=100,
    ):
        """
        Speech dataset class.
        This class helps to creare generic dataloaders. Splits the data into train and validation and filters sequence length.


        :param main_index: This parameter is used for compatability only - should be removed later
        :param val_portion: A portion of data split into train and validation sets
        :param time_index=None: This parameter is used for compatability only - should be removed later
        :param target_index: Specify the name of the target column or dictionary key
        :param categorical=None: Specify which columns are categorical
        :param continious=None: Specify which columns are continious
        :param path=&quot;&quot;: Specify the path to the json file that contains all of the data. Paths to audo files and descriptive information
        :param store_series_path: This parameter is used for compatability only - should be removed later
        :param overwrite_stored_series=True: This parameter is used for compatability only - should be removed later
        :param max_len=1000: Specify the maximum length of a time series
        :param min_len=100: Filter out series that are too short
        """

        self.max_len = max_len
        self.min_len = min_len

        folder_path = "/" + os.path.join(*path.split("/")[:-1])
        print(path)
        with open(path, "r") as f:
            data = json.load(f)
        self.data = self._set_full_path(data, folder_path)

        (
            self.data,
            self.unique_ids,
            self.total_size,
        ) = self._filter_series_by_min_len(self.data)

        # This solution is good for computing series only once, but at
        # the first run `_generate_series` process everything in memory

        self.target_index = target_index

        self.train_ids, self.val_ids = self._split_train_val(val_portion)
        self.indices = self.train_ids + self.val_ids
        self.mean_len = self.total_size / len(self.indices)

        # made for compatability
        self.cardinalities = []

    def _set_full_path(self, data, folder):
        for idx in data:
            if idx == "max_len" or idx == "min_len":
                continue
            path = data[idx]["path"]
            path = os.path.join(folder, path[2:])
            data[idx]["path"] = path

        return data

    def _split_train_val(self, val_portion):
        val_porrion = int(len(self.unique_ids) * val_portion)
        train_ids = self.unique_ids[val_porrion:]
        val_ids = self.unique_ids[:val_porrion]

        return train_ids, val_ids

    def _filter_series_by_min_len(self, series):
        new_series = dict()
        total_size = 0
        for k in series:
            if k == "max_len" or k == "min_len":
                continue
            if series[k]["len"] > self.min_len:
                total_size += series[k]["len"]
                new_series[k] = dict()
                for g in series[k]:
                    new_series[k][g] = series[k][g]
        return new_series, list(new_series.keys()), total_size

    def get_data_loader(self, batch_size, workers, train=True):
        """
        Creates train or validation dataloaders.

        returns: dataloader objec
        """

        if train:
            indices = self.train_ids
        else:
            indices = self.val_ids

        dataset = LocalDataset(
            indices,
            self.target_index,
            self.data,
            self.max_len,
            train=train,
        )

        # pass sampler instead of copy
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=False,
            shuffle=train,
            drop_last=True,
        )

    def get_splitted_train(self, batch_size, workers, portions=[0.5, 0.5]):
        dataset = LocalDataset(
            self.indices,
            self.target_index,
            self.data,
            self.max_len,
            train=True,
        )

        assert sum(portions) <= 1, "sum of portions should be <= 1"

        total_len = len(self.train_ids)
        splits = [int(total_len * p) for p in portions]

        loaders = []
        indices = set(self.indices)

        running_split = 0
        for split in splits:
            current_ids = set(self.train_ids[running_split : split + running_split])
            sample = [i for i, s in enumerate(indices) if s in current_ids]

            running_split += split

            loaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=workers,
                    pin_memory=False,
                    sampler=SubsetRandomSampler(sample),
                    drop_last=True,
                )
            )

        return loaders

    def print_report(self):
        print("DATASET META INFO")
        print(
            f"Total unique indices: {len(self.unique_ids)}, train: {len(self.train_ids)}, val: {len(self.val_ids)}"
        )
        print(f"Mean series len after min filter {self.mean_len}")
        print("\n")
