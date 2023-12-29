import os
import json
import bisect
import random
import copy
import pandas as pd
from tqdm import tqdm
import math
import zipfile
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import SubsetRandomSampler, default_collate


from ..nash_logging.io_utils import makedir
from . import register_dataset


def _pad_tensor_valid(tensor, target_bs):
    """
    Pads tensor to desired batch size with data already present in the tensor.
    """
    actual_bs = len(tensor)
    reps = (math.ceil(target_bs / actual_bs), *(1,) * (tensor.ndim - 1))
    return torch.tile(tensor, reps)[:target_bs]


class LocalDataset(Dataset):
    def __init__(
        self,
        indices,
        target_index,
        store_series_path,
        max_len,
        main_keys,
        train=True,
    ):
        self.indices = indices
        self.target_index = target_index
        self.store_series_path = store_series_path
        self.max_len = max_len
        self.main_keys = main_keys
        self.train = train

    def __getitem__(self, index):
        idx = self.indices[index]

        record = dict()
        with open(f"{self.store_series_path}/{idx}.json", "r") as f:
            cur_series = json.load(f)
        for g in self.main_keys:
            if g == self.target_index:
                record[g] = torch.tensor(cur_series[g])
            else:
                if cur_series["size"] < self.max_len:
                    record[g] = self._pad_zeros(
                        self.max_len,
                        cur_series["size"],
                        torch.tensor(cur_series[g], dtype=torch.float),
                    )
                else:
                    record[g] = torch.tensor(cur_series[g], dtype=torch.float)

        if self.train:
            record = self._sample_from_max(self.max_len, cur_series["size"], record)
        else:
            record = self._sample_last(self.max_len, cur_series["size"], record)

        record["index"] = torch.tensor(idx, dtype=torch.float)

        # Set output to a form that all what goes into a model is sotred in 'model_input'
        final_record = dict()
        final_record["model_input"] = dict()

        for g in record:
            # rename target feature to "target" for training
            if g == self.target_index:
                final_record["target"] = record[g]
            elif g == "index":
                final_record[g] = record[g]
            else:
                final_record["model_input"][g] = record[g]

        return final_record

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

    def _sample_from_max(self, max_len, size, records):
        diff = size - max_len
        if diff > 0:
            start = random.randint(0, diff - 1)
            end = start + max_len

            for g in records:
                if g == self.target_index:
                    continue
                records[g] = records[g][start:end]
        return records

    def _sample_last(self, max_len, size, records):
        diff = size - max_len
        if diff > 0:
            start = diff
            end = start + max_len
            for g in records:
                if g == self.target_index:
                    continue
                records[g] = records[g][start:end]
        return records


@register_dataset("TimeSeriesInMemoryDataset")
class TimeSeriesInMemoryDataset:
    """
    A wrapper over a Dataset class that contains meta-information and useful methods.

    :param main_index: name of column with object's index
    :type main_index: string
    :param val_portion: part of object to validation
    :type val_portion: float in [0; 1]
    :param time_index: name of column with sample's time
    :type time_index: string
    :param target_index: name of column with object's target
    :type target_index: string
    :param categorical: list with names of columns with categorical features, defaults to []
    :type categorical: list, optional
    :param continious: list with names of columns with numerical features, defaults to []
    :type continious: list, optional
    :param path: path to csv file with data, defaults to ""
    :type path: str, optional
    :param store_series_path: path to which would save preprocessed objects from dataset in json format, defaults to ""
    :type store_series_path: str, optional
    :param overwrite_stored_series: is dataset should create new json files from csv file, defaults to True
    :type overwrite_stored_series: bool, optional
    :param max_len: length of sequences in dataset to which objects would padded or cropped, defaults to 1000
    :type max_len: int, optional
    :param min_len: minimum length of objects in dataset before padding, objects with smaller length would be dropped, defaults to 100
    :type min_len: int, optional
    """

    def __init__(
        self,
        data_path,
        main_index,
        time_index,
        target_index,
        val_portion,
        seq_len=1000,
        min_len=100,
        categorical=[],
        continious=[],
        store_series_path=None,
        overwrite_stored_series=False,
        seed=0,
    ):
        """ """

        self.max_len = seq_len
        self.min_len = min_len
        self.target_index = target_index

        if store_series_path is None:
            store_series_path = os.path.join(os.path.dirname(data_path), "data")
        self.store_series_path = store_series_path
        # This solution is good for computing series only once, but at
        # the first run `_generate_series` process everything in memory
        if not os.path.exists(self.store_series_path) or overwrite_stored_series:
            if data_path.endswith(".csv"):
                data = pd.read_csv(data_path)
            elif data_path.endswith(".csv.zip"):
                with zipfile.ZipFile(data_path, "r") as thezip:
                    file_name = os.path.basename(data_path)
                    file_name = ".".join(file_name.split(".")[:-1])
                    data = pd.read_csv(thezip.open(file_name))
            else:
                raise Exception(
                    "Not supported type of dataset, supports only: [.csv, .csv.zip]"
                )

            data = data.sample(frac=1)
            data = data.to_dict(orient="list")

            (
                series,
                self.groups_mappers,
                self.total_size,
            ) = self._generate_series(
                data,
                main_index,
                time_index,
                categorical,
                continious,
                target_index,
            )
            # NOTE: `self.total_size` returned from function above might be irrelevant after filtering
            series, self.unique_ids = self._filter_series_by_min_len(series, time_index)
            self._store_series_and_meta(
                series,
                self.groups_mappers,
                self.unique_ids,
                self.total_size,
                self.store_series_path,
            )
            del series
        else:
            (
                self.groups_mappers,
                self.unique_ids,
                self.target_list,
                self.total_size,
            ) = self._load_meta(self.store_series_path)

        # TODO: check whether we need this field
        self.cardinalities = [
            (n, len(self.groups_mappers[n].keys()))
            for n in self.groups_mappers
            if n in categorical
        ]

        self.train_ids, self.val_ids = self._stratified_shuffle_split(val_portion, seed)
        self.indices = self.train_ids + self.val_ids
        self.mean_len = self.total_size / len(self.indices)

        self.main_keys = set(categorical + continious + [target_index])

    def _store_series_and_meta(
        self, series, groups_mappers, unique_ids, total_size, path_to_store
    ):
        self.target_list = []
        for _un_idx in unique_ids:
            self.target_list.append(series[_un_idx][self.target_index])

        makedir(path_to_store)
        with open(f"{path_to_store}/metadata.json", "w") as f:
            metadata = {
                "groups_mappers": groups_mappers,
                "unique_ids": unique_ids,
                "target": self.target_list,
                "total_size": total_size,
            }
            json.dump(metadata, f)
        for key in series.keys():
            with open(f"{path_to_store}/{str(key)}.json", "w") as f:
                json.dump(series[key], f)

    def _load_meta(self, path_to_store):
        with open(f"{path_to_store}/metadata.json", "r") as f:
            metadata = json.load(f)
        return (
            metadata["groups_mappers"],
            metadata["unique_ids"],
            metadata["target"],
            metadata["total_size"],
        )

    def _split_train_val(self, val_portion):
        # random.seed(777)
        # random.shuffle(self.unique_ids)
        val_porrion = int(len(self.unique_ids) * val_portion)
        train_ids = self.unique_ids[val_porrion:]
        val_ids = self.unique_ids[:val_porrion]

        return train_ids, val_ids

    def _stratified_shuffle_split(self, val_portion, seed=0):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_portion, random_state=seed
        )
        train_index, val_index = sss.split(self.unique_ids, self.target_list).__next__()
        return [self.unique_ids[idx] for idx in train_index], [
            self.unique_ids[idx] for idx in val_index
        ]

    def _filter_series_by_min_len(self, series, time_index):
        new_series = dict()
        for k in series:
            if len(series[k][time_index]) > self.min_len:
                new_series[k] = dict()
                for g in series[k]:
                    new_series[k][g] = series[k][g]
        return new_series, list(new_series.keys())

    def _generate_series(
        self,
        data,
        main_index,
        time_index,
        categorical,
        continious,
        target_index,
    ):
        """
        Generate ordered in time time series a dict of value.
        """
        print("Generating sequencial dataset...")

        series = dict()
        columns = set(categorical + [time_index, target_index, "size"])
        groups_mappers = {g: dict() for g in categorical}
        total_size = 0
        main_index_mapper = {
            data[main_index][i]: i for i in range(len(data[main_index]))
        }
        for i, user in tqdm(enumerate(data[main_index]), total=len(data[main_index])):
            user = main_index_mapper[user]
            if not user in series:
                series[user] = {key: [] for key in columns}
                series[user]["size"] = 0
                series[user][target_index] = data[target_index][i]
            current_time = data[time_index][i]
            insert_index = bisect.bisect(series[user][time_index], current_time)
            series[user][time_index].insert(insert_index, current_time)
            series[user]["size"] = series[user]["size"] + 1
            total_size += 1

            for g in categorical + continious:
                if g in continious:
                    if g in series[user]:
                        series[user][g].insert(insert_index, data[g][i])
                    else:
                        series[user][g] = [data[g][i]]
                else:
                    # categorical encoding
                    if g in groups_mappers:
                        if data[g][i] in groups_mappers[g]:
                            value = groups_mappers[g][data[g][i]]
                        else:
                            value = (
                                len(groups_mappers[g]) + 1
                            )  # keep 0 for padding, no zero entries
                            groups_mappers[g][data[g][i]] = value

                    series[user][g].insert(insert_index, value)

        return series, groups_mappers, total_size

    def get_data_loader(self, batch_size, workers, train=True, last_action="drop"):
        """
        General method which create torch.utils.data.Dataloader
        from TimeSeriesInMemoryDataset


        :param batch_size: size of batch in dataloader
        :type batch_size: int
        :param workers: number of torch workers in dataloader
        :type workers: int
        :param train: is dataloader for train, default to True
        :type train: bool, optional
        :param last_action: What to do with the last batch. If "drop" (default),
            the last batch is dropped. If "keep", the last batch remains
            unchanged as if `drop_last=False` was passed to DataLoader. If
            "pad", the last batch is padded with some data to `batch_size` size.
            The `index` of padded samples is set to -1.
        :type last_action: str

        :return: dataloader
        :rtype: torch.utils.data.Dataloader
        """

        assert last_action in ("drop", "keep", "pad")

        dataset = LocalDataset(
            self.indices,
            self.target_index,
            self.store_series_path,
            self.max_len,
            self.main_keys,
            train=train,
        )

        indices = set(self.indices)
        train_ids = set(self.train_ids)
        val_dis = set(self.val_ids)
        if train:
            sample = [i for i, s in enumerate(indices) if s in train_ids]
        else:
            sample = [i for i, s in enumerate(indices) if s in val_dis]

        def collate_fn(data):
            actual_bs = len(data)
            batch = default_collate(data)
            if actual_bs == batch_size or last_action != "pad":
                return batch

            inputs = batch["model_input"]
            for k in inputs:
                inputs[k] = _pad_tensor_valid(inputs[k], batch_size)

            batch["target"] = _pad_tensor_valid(
                batch["target"],
                batch_size,
            )

            missing_len = batch_size - actual_bs
            batch["index"] = torch.cat((batch["index"], torch.full((missing_len,), -1)))

            return batch

        # pass sampler instead of copy
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=SubsetRandomSampler(sample),
            collate_fn=collate_fn,
            drop_last=(last_action == "drop"),
        )

    def get_splitted_train(self, batch_size, workers, portions=[0.5, 0.5]):
        """
        Support method for create several dataloaders from train part of dataset

        :param batch_size: size of batch
        :type batch_size: int
        :param workers: number of torch workers to dataset
        :type workers: int
        :param portions: parts of train dataset in each dataloader, defaults to [0.5, 0.5]
        :type portions: list of floats in [0, 1] with shape (n_loaders,) and sum of elements <= 1, optional

        :return: N dataloaders where N is length of portions
        :rtype: list of torch.utils.data.Dataloader
        """
        dataset = LocalDataset(
            self.indices,
            self.target_index,
            self.store_series_path,
            self.max_len,
            self.main_keys,
            train=True,
        )

        assert sum(portions) <= 1, "sum of portions should be <= 1"

        ids = copy.copy(self.train_ids + self.val_ids)
        random.shuffle(ids)

        total_len = len(ids)
        splits = [int(total_len * p) for p in portions]

        loaders = []
        indices = set(self.indices)

        running_split = 0
        for split in splits:
            current_ids = set(ids[running_split : split + running_split])
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
        """
        Method for print metainformation about dataset
        """
        print("DATASET META INFO")
        print(
            f"Total unique indices: {len(self.unique_ids)}, train: {len(self.train_ids)}, val: {len(self.val_ids)}"
        )
        print("Cardinalities:")
        for cardinalities_tuple in self.cardinalities:
            name, val = cardinalities_tuple
            print(f"{name}: {val}")
        print(f"Mean series len after min filter {self.mean_len}")
