from ..preprocess.preprocessor import SparkPreprocessor
from ..preprocess.info import DataInfo
from ..preprocess.config import Config
from .pytorchshardlist import (
    SimpleShardList,
    ResampledShards,
    split_by_node,
    split_by_worker,
)
from .utils import (
    read_table,
    group_series,
    shuffle,
    SequenceCollator,
    pytorch_worker_info,
)
from .connection import download_shards
from .pipeline import DataPipeline

import os
import math
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader
import random

from ... import register_dataset


def SequenceDataset(
    urls: str,
    index_column: str,
    target_column: str,
    sort_columns: str,
    length: int,
    batch_size: int = None,
    num_workers: int = None,
    shuffle_buffer_size: int = 1000,
    transforms: List[Any] = [],
) -> Any:
    """
    Function that returns sequence dataset object

    :param urls: path to dataset files in parquet format
    :param index_column: index column
    :param target_column: target column
    :param sort_columns: time columns for sorting
    :param length: number of sequnces
    :param batch_size: batch_size
    :param num_workers: number of dataloaders (parallel processes which load data)
    :param shuffle_buffer_size: buffer size for shuffling
    :param transforms: transformations for dataset
    :return: dataset instance
    """
    new_transforms = []
    _, world_size, _, _ = pytorch_worker_info()
    use_ddp = int(os.environ.get("USE_DDP", "1"))
    if world_size > 1 and use_ddp:
        new_transforms = [split_by_node]
    new_transforms = (
        new_transforms
        + [
            split_by_worker,
            download_shards,
            read_table,
            group_series(index_column, target_column, sort_columns),
            shuffle(shuffle_buffer_size),
        ]
        + transforms
    )

    if world_size > 1 and use_ddp:
        dataset = DataPipeline(ResampledShards(urls), *new_transforms)
        samples_per_worker = (
            math.ceil(length / (num_workers * world_size * batch_size)) * batch_size
        )
        dataset = dataset.with_epoch(samples_per_worker).with_length(
            samples_per_worker * num_workers
        )
    else:
        UINT_MAX = 0xFFFFFFFF
        shards_seed = random.randint(0, UINT_MAX)
        dataset = DataPipeline(SimpleShardList(urls, seed=shards_seed), *new_transforms)
        dataset = dataset.with_length(length)

    return dataset


@register_dataset("WebSequenceDataset")
class WebSequenceDataset:
    """
    Main dataset class

    There is a realization of Pytorch Webdataset that can work with sequences
    Source implementation: https://github.com/webdataset/webdataset
    """

    def __init__(
        self,
        data_path: str,
        index_columns: List[str],
        sort_columns: List[str],
        target_column: str,
        save_path: Optional[str] = None,
        local_path: Optional[str] = None,
        classification: bool = True,
        val_dataset: Optional[str] = None,
        test_dataset: Optional[str] = None,
        categorical_columns: List[str] = [],  # mutable
        numerical_columns: List[str] = [],  # mutable
        skip_columns: List[str] = [],
        categorical_cardinality_limit=1000,
        export_index_name: str = "seq_id",
        seq_len_limit: int = 800,
        seq_len_trunc_type: str = "last",
        min_partition_num: int = 4,
        partition_size_mbytes: int = 64,
        split_sizes: Dict[str, float] = lambda: dict(train=0.8, val=0.2, test=0.0),
        seed: int = 42,
    ):
        """
        Init WebSequenceDataset

        :param data_path: path to source dataset in [.csv.zip, .csv, .parquet] format
        :param index_columns: List of index columns (supports only single index)
        :param sort_columns: List of time columns (supports only single)
        :param target_column: target column
        :param save_path: path where preprocessed data will be saved
        :param local_path: always equal to save_path (appendix from original webdataset)
        :param classification: whether classification task or regression
        :param val_dataset: path to validation dataset if it exists
        :param test_dataset: path to test dataset if it exists
        :param categorical_columns: categorical columns
        :param numerical_columns: numerical columns
        :param skip_columns: skip columns
        :param categorical_cardinality_limit: limit on different categorical values
        :param export_index_name: index name after preprocessing
        :param seq_len_limit: max sequence length
        :param seq_len_trunc_type: if "last" - keep values from the end, if "first" - keep values from the beginning
        :param min_partition_num: minimum number of partitions into which the dataset will be split
        :param partition_size_mbytes: size of single partition
        :param split_sizes: sizes for splitting into train/val/test
        :param seed: value that fixes splitting dataset in different launches
        """
        if save_path is None:
            save_path = os.path.join(os.path.dirname(data_path), "data_web")
        if local_path is None:
            local_path = save_path

        # change types
        if type(index_columns) is not list:
            index_columns = list(index_columns)
        if type(sort_columns) is not list:
            sort_columns = list(sort_columns)
        if type(categorical_columns) is not list:
            categorical_columns = list(categorical_columns)
        if type(numerical_columns) is not list:
            numerical_columns = list(numerical_columns)
        if type(skip_columns) is not list:
            skip_columns = list(skip_columns)

        assert (
            local_path == save_path
        ), f"Currently don't know difference btw local and save path"

        self.info_path = os.path.join(local_path, "info.yaml")
        self.cfg = Config(
            train_dataset=data_path,
            index_columns=index_columns,
            sort_columns=sort_columns,
            target_column=target_column,
            save_path=save_path,
            local_path=local_path,
            classification=classification,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            skip_columns=skip_columns,
            categorical_cardinality_limit=categorical_cardinality_limit,
            export_index_name=export_index_name,
            seq_len_limit=seq_len_limit,
            seq_len_trunc_type=seq_len_trunc_type,
            min_partition_num=min_partition_num,
            partition_size_mbytes=partition_size_mbytes,
            split_sizes=split_sizes,
            seed=seed,
        )

        self.info = None
        self.continious = numerical_columns
        self.seq_len = seq_len_limit
        self.export_index_name = export_index_name
        self.cardinalities = []
        self.class_weights = []

    def is_created(self):
        """
        Function that checks if dataset is created

        :return: True if dataset is created
        """
        if not os.path.exists(self.info_path) or not os.path.exists(self.cfg.save_path):
            return False
        return True

    def create_dataset(self):
        """
        Function that creates dataset

        :return:
        """
        if not os.path.exists(self.info_path) or not os.path.exists(self.cfg.save_path):
            SparkPreprocessor(self.cfg).run()

            # remove unzipped file
            for unzipped_file_path in [
                self.cfg.train_dataset,
                self.cfg.val_dataset,
                self.cfg.test_dataset,
            ]:
                if unzipped_file_path is not None:
                    if unzipped_file_path.endswith(".csv.zip"):
                        dumped_file_name = os.path.basename(unzipped_file_path)
                        dumped_file_path = os.path.join(
                            os.path.dirname(unzipped_file_path),
                            ".".join(dumped_file_name.split(".")[:-1]),
                        )
                        os.remove(dumped_file_path)

    def load_dataset(self):
        """
        Function that loads dataset info

        :return:
        """
        info = DataInfo.from_file(self.info_path, replace_dataset_dir=False)
        self.info = info
        for i in range(len(self.info.categorical_cardinality)):
            self.cardinalities.append(
                (self.info.categorical_columns[i], self.info.categorical_cardinality[i])
            )
        self.class_weights = self.info.class_weights

    def get_class_weights(self):
        """
        Function that return class weights if use WeightedCrossEntropy loss

        :return: class weights
        """
        return self.class_weights

    def get_data_loader(
        self,
        batch_size: int,
        workers: int,
        train: bool = True,
        last_action: str = "drop",
    ) -> Any:
        """
        General method which create torch.utils.data.Dataloader
        from WebSequenceDataset

        :param batch_size: size of batch in dataloader
        :param workers: number of torch workers in dataloader
        :param train: is dataloader for train, default to True
        :param last_action: What to do with the last batch. If "drop" (default),
            the last batch is dropped. If "keep", the last batch remains
            unchanged as if `drop_last=False` was passed to DataLoader. If
            "pad", the last batch is padded with some data to `batch_size` size.
            The `index` of padded samples is set to -1.
        :return: dataloader
        :rtype: torch.utils.data.Dataloader
        """
        collator = SequenceCollator(
            batch_size,
            self.cfg.classification,
            last_action,
            cat_features=self.info.categorical_columns,
            pad_tail=self.cfg.seq_len_trunc_type == "last",
        )

        if train:
            dataset = SequenceDataset(
                os.path.join(self.info.dataset_dir, self.info.train_dataset),
                index_column=self.export_index_name,
                target_column=self.info.target_column,
                sort_columns=self.info.sort_columns,
                length=self.info.train_length,
                batch_size=batch_size,
                num_workers=workers,
            )
        else:
            dataset = SequenceDataset(
                os.path.join(self.info.dataset_dir, self.info.val_dataset),
                index_column=self.export_index_name,
                target_column=self.info.target_column,
                sort_columns=self.info.sort_columns,
                length=self.info.val_length,
                batch_size=batch_size,
                num_workers=workers,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=(last_action == "drop"),
        )

    def get_test_dataloader(
        self, batch_size: int, workers: int, with_target_column: bool = False
    ) -> Any:
        """
        Method that creates torch.utils.data.Dataloader for test dataset

        :param batch_size: size of batch in dataloader
        :param workers: number of torch workers in dataloader
        :param with_target_column: whether target column exists or not
        :return: dataloader
        :rtype: torch.utils.data.Dataloader
        """
        assert self.info.test_dataset is not None, (
            "You didn't preprocess test dataset. "
            "Pls regenerate data with `test_dataset` parameter in "
            "dataset.dataset_params. Don't forget delete previously generated "
            "data."
        )

        collator = SequenceCollator(
            batch_size,
            self.cfg.classification,
            "pad",
            cat_features=self.info.categorical_columns,
        )

        dataset = SequenceDataset(
            os.path.join(self.info.dataset_dir, self.info.test_dataset),
            index_column=self.export_index_name,
            target_column=None if not with_target_column else self.info.target_column,
            sort_columns=self.info.sort_columns,
            length=self.info.test_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=False,
        )

    def print_report(self):
        """
        Method for print metainformation about dataset

        """
        dataset_info_str = (
            f"DATASET META INFO:\nSeq len: {self.info.seq_len}\n"
            f"Total unique indices: {self.info.train_length+self.info.val_length}, "
            f"train: {self.info.train_length}, val: {self.info.val_length}\n"
            f"Cardinalities:\n"
        )

        cardinality_info_str = ""
        for i in range(len(self.info.categorical_columns)):
            cardinality_info_str += f"{self.info.categorical_columns[i]}: {self.info.categorical_cardinality[i]}\n"

        return dataset_info_str + cardinality_info_str
