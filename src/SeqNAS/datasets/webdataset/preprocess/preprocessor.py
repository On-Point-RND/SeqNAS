import os
import attr
import numpy as np
from typing import Tuple, List, Dict, Union, Any, Optional

from pyspark import StorageLevel

from .config import Config
from .dataset_container import Dataset
from .transform import (
    DatasetTransform,
    SequenceTruncation,
    EncodeCategorical,
    EncodeNumerical,
    EncodeIndex,
    ToSequenceDataset,
)
from .info import DataInfo


@attr.s(auto_attribs=True)
class SparkPreprocessor:
    """
    Class for preparing source dataset to Webdataset format
    It splits data into partitions

    :param config: config in SeqNAS.datasets.webdataset.preprocess.config.Config format
    :param pipeline: transforms for source dataset
    """

    config: Config
    pipeline: Optional[Tuple[DatasetTransform, ...]] = None

    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Creates SparkPreprocessor instance from config file

        :param config_path: path to dataset config .yaml
        :return: SparkPreprocessor class instance
        """
        config = Config.from_file(config_path)
        return cls(config)

    def test_split(
        self, dataset: Any, test_weight: float, index_columns: List[str], seed: int
    ):
        """
        Splits data into train/test

        :param dataset: dataset instance
        :param test_weight: test part
        :param index_columns: index columns
        :param seed: seed to fix splitting
        :return: train and test dataset instances
        """
        # use stratified split if classification task
        if self.config.classification:
            target_column = self.config.target_column
            index_target_df = dataset.select(index_columns + [target_column]).distinct()
            uniq_classes = index_target_df.select(target_column).distinct().collect()

            fractions_dict = {}
            for row in uniq_classes:
                fractions_dict[row[target_column]] = 1 - test_weight

            train = index_target_df.sampleBy(
                target_column, fractions=fractions_dict, seed=seed
            ).select(index_columns)
            test = index_target_df.select(index_columns).subtract(train)
        else:
            indices = dataset.select(*index_columns).distinct()
            indices = indices.persist(StorageLevel.DISK_ONLY_2)
            train, test = indices.randomSplit([1 - test_weight, test_weight], seed=seed)
        train = dataset.join(train, on=list(index_columns), how="inner")
        test = dataset.join(test, on=list(index_columns), how="inner")
        return train, test

    def read_datasets(self, config: Config):
        """
        Reads dataset using config file

        :param config: config in SeqNAS.datasets.webdataset.preprocess.config.Config format
        :return: train, val, test dataset and class weights
        """
        train_dataset = Dataset.from_path(config.train_dataset, "train")
        class_weights = self.get_class_weights(train_dataset.data, config.index_columns)
        val_dataset = None
        test_dataset = None
        if config.val_dataset is not None:
            val_dataset = Dataset.from_path(config.val_dataset, "val")
        else:
            if config.split_sizes["val"] != 0:
                train_dataset.data, val_dataset = self.test_split(
                    train_dataset.data,
                    config.split_sizes["val"],
                    config.index_columns,
                    config.seed,
                )
                val_dataset = Dataset(val_dataset, "val")
                train_dataset.cache(False)
                val_dataset.cache()
        if config.test_dataset is not None:
            test_dataset = Dataset.from_path(config.test_dataset, "test")
        else:
            if config.split_sizes["test"] != 0:
                if config.val_dataset is not None:
                    test_weight = config.split_sizes["test"]
                else:
                    test_weight = config.split_sizes["test"] / (
                        config.split_sizes["test"] + config.split_sizes["train"]
                    )
                train_dataset.data, test_dataset = self.test_split(
                    train_dataset.data, test_weight, config.index_columns, config.seed
                )
                test_dataset = Dataset(test_dataset, "test")
                train_dataset.cache()
                test_dataset.cache()
        return train_dataset, val_dataset, test_dataset, class_weights

    def apply_pipeline(self, dataset: Any):
        """
        Makes dataset transforms

        :param dataset:
        :return: preprocessed dataset
        """
        if dataset is not None:
            for transform in self.pipeline:
                dataset = transform(dataset)
                dataset.cache()
            dataset.unpersist()
        return dataset

    def get_class_weights(self, dataset, index_columns):
        """
        Return class weights according to INS:
        https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4

        :param dataset: spark dataset
        :param index_columns: index column
        :return: list of class weights
        """
        class_weights = []
        if self.config.classification:
            target_column = self.config.target_column
            index_target_df = dataset.select(index_columns + [target_column]).distinct()
            class_weights = [
                int(row["count"])
                for row in index_target_df.groupBy(target_column)
                .count()
                .select("count")
                .collect()
            ]
            class_weights = 1.0 / np.array(class_weights)
            class_weights = (
                class_weights / np.sum(class_weights) * class_weights.shape[0]
            )
            class_weights = class_weights.tolist()
        return class_weights

    def run(self):
        """
        Main function that prepares data. Split data into partitions

        :return: info of dataset
        """
        train_dataset, val_dataset, test_dataset, class_weights = self.read_datasets(
            self.config
        )
        self.pipeline = (
            SequenceTruncation(self.config),
            EncodeCategorical(self.config),
            EncodeNumerical(self.config),
            EncodeIndex(self.config),
            ToSequenceDataset(self.config),
        )
        train_dataset = self.apply_pipeline(train_dataset)
        print("FINISHED TRAIN GENERATION")
        val_dataset = self.apply_pipeline(val_dataset)
        print("FINISHED VAL GENERATION")
        test_dataset = self.apply_pipeline(test_dataset)
        print("FINISHED TEST GENERATION")

        info = DataInfo.from_datasets(
            self.config, train_dataset, val_dataset, test_dataset, class_weights
        )
        info_path = "info.yaml"
        if self.config.local_path is not None:
            info_path = os.path.join(self.config.local_path, info_path)
        info.write(info_path)
        return info
