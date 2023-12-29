import os
import attr
import regex
import logging as log
from typing import Tuple, List, Dict, Union, Any

try:
    import pyspark.sql.functions as fs
except ModuleNotFoundError:
    log.error("PySpark not found!")

from .utils import read_yaml, write_yaml


def download_dir(remote_path, local_path):
    os.system(f"hadoop fs -get {remote_path} {local_path}")


@attr.s(auto_attribs=True)
class DataInfo:
    """
    Class for keeping dataset info

    :param dataset_dir: path where Webdataset will be saved
    :param target_column: target column
    :param target_cardinality: target cardinality
    :param categorical_columns: categorical columns
    :param categorical_cardinality: categorical cardinalities
    :param numerical_columns: numerical column
    :param train_dataset: path to train dataset
    :param val_dataset: path to validation dataset if it exists
    :param test_dataset: path to test dataset if it exists
    :param seq_len: max sequence length
    :param hidden_size: hidden_size of model (optional)
    :param output_size: number of outputs (classes for classification)
    :param emb_hidden: emb_hidden of model (optional)
    :param train_length: number of train sequences
    :param test_length: number of test sequences
    :param val_length: number of val sequences
    :param class_weights: weights for classes when loss is WeightedCrossEntropy (optional)
    :param sort_columns: List of time columns (supports only single)
    """

    dataset_dir: str
    target_column: str
    target_cardinality: int
    categorical_columns: List[str]
    categorical_cardinality: List[int]
    numerical_columns: List[str]
    train_dataset: str
    val_dataset: str
    test_dataset: str
    seq_len: int
    hidden_size: int
    output_size: int
    emb_hidden: int
    train_length: int
    test_length: int
    val_length: int
    class_weights: List[float]
    sort_columns: List[str]

    @staticmethod
    def get_categorical_cardinality(dataset, categorical_columns):
        """
        Returns categorical features cardinalities

        :param dataset: dataset instance
        :param categorical_columns: categorical columns
        :return: categorical cardinalities for dataset
        """
        cardinality = dataset.select(
            *map(lambda c: fs.max(c).alias(c), categorical_columns)
        ).first()
        cardinality = tuple(map(lambda c: cardinality[c] + 1, categorical_columns))
        return cardinality

    @staticmethod
    def get_brace(dataset_meta):
        """
        Returns string paths of partitions

        :param dataset_meta: dataset meta info
        :return: path in string format for all partitions of dataset
        """
        brace = None
        if dataset_meta is not None:
            uri = dataset_meta.data.select(
                fs.input_file_name().alias("input_file")
            ).first()
            uri = uri["input_file"]
            path_to_dir, file = os.path.split(uri)
            _, dir_name = os.path.split(path_to_dir)
            part_range = "{00000.." + f"{dataset_meta.partition_num-1:05d}" + "}"
            brace = regex.sub(r"(?<=part\-)(\d{5})", part_range, file)
            brace = os.path.join(dir_name, brace)
        return brace

    @classmethod
    def from_datasets(
        cls, config, train_dataset, val_dataset, test_dataset, class_weights
    ):
        """
        Function that makes info for dataset

        :param config: SeqNAS.datasets.webdataset.preprocess.config.Config format
        :param train_dataset: train spark dataset
        :param val_dataset: val spark dataset
        :param test_dataset: test spark dataset
        :param class_weights: class weights
        :return: Datainfo instance
        """
        info = dict()
        if config.local_path is None:
            info["dataset_dir"] = config.save_path
        else:
            info["dataset_dir"] = config.local_path
            download_dir(config.save_path, config.local_path)
        info["target_column"] = config.target_column
        if config.classification:
            target_cardinality = (
                train_dataset.data.select(config.target_column).distinct().count()
            )
        else:
            target_cardinality = 1
        info["target_cardinality"] = target_cardinality

        info["categorical_columns"] = config.categorical_columns
        cardinality = cls.get_categorical_cardinality(
            train_dataset.data, config.categorical_columns
        )
        info["categorical_cardinality"] = cardinality
        info["numerical_columns"] = config.numerical_columns

        info["train_dataset"] = cls.get_brace(train_dataset)
        info["val_dataset"] = cls.get_brace(val_dataset)
        info["test_dataset"] = cls.get_brace(test_dataset)

        info["train_length"] = (
            train_dataset.data.select(config.export_index_name).distinct().count()
        )
        info["val_length"] = None
        info["test_length"] = None

        if val_dataset is not None:
            info["val_length"] = (
                val_dataset.data.select(config.export_index_name).distinct().count()
            )
        if test_dataset is not None:
            info["test_length"] = (
                test_dataset.data.select(config.export_index_name).distinct().count()
            )

        info["seq_len"] = config.seq_len_limit
        info["hidden_size"] = None
        info["output_size"] = None
        info["emb_hidden"] = None
        info["class_weights"] = class_weights
        info["sort_columns"] = config.sort_columns

        return cls(**info)

    @classmethod
    def from_file(cls, path, replace_dataset_dir=True):
        """
        Creates DataInfo instance from ready info.yaml file

        :param path: path to info.yaml file
        :param replace_dataset_dir: if replace dataset dir
        :return: DataInfo instance
        """
        cfg = read_yaml(path)
        if replace_dataset_dir:
            dataset_dir, config_name = os.path.split(path)
            cfg["dataset_dir"] = dataset_dir
        return cls(**cfg)

    def write(self, path):
        """
        Function Saves info.yaml file

        :param path: path for saving info
        """
        info = attr.asdict(self)
        write_yaml(info, path)
