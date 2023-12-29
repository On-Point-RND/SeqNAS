import os
import attr
import zipfile
from typing import Tuple, List, Dict, Union, Any, Optional

from pyspark.sql import SparkSession


def check_dataset_type(dtype: str):
    """
    Checks that dtype of dataset in train/val/test

    :param dtype: type of data
    """
    known_types = ("train", "val", "test")
    if dtype not in known_types:
        ValueError(f"Unknown dataset type {dtype}! Use one of {known_types}")


@attr.s
class Dataset:
    """
    Dataset low level class

    :param data: data to be cached
    :param dtype: type of dataset train/val/test
    :param partition_num: number of partitions
    :param cached_versions: cached versions of data
    """

    data: Any = attr.ib()
    dtype: str = attr.ib(validator=attr.validators.in_(("train", "val", "test")))
    partition_num: Optional[int] = attr.ib(default=None)
    cached_versions: List[int] = attr.ib(factory=list)

    @staticmethod
    def read_data(path: str):
        """
        Function that reads dataset

        :param path: path to dataset in [.csv.zip, .csv, .parquet format]
        :return: spark dataset
        """
        spark = SparkSession.builder.getOrCreate()
        if path.endswith(".parquet"):
            dataset = spark.read.parquet(path)
        elif path.endswith(".csv"):
            dataset = spark.read.option("header", True).csv(path)
        elif path.endswith(".csv.zip"):
            _dirname = os.path.dirname(path)
            with zipfile.ZipFile(path, "r") as thezip:
                thezip.extractall(_dirname)
            _file_name = os.path.basename(path)
            _file_path = os.path.join(_dirname, ".".join(_file_name.split(".")[:-1]))
            dataset = spark.read.option("header", True).csv(_file_path)
        else:
            raise Exception(
                "Not supported type of dataset, supports only: [.csv, .csv.zip]"
            )

        return dataset

    @classmethod
    def from_path(cls, path: str, dtype: str):
        """
        Read dataset from path

        :param path: path to dataset
        :param dtype: dataset type train/val/test
        :return: Dataset class instance
        """
        dataset = cls.read_data(path)
        return cls(dataset, dtype)

    def cache(self, releasable=True):
        """
        Caches dataset

        :param releasable:
        :return:
        """
        self.data = self.data.cache()
        if releasable:
            self.cached_versions.append(self.data)

    def unpersist(self):
        for data in self.cached_versions:
            data.unpersist()
        self.cached_versions = list()
