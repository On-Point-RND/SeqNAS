import attr
from typing import Tuple, List, Dict, Union, Any, Optional

from .utils import read_yaml, write_yaml


@attr.s(auto_attribs=True)
class Config:
    """
    Main config for dataset

    :param train_dataset: path to train dataset
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

    train_dataset: str
    index_columns: List[str]
    sort_columns: List[str]
    target_column: str
    save_path: str
    local_path: Optional[str] = None
    classification: bool = True
    val_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    categorical_columns: List[str] = attr.Factory(list)  # mutable
    numerical_columns: List[str] = attr.Factory(list)  # mutable
    skip_columns: List[str] = attr.Factory(list)
    categorical_cardinality_limit: int = 1000
    export_index_name: str = "seq_id"
    seq_len_limit: int = 800
    seq_len_trunc_type: str = "last"
    min_partition_num: int = 4
    partition_size_mbytes: int = 64
    split_sizes: Dict[str, float] = attr.Factory(
        lambda: dict(train=0.7, val=0.1, test=0.2)
    )
    seed: int = 42

    def __attrs_post_init__(self):
        self.skip_columns += self.index_columns + [self.target_column]

    @classmethod
    def from_file(cls, path: str) -> Any:
        """
        Creates config from file

        :param path: path to info.yaml file
        :return: Config class instance
        """
        cfg = read_yaml(path)
        return cls(**cfg)

    def write(self, path: str):
        """
        Writes config to file

        :param path: path to info.yaml file
        """
        cfg = attr.asdict(self)
        write_yaml(cfg, path)
