import os
from typing import Tuple, List, Dict, Union, Any
from abc import ABC, abstractmethod

import pyspark.sql.functions as fs
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from .dataset_container import Dataset
from .sparklabelencoder import LabelEncoder

from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline


class DatasetTransform(ABC):
    """
    Base class for dataset transforms

    """

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def transform(self, dataset):
        pass

    def fit_transform(self, dataset):
        return self.fit(dataset).transform(dataset)

    def __call__(self, dataset_meta: Dataset):
        if dataset_meta.dtype == "train":
            dataset = self.fit_transform(dataset_meta.data)
        else:
            dataset = self.transform(dataset_meta.data)
        dataset_meta.data = dataset
        return dataset_meta


class SequenceTruncation(DatasetTransform):
    """
    Truncate sequence according to config

    """

    def __init__(self, config):
        self.index_columns = config.index_columns
        self.sort_columns = config.sort_columns
        self.seq_len_limit = config.seq_len_limit
        self.seq_len_trunc_type = config.seq_len_trunc_type
        self.target_column = config.target_column
        self.numerical_columns = config.numerical_columns

    def transform(self, dataset):
        if self.seq_len_limit < 1:
            return dataset

        if self.seq_len_trunc_type == "last":
            order_type = fs.desc
        else:
            order_type = fs.asc

        assert (
            len(self.sort_columns) == 1
        ), "Currenctly supports only single sort column"
        sort_column = self.sort_columns[0]
        if sort_column not in self.numerical_columns:
            dataset = dataset.withColumn(
                sort_column,
                fs.when(
                    fs.to_timestamp(fs.col(sort_column)).cast("long").isNotNull(),
                    fs.to_timestamp(fs.col(sort_column)).cast("long"),
                ).otherwise(fs.col(sort_column).cast("long")),
            )

        window = Window.partitionBy(*self.index_columns).orderBy(
            *map(order_type, [sort_column])
        )
        dataset = (
            dataset.withColumn("row_number", fs.row_number().over(window))
            .filter(fs.col("row_number") <= self.seq_len_limit)
            .drop("row_number")
        )
        return dataset

    def fit(self, dataset):
        return self


class EncodeCategorical(DatasetTransform):
    """
    Encode categorical features according to config

    """

    def __init__(self, config):
        self.categorical_columns = config.categorical_columns
        self.numerical_columns = config.numerical_columns
        self.sort_columns = config.sort_columns
        self.skip_columns = config.skip_columns
        self.cardinality_limit = config.categorical_cardinality_limit
        self.encoders_path = os.path.join(config.save_path, "categorical_encoders")
        self.classification = config.classification
        self.target_column = config.target_column
        self.encoder_parititon_num = 8
        self.broadcast = True
        self.target_column_is_cat = False

    def guess_categorical_columns(self, dataset):
        not_cat_cols = self.skip_columns + self.numerical_columns + self.sort_columns
        cat_cols = tuple(filter(lambda c: c not in not_cat_cols, dataset.columns))
        stats = dataset.select(
            *map(lambda c: fs.approx_count_distinct(c).alias(c), cat_cols)
        ).first()
        cat_cols = tuple(filter(lambda c: stats[c] <= self.cardinality_limit, cat_cols))
        self.categorical_columns.extend(cat_cols)

    def fit(self, dataset):
        if len(self.categorical_columns) < 1:
            self.guess_categorical_columns(dataset)

        for col in self.categorical_columns:
            encoder = LabelEncoder(
                col,
                f"{col}_index_le",
                path=os.path.join(self.encoders_path, col),
                partitions_num=self.encoder_parititon_num,
                start_value=1,
            )
            encoder.fit(dataset)

        col_dtypes = dataset.dtypes
        for col in col_dtypes:
            if col[0] == self.target_column:
                if col[1] == "string" and self.classification:
                    self.target_column_is_cat = True
                    encoder = LabelEncoder(
                        self.target_column,
                        f"{self.target_column}_index_le",
                        path=os.path.join(self.encoders_path, self.target_column),
                        partitions_num=1,
                        start_value=0,
                    )
                    encoder.fit(dataset)
                break

        return self

    def transform(self, dataset):
        for col in self.categorical_columns + [self.target_column]:
            if col == self.target_column:
                if col not in dataset.columns or not self.target_column_is_cat:
                    continue
            encoder = LabelEncoder(
                col,
                f"{col}_index_le",
                broadcast=self.broadcast,
                path=os.path.join(self.encoders_path, col),
            )
            dataset = encoder.transform(dataset)
            if col in self.sort_columns:
                dataset = dataset.withColumn(f"_sort_{col}", fs.col(col))
            dataset = dataset.drop(col).withColumnRenamed(f"{col}_index_le", col)
            dataset = dataset.fillna({col: 0})
        if self.classification and self.target_column in dataset.columns:
            dataset = dataset.withColumn(
                self.target_column, fs.col(self.target_column).cast("long")
            )
        return dataset


class EncodeNumerical(DatasetTransform):
    """
    Encode numerical features according to config

    """

    def __init__(self, config):
        self.categorical_columns = config.categorical_columns
        self.numerical_columns = config.numerical_columns
        self.sort_columns = config.sort_columns
        self.skip_columns = config.skip_columns
        self.numerical_types = (
            "float",
            "double",
            "decimal",
            "int",
            "bigint",
            "smallint",
            "tinyint",
        )
        self.classification = config.classification
        self.target_column = config.target_column

    def guess_numerical_columns(self, dataset):
        not_num_cols = (
            self.skip_columns
            + self.sort_columns
            + self.categorical_columns
            + [
                f"_sort_{col}"
                for col in self.sort_columns
                if col in self.categorical_columns
            ]
        )

        numerical_cols = filter(lambda c: c not in not_num_cols, dataset.columns)
        dtypes = dict(dataset.dtypes)
        numerical_cols = filter(
            lambda c: dtypes[c] in self.numerical_types, numerical_cols
        )
        self.numerical_columns.extend(numerical_cols)

    def fit(self, dataset):
        if len(self.numerical_columns) < 1:
            self.guess_numerical_columns(dataset)
        return self

    def transform(self, dataset):
        numerical_columns = self.numerical_columns
        if not self.classification and self.target_column in dataset.columns:
            numerical_columns = numerical_columns + [self.target_column]
        for col in numerical_columns:
            dataset = dataset.fillna({col: 0})
            dataset = dataset.withColumn(col, fs.col(col).cast("float"))
        return dataset


class EncodeIndex(DatasetTransform):
    """
    Encode index feature according to config

    """

    def __init__(self, config):
        self.index_columns = config.index_columns
        self.index_name = config.export_index_name
        self.index_path = os.path.join(config.save_path, self.index_name)
        self.index_parititon_num = 80

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        pass

    def __call__(self, dataset_meta: Dataset):
        dataset = dataset_meta.data
        encoder = LabelEncoder(
            self.index_columns,
            self.index_name,
            path=self.index_path + f"_{dataset_meta.dtype}",
            partitions_num=self.index_parititon_num,
            broadcast=False,
        )
        encoder = encoder.fit(dataset)
        dataset = encoder.transform(dataset)
        dataset = dataset.drop(*self.index_columns)
        dataset_meta.data = dataset
        return dataset_meta


class ToSequenceDataset(DatasetTransform):
    """
    Make SequenceDataset

    """

    def __init__(self, config):
        self.index_column = config.export_index_name
        self.sort_columns = config.sort_columns
        self.categorical_columns = config.categorical_columns
        self.numerical_columns = config.numerical_columns
        self.target_column = config.target_column
        self.save_path = config.save_path
        self.approx_row_size = len(config.categorical_columns) + 4 * (
            len(config.numerical_columns) + 1
        )
        self.min_partition_num = config.min_partition_num
        self.partition_size_mbytes = config.partition_size_mbytes

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        pass

    def get_optimal_partition_num(self, dataset):
        mbyte = 2**20
        partition_num = (dataset.count() * self.approx_row_size) // (
            self.partition_size_mbytes * mbyte
        )
        partition_num = max(partition_num, self.min_partition_num)
        return partition_num

    def _scale_time(self, df, sort_columns):
        """
        Scales time feature in [0,1]

        :param df: dataset
        :param sort_columns: time column, must be single
        :return: dataset with scaled time
        """
        sort_column = sort_columns[0]
        assembler = VectorAssembler(
            inputCols=[sort_column], outputCol=f"{sort_column}_vec"
        )
        scaler = MinMaxScaler(
            inputCol=f"{sort_column}_vec", outputCol=f"{sort_column}_scaled"
        )
        pipeline = Pipeline(stages=[assembler, scaler])
        scalerModel = pipeline.fit(df)
        df = scalerModel.transform(df)

        df = df.drop(sort_column, f"{sort_column}_vec")
        firstelement = fs.udf(lambda v: float(v[0]), FloatType())
        df = df.withColumn(sort_column, firstelement(f"{sort_column}_scaled"))
        df = df.drop(f"{sort_column}_scaled")
        return df

    def __call__(self, dataset_meta: Dataset):
        dataset = dataset_meta.data
        partition_num = self.get_optimal_partition_num(dataset)
        dataset_meta.partition_num = partition_num
        dataset = dataset.repartition(partition_num, self.index_column)
        sort_columns = []
        for col in self.sort_columns:
            if col in self.categorical_columns:
                sort_columns.append(f"_sort_{col}")
            else:
                sort_columns.append(col)
        dataset = dataset.sortWithinPartitions(self.index_column, *sort_columns)
        columns_to_select = (
            [self.index_column]
            + self.numerical_columns
            + self.categorical_columns
            + sort_columns
        )

        if self.target_column in dataset.columns:
            columns_to_select.append(self.target_column)
        dataset = dataset.select(*columns_to_select)
        dataset = self._scale_time(dataset, sort_columns)
        dataset_path = os.path.join(self.save_path, dataset_meta.dtype)
        dataset.write.parquet(dataset_path)
        spark = SparkSession.builder.getOrCreate()
        dataset = spark.read.parquet(dataset_path)
        dataset_meta.data = dataset
        return dataset_meta
