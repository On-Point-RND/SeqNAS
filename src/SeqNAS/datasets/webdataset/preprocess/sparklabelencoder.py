import logging as log
from contextlib import contextmanager

from pyspark.sql import SparkSession, Row, functions as fs
from pyspark.sql.types import StructField, LongType


@contextmanager
def cache(df):
    df.cache()
    try:
        yield df
    finally:
        df.unpersist()


def add_row_id_to_dataframe(df, column_name, start_value=0):
    spark = SparkSession.builder.getOrCreate()
    starts = [start_value]
    if df.rdd.getNumPartitions() > 1:
        nums = df.rdd.mapPartitions(lambda it: [sum(1 for i in it)]).collect()
        for i in range(len(nums) - 1):
            starts.append(starts[-1] + nums[i])

    RowWithId = Row(*df.columns, column_name)

    def func(k, it):
        for i, v in enumerate(it, starts[k]):
            yield RowWithId(*v, i)

    df = spark.createDataFrame(
        df.rdd.mapPartitionsWithIndex(func, preservesPartitioning=True),
        df.schema.add(StructField(column_name, LongType(), True)),
    )
    return df


class LabelEncoder:
    def __init__(
        self,
        input_cols,
        output_col,
        path=None,
        broadcast=False,
        fillna_value=None,
        partitions_num=None,
        start_value=0,
    ):
        if not isinstance(input_cols, (tuple, list)):
            input_cols = [input_cols]
        self.input_cols = input_cols
        self.output_col = output_col
        self.path = path
        self.broadcast = broadcast
        self.fillna_value = fillna_value
        self.partitions_num = partitions_num
        self.encoder_table = None
        self.start_value = start_value

    def fillna(self, df):
        df = df.fillna({c: self.fillna_value for c in self.input_cols})
        return df

    def fit(self, df):
        raw_column = df.select(*self.input_cols)
        if self.fillna_value is not None:
            raw_column = self.fillna(raw_column)
        unique_values = raw_column.distinct()
        if self.partitions_num is not None:
            unique_values = unique_values.repartition(self.partitions_num)
        with cache(unique_values):
            encoded_table = add_row_id_to_dataframe(
                unique_values, self.output_col, self.start_value
            )
            encoded_table.write.parquet(self.path)
        return self

    def count(self):
        spark = SparkSession.builder.getOrCreate()
        encoded_table = spark.read.parquet(self.path)
        return encoded_table.count()

    def transform(self, df):
        spark = SparkSession.builder.getOrCreate()
        encoded_table = spark.read.parquet(self.path)
        if self.fillna_value is not None:
            df = self.fillna(df)
        if self.broadcast:
            encoded_table = fs.broadcast(encoded_table)
        df = df.join(encoded_table, self.input_cols, "left")
        if self.fillna_value is not None:
            df = df.replace(self.fillna_value, None, self.input_cols)
        return df
