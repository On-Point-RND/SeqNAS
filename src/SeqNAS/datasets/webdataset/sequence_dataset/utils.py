import os
import pyarrow.parquet
import pyarrow.csv
import numpy as np
import torch
import time
import random
from collections import namedtuple
from typing import Any, List

from webdataset import reraise_exception
from .pipeline import pipelinefilter, pick
from ...timeseries_dataset import _pad_tensor_valid
from ....utils.distribute import pytorch_worker_info

Sequence = namedtuple("Sequence", ["index", "features", "target", "time"])


def read_csv(source, columns=None):
    dataframe = pyarrow.csv.read_csv(source).to_pandas()
    if columns is not None:
        dataframe = dataframe.loc[:, columns]
    return dataframe


def read_parquet(source, columns=None):
    dataframe = pyarrow.parquet.read_table(
        source, columns=columns, use_legacy_dataset=True
    ).to_pandas()
    return dataframe


def read_unknown(source, *args, **kw):
    raise ValueError(f"{source}: unknown file type")


table_readers = dict(__default__=read_unknown, csv=read_csv, parquet=read_parquet)


def read_table(shards, file_type="parquet", columns=None, handler=reraise_exception):
    table_reader = table_readers.get(file_type, table_readers["__default__"])
    for shard in shards:
        try:
            assert isinstance(shard, dict)
            if not os.path.exists(shard["data"]):
                continue
            dataframe = table_reader(shard["data"], columns=columns)
            if dataframe.shape[0] < 1:
                continue
            dataframe = {k: v.values for (k, v) in dataframe.items()}
            yield dataframe
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def get_bounds_for_increasing_index(index):
    inequality_mask = index[1:] != index[:-1]
    bounds = np.arange(1, index.size)[inequality_mask].copy()
    bounds = zip(np.append(0, bounds), np.append(bounds, index.size))
    return bounds


def _group_series(
    dataframes: Any,
    index_column: str,
    target_column: str,
    sort_columns: List[str],
    handler: Any = reraise_exception,
) -> Any:
    """
    Function that groups sequences and yields results to SequenceCollator

    :param dataframes: dataframes
    :param index_column: index column
    :param target_column: target column
    :param sort_columns: time columns. Must be single
    :param handler:
    :return yields seq_ids, seq_features, seq_labels, seq_time
    """
    for df in dataframes:
        try:
            assert len(sort_columns) == 1, "Currenctly supports only single sort column"
            sort_column = sort_columns[0]
            index = df.pop(index_column)
            time = df.pop(sort_column)
            # if target_column is None(test dataset) then return None
            if target_column is not None:
                target = df.pop(target_column)
                for start, end in get_bounds_for_increasing_index(index):
                    sample = {k: v[start:end].copy() for (k, v) in df.items()}
                    yield Sequence(index[start], sample, target[start], time[start:end])
            else:
                for start, end in get_bounds_for_increasing_index(index):
                    sample = {k: v[start:end].copy() for (k, v) in df.items()}
                    yield Sequence(index[start], sample, None, time[start:end])
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


group_series = pipelinefilter(_group_series)


def _shuffle(data, bufsize=1000, initial=100, rng=None, handler=None):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator
    rng: either random module or random.Random instance

    """
    if rng is None:
        UINT_MAX = 0xFFFFFFFF
        # for reproducibility with global seed
        rng = random.Random(random.randint(0, UINT_MAX))
    initial = min(initial, bufsize)
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) < bufsize:
            try:
                buf.append(next(data))  # skipcq: PYL-R1708
            except StopIteration:
                pass
        if len(buf) >= initial:
            yield pick(buf, rng)
    while len(buf) > 0:
        yield pick(buf, rng)


shuffle = pipelinefilter(_shuffle)


def get_feature_tensor(features_dict):
    return torch.FloatTensor(np.vstack(list(features_dict.values())).T)


class SequenceCollator:
    """
    Class that forms batches of data

    """

    def __init__(
        self,
        batch_size,
        classification,
        last_action="drop",
        cat_features=[],
        pad_tail=True,
    ):
        """
        Init collator

        :param batch_size: batch size
        :param classification: True if classification task, False - regression
        :param last_action: [drop, pad] for last batch (when last batch isn't full)
        :param cat_features: categorial features
        :param pad_tail: add zeros to tail of sequence
        """
        self.cat_features = cat_features
        self.batch_size = batch_size
        self.last_action = last_action
        self.pad_tail = pad_tail
        self.classification = classification

    def __call__(self, batch):
        """
        Function that preprocess batch

        :param batch: batch from _group_series function above
        :return: batch in dict format
        """
        seq_ids, seq_features, seq_labels, seq_time = zip(*batch)

        seq_ids = torch.tensor(seq_ids)
        seq_time = [torch.FloatTensor(val) for val in seq_time]

        packed_seq_time = torch.nn.utils.rnn.pack_sequence(seq_time, False)
        padded_seq_time = torch.nn.utils.rnn.pad_packed_sequence(
            packed_seq_time, padding_value=0.0 if self.pad_tail else 1.0
        )
        padded_seq_time = torch.transpose(padded_seq_time[0], 0, 1)

        # if there is no target in data seq_labels are None
        target = None
        if seq_labels[0] is not None:
            target = torch.tensor(seq_labels)

        features = []
        features_names = seq_features[0].keys()

        for i in range(len(seq_ids)):
            # Considering there are no null values in features (all feature values arrays are the same length)
            features.append(get_feature_tensor(seq_features[i]))

        packed = torch.nn.utils.rnn.pack_sequence(features, False)
        padded = torch.nn.utils.rnn.pad_packed_sequence(packed)

        trans_features = dict(zip(features_names, torch.transpose(padded[0], 0, 2)))

        for k in trans_features.keys():
            if k in self.cat_features:
                trans_features[k] = trans_features[k].long()

        actual_bs = len(seq_ids)
        if actual_bs != self.batch_size and self.last_action == "pad":
            if target is not None:
                target = _pad_tensor_valid(
                    target,
                    self.batch_size,
                )
            padded_seq_time = _pad_tensor_valid(padded_seq_time, self.batch_size)
            for k in trans_features.keys():
                trans_features[k] = _pad_tensor_valid(
                    trans_features[k], self.batch_size
                )

            missing_len = self.batch_size - actual_bs
            seq_ids = torch.cat((seq_ids, torch.full((missing_len,), -1)))

        if target is not None:
            if self.classification:
                target = target.long()
            else:
                target = target.float()

        return {
            "model_input": trans_features,
            "target": target,
            "index": seq_ids,
            "time": padded_seq_time,
        }


def make_seed(*args):
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


def pytorch_worker_seed(group=None):
    """Compute a distinct, deterministic RNG seed for each worker and node."""
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)
    return rank * 1000 + worker
