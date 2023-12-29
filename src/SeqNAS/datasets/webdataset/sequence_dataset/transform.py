from abc import ABC, abstractmethod
from typing import List, Callable, Any, Dict, Optional, Union, Tuple
from webdataset import reraise_exception

from .utils import Sequence
import random


class Transform(ABC):
    @abstractmethod
    def transform(self, seq: Sequence) -> Optional[Sequence]:
        pass

    def __call__(self, seqs: List[Sequence]):
        for seq in seqs:
            seq = self.transform(seq)
            if seq is not None:
                yield seq


class RandomSubseqTransform(Transform):
    def __init__(self, length: Union[int, Tuple[int, int]]):
        if isinstance(length, (list, tuple)):
            self.min_length = min(length)
            self.max_length = max(length)
        else:
            self.min_length = self.max_length = length

    def transform(self, seq: Sequence) -> Optional[Sequence]:
        seq_length = next(iter(seq.features.values())).shape[0]
        subseq_length = min(
            seq_length, random.randint(self.min_length, self.max_length)
        )
        start_pos = random.randint(0, seq_length - subseq_length)
        end_pos = start_pos + subseq_length
        features = {
            name: val[start_pos:end_pos].copy() for name, val in seq.features.items()
        }
        return seq._replace(features=features)


class TailSubseqTransform(Transform):
    def __init__(self, length: int):
        self.length = length

    def transform(self, seq: Sequence) -> Optional[Sequence]:
        seq_length = next(iter(seq.features.values())).shape[0]
        start_pos = max(0, seq_length - self.length)
        features = {name: val[start_pos:].copy() for name, val in seq.features.items()}
        return seq._replace(features=features)
