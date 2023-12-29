from torch.utils.data import DataLoader, IterableDataset
import copy, sys
from itertools import islice


def add_length_method(obj):
    def length(self):
        return self.size

    Combined = type(
        obj.__class__.__name__ + "_Length",
        (obj.__class__, IterableDataset),
        {"__len__": length},
    )
    obj.__class__ = Combined
    return obj


class PipelineStage:
    def invoke(self, *args, **kw):
        raise NotImplementedError


class DataPipeline(IterableDataset, PipelineStage):
    """A pipeline starting with an IterableDataset and a series of filters."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline = []
        self.length = 512
        self.repetitions = 1
        self.nsamples = -1
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                self.pipeline.extend(arg)
            else:
                self.pipeline.append(arg)

    def invoke(self, f, *args, **kwargs):
        """Apply a pipeline stage, possibly to the output of a previous stage."""
        if isinstance(f, PipelineStage):
            return f.run(*args, **kwargs)
        if isinstance(f, (IterableDataset, DataLoader)) and len(args) == 0:
            return iter(f)
        if isinstance(f, list):
            return iter(f)
        if callable(f):
            result = f(*args, **kwargs)
            return result
        raise ValueError(f"{f}: not a valid pipeline stage")

    def iterator1(self):
        """Create an iterator through one epoch in the pipeline."""
        source = self.invoke(self.pipeline[0])
        for step in self.pipeline[1:]:
            source = self.invoke(step, source)
        return source

    def iterator(self):
        """Create an iterator through the entire dataset, using the given number of repetitions."""
        for i in range(self.repetitions):
            for sample in self.iterator1():
                yield sample

    def __iter__(self):
        """Create an iterator through the pipeline, repeating and slicing as requested."""
        if self.repetitions != 1:
            if self.nsamples > 0:
                return islice(self.iterator(), self.nsamples)
            else:
                return self.iterator()
        else:
            return self.iterator()

    def stage(self, i):
        """Return pipeline stage i."""
        return self.pipeline[i]

    def append(self, f):
        """Append a pipeline stage (modifies the object)."""
        self.pipeline.append(f)

    def compose(self, *args):
        """Append a pipeline stage to a copy of the pipeline and returns the copy."""
        result = copy.copy(self)
        for arg in args:
            result.append(arg)
        return result

    def __len__(self):
        return self.length

    def with_length(self, n):
        """Add a __len__ method returning the desired value.

        This does not change the actual number of samples in an epoch.
        PyTorch IterableDataset should not have a __len__ method.
        This is provided only as a workaround for some broken training environments
        that require a __len__ method.
        """
        self.length = n
        return self

    def with_epoch(self, nsamples=-1, nbatches=-1):
        """Change the epoch to return the given number of samples/batches.

        The two arguments mean the same thing."""
        self.repetitions = sys.maxsize
        self.nsamples = max(nsamples, nbatches)
        return self

    def repeat(self, nepochs=-1, nbatches=-1):
        """Repeat iterating through the dataset for the given #epochs up to the given #samples."""
        if nepochs > 0:
            self.repetitions = nepochs
            self.nsamples = nbatches
        else:
            self.repetitions = sys.maxsize
            self.nsamples = nbatches
        return self


class FilterFunction(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct becauce it can be pickled.
    """

    def __init__(self, f, *args, **kw):
        """Create a curried function."""
        self.f = f
        self.args = args
        self.kw = kw

    def __call__(self, data):
        """Call the curried function with the given argument."""
        return self.f(data, *self.args, **self.kw)

    def __str__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"

    def __repr__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"


class RestCurried(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct because it can be pickled.
    """

    def __init__(self, f):
        """Store the function for future currying."""
        self.f = f

    def __call__(self, *args, **kw):
        """Curry with the given arguments."""
        return FilterFunction(self.f, *args, **kw)


def pipelinefilter(f):
    """Turn the decorated function into one that is partially applied for
    all arguments other than the first."""
    result = RestCurried(f)
    return result


def pick(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample
