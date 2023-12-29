import importlib
import os

DATASET_REGISTRY = {}


def register_dataset(name):
    """Decorator to register a new dataset (e.g., TimeSeriesInMemoryDataset)."""

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


_dataset_dir_name = os.path.dirname(__file__)
# automatically import any Python files in the models/ directory
for file in os.listdir(_dataset_dir_name):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("SeqNAS.datasets." + module)

importlib.import_module("SeqNAS.datasets.webdataset.sequence_dataset.dataset")
