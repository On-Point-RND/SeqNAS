import importlib
import os

MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a new model (e.g., LSTM)."""

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


_model_dir_name = os.path.dirname(__file__)
# automatically import any Python files in the models/ directory
for file in os.listdir(_model_dir_name):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("SeqNAS.models." + module)

# automatically import any Python files in the models/transformer directory
importlib.import_module("SeqNAS.models.transformer.searchable_transformer")
importlib.import_module("SeqNAS.models.transformer.flexible_transformer")
importlib.import_module("SeqNAS.models.transformer.transformer_models")
importlib.import_module("SeqNAS.models.simplerick_rnn.searchable_rnn")
