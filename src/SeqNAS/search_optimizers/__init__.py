import importlib
import os

SEARCH_METHOD_REGISTRY = {}


def register_search_method(name):
    """Decorator to register a new search method (e.g., Bananas)."""

    def register_search_method_cls(cls):
        if name in SEARCH_METHOD_REGISTRY:
            raise ValueError(
                "Cannot register duplicate search method ({})".format(name)
            )
        SEARCH_METHOD_REGISTRY[name] = cls
        return cls

    return register_search_method_cls


_seach_method_dir_name = os.path.dirname(__file__)
# automatically import any Python files in the search_optimizers/ directory
for file in os.listdir(_seach_method_dir_name):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("SeqNAS.search_optimizers." + module)

# automatically import any Python files in the models/transformer directory
importlib.import_module("SeqNAS.search_optimizers.bananas.bananas_searcher")
