"""
Methods are used in basic omnimodel

"""

import torch.fx

from .basic_ops import LayerChoice, Repeat, Residual, Identity
from ..models.transformer.basic_ops import SinTimeEncoding


def _replace_module_with_type(root_module, init_fn, basic, verbose=False):
    if verbose:
        verb = print
    else:

        def verb(*args, **kwargs):
            pass

    verb("\n###############################")
    verb(f"replacing all {basic} with {init_fn}")
    verb("###############################\n")
    new_dict = root_module._modules.copy()

    def apply(m):
        for name, child in m.named_children():
            verb(f"processing class {child} with name {name}")
            if isinstance(child, basic):
                verb(f"replacing {basic} with {init_fn}")
                f = init_fn(child)
                setattr(m, name, f)
                new_dict[name] = f

            else:
                apply(child)

    apply(root_module)


def replace_layer_choice(root_module, init_fn):
    return _replace_module_with_type(root_module, init_fn, LayerChoice)


def replace_repeat(root_module, init_fn):
    return _replace_module_with_type(root_module, init_fn, Repeat)


def _sample_random(model, basic):
    def apply(m):
        for name, child in m.named_children():
            # print(child)
            if isinstance(child, basic):
                child.sample_path()
            else:
                apply(child)

    apply(model)


def _set_final_and_clean(f, type_names_tuple):
    # if child has parent
    def apply(m):
        for name, child in m.named_children():
            apply(child)
            if any(isinstance(child, sub) for (_, sub) in type_names_tuple):
                final = child._get_final_and_clean()
                setattr(m, name, final)
                del child

            # else:
            #     apply(child)

    apply(f)


def get_searchable_modules(model):
    modules = dict()
    for m in model.named_modules():
        if hasattr(m[1], "_searchable"):
            if len(m[0]) > 0:
                print(m[0])
            modules[m[0]] = m[1]

    return modules


"""
Torch FX allows more flexible graph manipulation

Usage below might be redundant but in general it is a usefull tool.
"""


def _get_a_trace(model, sub_classes: list):
    class Tracer(torch.fx.Tracer):
        def __init__(self, modules=[]):
            super().__init__()
            self.modules = modules

        def is_leaf_module(self, m, module_qualified_name):
            from ..models.transformer.basic_ops import FixedAttention

            return (
                (
                    m.__module__.startswith("torch.nn")
                    and not isinstance(m, torch.nn.Sequential)
                    and not isinstance(m, torch.nn.ModuleList)
                    and not isinstance(m, torch.nn.ModuleDict)
                )
                or (any(isinstance(m, n) for n in self.modules))
                or (isinstance(m, FixedAttention) or isinstance(m, SinTimeEncoding))
                or (m.__module__ == "apex.normalization.fused_layer_norm")
                or (
                    # to work with wrappers in models/transformers/basic_ops.py
                    isinstance(m, torch.nn.GRU)
                    or isinstance(m, torch.nn.Conv1d)
                )
            )

    tracer = Tracer(list(sub_classes))
    graph = tracer.trace(model)
    return torch.fx.GraphModule(model, graph)


def _get_arch_weights(traced, modules) -> dict:
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    weights = dict()
    for node in traced.graph.nodes:
        if node.op == "call_module":
            # The target attribute is the module
            if node.target in modules:
                if hasattr(modules[node.target], "_get_weights"):
                    weights[node.target] = modules[node.target]._get_weights()

    return weights


def _set_weights(modules: dict, weights: dict):
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    keys = set(modules.keys()) & set(weights.keys())
    for key in keys:
        if hasattr(modules[key], "_load_weights"):
            modules[key]._load_weights(weights[key])
