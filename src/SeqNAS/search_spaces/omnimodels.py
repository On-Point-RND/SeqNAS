import torch
import numpy as np
from .multi_trail_model import CountReLU

from .tools import (
    _replace_module_with_type,
    _sample_random,
    _set_final_and_clean,
    _get_a_trace,
    _get_arch_weights,
    _set_weights,
)


def omnimodel(type_names_tuple=[]):
    """
    Decorator for nn.Module to make it searchable

    Args:
        type_names_tuple ([(Fake class, Real class), ...]): list of pairs: Fake class to initialization
                                                            and Real class to realize search
    """

    def set_final_and_clean(self):
        """
        Remove all unused modules from supernet after search
        """
        self.can_sample = False
        print(
            "Final architechure is going to be set, sampling is NOT posible after that"
        )
        _set_final_and_clean(self, type_names_tuple)

    def run_replacement(self, verbose=False):
        """
        Change Fake class to Real class
        """
        if self.can_sample:
            for basic, sub in type_names_tuple:
                _replace_module_with_type(self, sub, basic, verbose)

    def sample_random(self):
        """
        Recursive sample random architecture from supernet
        """
        if self.can_sample:
            for basic, sub in type_names_tuple:
                _sample_random(self, sub)
                run_replacement(self)

        if hasattr(self, "__init_weights__"):
            self.__init_weights__()

    def get_arch_weights(self):
        """
        Return random architecture as the dict wich can be use to build this architecture
        """
        # if self.can_sample:
        #     traced = _get_a_trace(self, (t[1] for t in type_names_tuple))
        #     modules = dict(traced.named_modules())
        #     weights = _get_arch_weights(traced, modules)
        #     return weights

        weights = {}
        if self.can_sample:
            for t in type_names_tuple:
                traced = _get_a_trace(self, (t[1],))
                modules = dict(traced.named_modules())
                w = _get_arch_weights(traced, modules)
                weights.update(w)
            return weights

        else:
            print('Searchable objects were deleted with "set_final_and_clean" call')
            return {"empty": 0}

    def set_arch_weights(self, weights):
        """
        Build architecture from weights dict
        """
        if self.can_sample:
            for t in type_names_tuple:
                traced = _get_a_trace(self, (t[1],))
                modules = dict(traced.named_modules())
                _set_weights(modules, weights)
                run_replacement(self)
        else:
            print('Searchable objects were deleted with "set_final_and_clean" call')

    def reset_weights(model):
        """
        Use to recursively reset weights in the model
        """

        @torch.no_grad()
        def apply(m):
            for name, child in m.named_children():
                if isinstance(child, CountReLU):
                    child.stats = None
                if hasattr(child, "_parameters"):
                    for param_name in child._parameters:
                        # print(name, param_name)
                        if child._parameters[param_name] is not None:
                            if len(child._parameters[param_name].shape) < 2:
                                torch.nn.init.normal_(
                                    child._parameters[param_name].data
                                )
                            else:
                                torch.nn.init.xavier_uniform_(
                                    child._parameters[param_name].data
                                )
                reset_parameters = getattr(child, "reset_parameters", None)
                if callable(reset_parameters):
                    child.reset_parameters()
                else:
                    apply(child)

        apply(model)
        if hasattr(model, "__init_weights__"):
            model.__init_weights__()

    def wrapper(f):
        setattr(f, "can_sample", True)
        setattr(f, "sample_random", sample_random)
        setattr(f, "run_replacement", run_replacement)
        setattr(f, "set_final_and_clean", set_final_and_clean)

        setattr(f, "reset_weights", reset_weights)
        setattr(f, "get_arch", get_arch_weights)
        setattr(f, "set_arch", set_arch_weights)
        return f

    return wrapper
