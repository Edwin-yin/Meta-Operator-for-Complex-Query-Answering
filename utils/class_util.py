import collections
from functools import partial
from itertools import repeat
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def nested_dict(): return collections.defaultdict(nested_dict)


def fixed_depth_nested_dict(default_factory, depth=1):
    result = partial(collections.defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(collections.defaultdict, result)
    return result()


class MyMetaLinear(nn.Linear):
    def forward(self, x, freeze=False, params=None):
        if freeze:
            return F.linear(x, self.weight.data, self.bias.data)
        else:
            if params:
                return F.linear(x, params['weight'], params['bias'])
            else:
                return F.linear(x, self.weight, self.bias)

    def meta_forward(self, x, params):
        return F.linear(x, params['weight'], params['bias'])


def rename_ordered_dict(old_dict, old_key, new_key):
    """
    Create a new OrderedDict for rename a given key in old OrderedDict
    """
    new_dict = collections.OrderedDict((new_key if k == old_key else k, v) for k, v in old_dict.items())
    return new_dict


def compare_torch_dict(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    final_compare_dict = {}
    for key in dict1:
        if isinstance(dict1[key], dict):
            sub_compare = compare_torch_dict(dict1[key], dict2[key])
            final_compare_dict[key] = sub_compare
        elif isinstance(dict1[key], torch.Tensor):
            final_compare_dict[key] = torch.all(dict1[key] == dict2[key])
        else:
            final_compare_dict[key] = (dict1[key] == dict2[key])
    return final_compare_dict


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r'\1', k) for k in all_names if key_re.match(k) is not None]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn('Module `{0}` has no parameter corresponding to the '
                          'submodule named `{1}` in the dictionary `params` '
                          'provided as an argument to `forward()`. Using the '
                          'default parameters for this submodule. The list of '
                          'the parameters in `params`: [{2}].'.format(
                self.__class__.__name__, key, ', '.join(all_names)),
                stacklevel=2)
            return None

        return collections.OrderedDict([(name, params[f'{key}.{name}']) for name in names])
