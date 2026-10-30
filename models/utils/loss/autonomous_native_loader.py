"""Exact worker-local acceleration of a native loader's unique(axis=0).

The native loader's function code, reader, rounding, sparse collation, and
Minkowski construction are retained. Only its local NumPy unique operation
uses a one-dimensional key when injective packing is safe.
"""
import functools
import types

import numpy as np


def lexicographic_unique_rows(array):
    if type(array) is not np.ndarray or array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind != 'i':
        return None
    if not len(array):
        return array.copy()
    coords = array.astype(np.int64, copy=False)
    low, high = coords.min(0), coords.max(0)
    span = [int(hi)-int(lo)+1 for lo, hi in zip(low, high)]
    if span[0]*span[1]*span[2] >= 2**63:
        return None
    shifted = (coords-low).astype(np.uint64)
    keys = ((shifted[:, 0]*np.uint64(span[1])+shifted[:, 1])*np.uint64(span[2])+shifted[:, 2])
    _, first = np.unique(keys, return_index=True)
    return array[first]


class _ExactUniqueNumpy:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def unique(self, array, *args, **kwargs):
        if not args and kwargs == {'axis': 0}:
            answer = lexicographic_unique_rows(array)
            if answer is not None:
                return answer
        return self.original.unique(array, *args, **kwargs)


def make_exact_loader(loader):
    if not isinstance(loader, types.FunctionType) or 'np' not in loader.__globals__:
        return None
    if getattr(loader, '_mynet_exact_unique_loader', False):
        return loader
    environment = dict(loader.__globals__)
    environment['np'] = _ExactUniqueNumpy(environment['np'])
    wrapper = types.FunctionType(loader.__code__, environment, loader.__name__,
                                 loader.__defaults__, loader.__closure__)
    wrapper.__kwdefaults__ = loader.__kwdefaults__
    functools.update_wrapper(wrapper, loader)
    wrapper._mynet_exact_unique_loader = True
    return wrapper


def install_exact_loader(encoder):
    basic = getattr(getattr(encoder, 'coder', None), 'basic_coder', None)
    bound = getattr(basic, 'load_data', None)
    method = getattr(bound, '__func__', bound)
    namespace = getattr(method, '__globals__', None)
    if not isinstance(namespace, dict):
        return False
    wrapped = make_exact_loader(namespace.get('load_sparse_tensor'))
    if wrapped is None:
        return False
    namespace['load_sparse_tensor'] = wrapped
    return True
