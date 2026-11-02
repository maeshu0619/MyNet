"""Worker-local exact symbol preparation; native model/coder stay unchanged.

The CPU membership specialization is restricted to nonnegative integer
coordinates whose native mixed-radix keys cannot overflow. Unsupported
inputs delegate to the original function. Caches live for one encode only.
"""
from collections import OrderedDict
from types import SimpleNamespace
import types

import numpy as np


class ExactCPUMembership:
    def __init__(self, reference, maximum_entries=4, verify=False):
        self.reference = reference
        self.maximum_entries = maximum_entries
        self.verify = verify
        self.clear()

    def clear(self):
        self.cache = OrderedDict()
        self.counts = dict(calls=0, fast=0, fallback=0, hits=0)

    def __call__(self, data, ground_truth):
        import torch
        self.counts['calls'] += 1
        if (not torch.is_tensor(data) or not torch.is_tensor(ground_truth)
                or data.device.type != 'cpu' or ground_truth.device.type != 'cpu'
                or data.dtype not in (torch.int32, torch.int64)
                or ground_truth.dtype not in (torch.int32, torch.int64)
                or data.ndim != 2 or ground_truth.ndim != 2
                or data.shape[1] != ground_truth.shape[1] or not 1 <= data.shape[1] <= 4
                or not data.numel() or not ground_truth.numel()):
            self.counts['fallback'] += 1
            return self.reference(data, ground_truth)
        a, b = data.detach().numpy(), ground_truth.detach().numpy()
        # This conservative bound ensures base**D < 2**63, including D=4.
        if a.min() < 0 or b.min() < 0 or a.max() > 32767 or b.max() > 32767:
            self.counts['fallback'] += 1
            return self.reference(data, ground_truth)
        try:
            version = ground_truth._version
        except RuntimeError:
            version = None
        token = (id(ground_truth), version)
        cached = self.cache.get(token) if version is not None else None
        if cached is None:
            limit = int(b.max())
            base = limit + 1
            keys = self._keys(b, base)
            keys = np.unique(keys)
            cached = (ground_truth, limit, base, keys)
            if version is not None:
                self.cache[token] = cached
                while len(self.cache) > self.maximum_entries:
                    self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(token)
            self.counts['hits'] += 1
        _, limit, base, keys = cached
        inside = (a <= limit).all(axis=1)
        result = np.zeros(len(a), dtype=bool)
        query = self._keys(a[inside], base)
        location = np.searchsorted(keys, query)
        valid = location < len(keys)
        result[inside] = valid & (keys[np.minimum(location, len(keys)-1)] == query)
        answer = torch.from_numpy(result)
        self.counts['fast'] += 1
        if self.verify and not torch.equal(answer, self.reference(data, ground_truth)):
            raise AssertionError('Native membership mask differs')
        return answer

    @staticmethod
    def _keys(rows, base):
        result = np.zeros(len(rows), dtype=np.int64)
        multiplier = 1
        for axis in range(rows.shape[1]):
            result += rows[:, axis].astype(np.int64, copy=False)*multiplier
            multiplier *= base
        return result


class ExactTensorMembership:
    """Native integer keys + cached GT sorting, on the original device.

    Uses the native key function itself, including its integer arithmetic.
    Tensor identity/version and the actual native radix identify each entry.
    """
    def __init__(self, reference, array2vector, maximum_entries=4, verify=False):
        self.reference, self.vector = reference, array2vector
        self.maximum_entries, self.verify = maximum_entries, verify
        self.clear()

    def clear(self):
        self.cache = OrderedDict()
        self.maxima = OrderedDict()
        self.counts = dict(calls=0, fast=0, fallback=0, hits=0)

    def __call__(self, data, ground_truth):
        import torch
        self.counts['calls'] += 1
        if (not torch.is_tensor(data) or not torch.is_tensor(ground_truth)
                or data.device != ground_truth.device or data.ndim != 2
                or ground_truth.ndim != 2 or data.shape[1] != ground_truth.shape[1]
                or data.dtype not in (torch.int32,torch.int64)
                or ground_truth.dtype not in (torch.int32,torch.int64)
                or not data.numel() or not ground_truth.numel()):
            self.counts['fallback'] += 1
            return self.reference(data,ground_truth)
        try:version=ground_truth._version
        except RuntimeError:version=None
        identity=(id(ground_truth),version)
        maximum=self.maxima.get(identity) if version is not None else None
        if maximum is None:
            maximum=(ground_truth,ground_truth.max())
            if version is not None:
                self.maxima[identity]=maximum
                while len(self.maxima)>self.maximum_entries:self.maxima.popitem(last=False)
        else:self.maxima.move_to_end(identity)
        step=torch.max(data.max(),maximum[1])+1
        token=(identity,int(step.item()))
        cached=self.cache.get(token) if version is not None else None
        if cached is None:
            cached=(ground_truth,torch.sort(self.vector(ground_truth,step)).values)
            if version is not None:
                self.cache[token]=cached
                while len(self.cache)>self.maximum_entries:self.cache.popitem(last=False)
        else:
            self.counts['hits']+=1;self.cache.move_to_end(token)
        keys=cached[1]
        query=self.vector(data,step)
        location=torch.searchsorted(keys,query)
        answer=(location<len(keys)) & (keys[location.clamp_max(len(keys)-1)]==query)
        self.counts['fast']+=1
        if self.verify and not torch.equal(answer,self.reference(data,ground_truth)):
            raise AssertionError('Native tensor membership mask differs')
        return answer


def sorted_symbol_view(tensor, array2vector):
    """Sort the same rows, avoiding a new ME coordinate map and CPU F copy.

    Valid only for the final occupancy classifier outputs consumed by encode;
    this proxy must never be passed to a neural layer or a native decoder.
    """
    import torch
    indices = torch.argsort(array2vector(tensor.C, tensor.C.max()+1))
    return SimpleNamespace(C=tensor.C.index_select(0, indices),
        F=tensor.F.index_select(0, indices.to(tensor.F.device)))


def install_symbol_preparation(encoder, *, membership=True, sorting=False,
                               verify=False, debug_decode=True):
    """Install on this encoder only; restore module membership after requests."""
    import torch
    basic = encoder.coder.basic_coder
    upsampler = basic.model.upsampler
    original = upsampler.encode
    body = getattr(original, '__wrapped__', None)
    if body is None or not isinstance(body, types.FunctionType):
        raise TypeError('Unsupported native upsampler.encode wrapper')
    namespace = body.__globals__
    native_membership = namespace['isin']
    native_sort = namespace['sort_sparse_tensor']
    from data_utils.sparse_tensor import array2vector
    cache = (ExactTensorMembership(native_membership,array2vector,verify=verify)
             if membership=='gpu' else ExactCPUMembership(native_membership, verify=verify))
    local = dict(namespace)
    if membership:
        local['isin'] = cache
    if sorting:
        def sort(value):
            answer = sorted_symbol_view(value, array2vector)
            if verify:
                expected = native_sort(value)
                if not torch.equal(answer.C, expected.C) or not torch.equal(answer.F, expected.F):
                    raise AssertionError('Sorted native symbol order/features differ')
                # Keep GPU sigmoid exactly where the native coder evaluates it.
                if not torch.equal(torch.sigmoid(answer.F), torch.sigmoid(expected.F)):
                    raise AssertionError('Native occupancy probabilities differ')
            return answer
        local['sort_sparse_tensor'] = sort
    cloned = types.FunctionType(body.__code__, local, body.__name__,
                                body.__defaults__, body.__closure__)
    cloned.__kwdefaults__ = body.__kwdefaults__
    wrapped = torch.no_grad()(cloned)
    upsampler.encode = types.MethodType(wrapped, upsampler)
    encode_one = encoder.encode_one
    module = __import__(namespace['__name__'], fromlist=['isin'])

    def encode(*args, **kwargs):
        cache.clear()
        saved = module.isin
        if membership:
            module.isin = cache
        try:
            result = encode_one(*args, **kwargs)
            result['native_symbol_preparation_counts'] = dict(cache.counts)
            return result
        finally:
            module.isin = saved
            cache.cache.clear()
            if hasattr(cache,'maxima'):cache.maxima.clear()
    if not debug_decode:
        # Only redundant arithmetic self-checks are omitted; bytes are made
        # by the same native BAC.encode and still independently decode-tested.
        cloned.__defaults__ = (False,)
    encoder.encode_one = encode
    return dict(membership_cache=cache, original_encode=original, original_encode_one=encode_one)
