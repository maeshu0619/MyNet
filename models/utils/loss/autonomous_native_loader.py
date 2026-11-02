"""Exact worker-local acceleration of native input preparation.

The native loader's function code, coordinate ordering, quantization result,
sparse collation and Minkowski construction are retained. The worker's own
binary PLY layout has a direct reader with a native fallback. Local functions
also use injective unique keys and bypass identity unit quantization.
"""
import functools
import os
import types

import numpy as np


def make_exact_binary_reader(reference):
    """Read the worker's own XYZ float32 PLY without Open3D allocation.

    Match read_ply_o3d's default int32 conversion *before* quantization.
    Any other header/layout/value range delegates to that original reader.
    No input is cached between requests.
    """
    def read_coords(path):
        if not str(path).endswith('.ply'):
            return reference(path)
        points = None
        try:
            with open(path, 'rb') as stream:
                lines = [stream.readline(256) for _ in range(7)]
                expected = [b'ply\n', b'format binary_little_endian 1.0\n', None,
                            b'property float x\n', b'property float y\n',
                            b'property float z\n', b'end_header\n']
                if all(want is None or got == want for got, want in zip(lines, expected)):
                    words = lines[2].split()
                    if len(words) == 3 and words[:2] == [b'element', b'vertex']:
                        count = int(words[2])
                        size = os.fstat(stream.fileno()).st_size-stream.tell()
                        if count >= 0 and size == count*12:
                            values = np.fromfile(stream, dtype='<f4', count=count*3).reshape(-1, 3)
                            # Native double -> int32 truncation equals float32
                            # -> int32 for finite, representable values only.
                            if (values.size == count*3 and np.isfinite(values).all()
                                    and (not count or (values.min() >= -(2.**31)
                                                       and values.max() < 2.**31))):
                                points = values.astype(np.int32)
        except (OSError, ValueError, OverflowError):
            pass
        return reference(path) if points is None else points
    read_coords._mynet_native_reader = reference
    return read_coords


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
    reader = environment.get('read_coords')
    if callable(reader):
        environment['read_coords'] = make_exact_binary_reader(reader)
    quantize = environment.get('quantize_precision')
    if callable(quantize):
        def exact_identity_quantize(points, precision=.001, quant_mode='round', return_offset=False):
            if (type(points) is np.ndarray and points.dtype.kind in 'iu'
                    and points.dtype.itemsize <= 4 and precision == 1
                    and quant_mode in ('round', 'floor') and not return_offset):
                # Every int32/uint32 is exactly representable in float64.
                # Native float64 division by 1 and round/floor are identity.
                return points.astype('int')
            return quantize(points, precision=precision, quant_mode=quant_mode,
                            return_offset=return_offset)
        environment['quantize_precision'] = exact_identity_quantize
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
