import unittest
import types
from types import SimpleNamespace

import numpy as np

from models.utils.loss.autonomous_native_loader import lexicographic_unique_rows, make_exact_loader, install_exact_loader


def _native_like_loader(points, factor=1.):
    coords = np.round(points/factor).astype('int')
    return np.unique(coords.astype('int'), axis=0).astype('int')


class ExactNativeLoaderTest(unittest.TestCase):
    def test_installation_changes_only_the_worker_coder_namespace(self):
        def caller(self, points):
            return load_sparse_tensor(points)
        environment = dict(caller.__globals__, load_sparse_tensor=_native_like_loader)
        method = types.FunctionType(caller.__code__, environment)
        basic_type = type('BasicFixture', (), dict(load_data=method))
        owner = SimpleNamespace(coder=SimpleNamespace(basic_coder=basic_type()))
        self.assertTrue(install_exact_loader(owner))
        installed = environment['load_sparse_tensor']
        self.assertIs(installed.__code__, _native_like_loader.__code__)
        self.assertTrue(install_exact_loader(owner))
        self.assertIs(environment['load_sparse_tensor'], installed)
        self.assertIs(_native_like_loader.__globals__['np'], np)
        points = np.array([[2., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        self.assertTrue(np.array_equal(owner.coder.basic_coder.load_data(points), _native_like_loader(points)))
        self.assertFalse(install_exact_loader(SimpleNamespace()))

    def test_integer_rows_preserve_native_lex_order_and_dtype(self):
        rng = np.random.RandomState(52)
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            for _ in range(20):
                coords = rng.randint(-100, 100, (300, 3)).astype(dtype)
                coords = np.r_[coords, coords[:30]]
                actual = lexicographic_unique_rows(coords)
                expected = np.unique(coords, axis=0)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(lexicographic_unique_rows(np.empty((0, 3), dtype=np.int64)).shape, (0, 3))

    def test_overflow_and_other_input_types_delegate(self):
        self.assertIsNone(lexicographic_unique_rows(np.array([[-2**63, 0, 0], [2**63-1, 0, 0]], dtype=np.int64)))
        self.assertIsNone(lexicographic_unique_rows(np.zeros((2, 3))))
        self.assertIsNone(lexicographic_unique_rows(np.zeros((2, 4), dtype=np.int64)))

    def test_original_function_code_quantization_and_numpy_are_preserved(self):
        optimized = make_exact_loader(_native_like_loader)
        self.assertIs(optimized.__code__, _native_like_loader.__code__)
        self.assertIs(_native_like_loader.__globals__['np'], np)
        rng = np.random.RandomState(12)
        points = rng.randn(500, 3)*10.
        for factor in (.5, 1., 3.):
            self.assertTrue(np.array_equal(optimized(points, factor), _native_like_loader(points, factor)))
        proxy = optimized.__globals__['np']
        self.assertIs(proxy.round, np.round)
        array = np.array([3, 1, 3])
        for actual, expected in zip(proxy.unique(array, return_index=True), np.unique(array, return_index=True)):
            self.assertTrue(np.array_equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
