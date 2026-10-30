import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import numpy as np

from models.utils.loss.autonomous_native_stats import quantized_morton_stats, make_pointcloud_stats, install_pointcloud_stats


class NativeStatsParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[2]/'compress/octree/SparsePCGC/encoder_multiple.py'
        if not source.is_file():
            raise unittest.SkipTest('Native SparsePCGC reference source unavailable')
        spec = importlib.util.spec_from_file_location('_test_native_stats_reference', str(source))
        cls.native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.native
        spec.loader.exec_module(cls.native)

    def reference(self, points, qs=1., pos=1):
        coords = self.native._quantized_unique_points(points, qs, pos)
        node, single = self.native._octree_occupancy_stats(coords)
        return len(coords), node, single

    def test_all_leaf_masks_and_translations(self):
        children = np.asarray([(x, y, z) for x in range(2) for y in range(2) for z in range(2)])
        for mask in range(1, 256):
            points = children[[bool(mask & (1 << bit)) for bit in range(8)]]
            for translation in ((0, 0, 0), (-7, 51, -99)):
                value = points+translation
                self.assertEqual(quantized_morton_stats(value), self.reference(value))

    def test_duplicates_negatives_noninteger_two_rounds_and_depths(self):
        rng = np.random.RandomState(190)
        for seed in range(20):
            points = rng.randint(-511, 512, (200, 3))/4.
            points = np.r_[points, points[:20]]
            for qs, pos in ((1., 1), (.5, 2), (3., 3), (.2, 4)):
                self.assertEqual(quantized_morton_stats(points, qs, pos), self.reference(points, qs, pos))
        for points in (np.empty((0, 3)), np.zeros((5, 3)),
                       np.array([[0., 0., 0.], [2**21-1., 5., 1.]]),
                       np.array([[-1.5, -.5, .5], [1.5, 2.5, 3.5]])):
            self.assertEqual(quantized_morton_stats(points), self.reference(points))

    def test_unsupported_ranges_and_nonfinite_fall_back(self):
        for points in (np.array([[0., 0., 0.], [2**21, 0., 0.]]),
                       np.full((1, 3), np.nan), np.full((1, 3), 2.**63)):
            self.assertIsNone(quantized_morton_stats(points))
            reference = mock.Mock(return_value=dict(native=True))
            wrapped = make_pointcloud_stats(lambda path: (points, len(points)), reference)
            self.assertEqual(wrapped('input', 1., 1), dict(native=True))
            reference.assert_called_once_with('input', 1., 1)
        with self.assertRaises(ValueError):
            quantized_morton_stats(np.zeros((1, 3)), 0.)

    def test_wrapper_retains_header_count_and_exact_metrics(self):
        points = np.array([[2., 1., 1.], [2., 1., 1.], [-1., 3., 5.]])
        wrapped = make_pointcloud_stats(lambda path: (points, 99), mock.Mock(side_effect=AssertionError))
        result = wrapped('unchanged reader', 1., 1)
        self.assertEqual(result['header_point_count'], 99)
        self.assertEqual(tuple(result[k] for k in ('point_count', 'node_count', 'single_child_count')),
                         self.reference(points))
        self.assertGreaterEqual(result['stats_time'], 0.)

    def test_installation_is_local_idempotent_and_missing_api_keeps_native(self):
        original = mock.Mock()
        module = SimpleNamespace(_read_ply_points=lambda path: (np.zeros((1, 3)), 1), _pointcloud_stats=original)
        self.assertTrue(install_pointcloud_stats(module))
        installed = module._pointcloud_stats
        self.assertTrue(install_pointcloud_stats(module))
        self.assertIs(installed, module._pointcloud_stats)
        self.assertEqual(installed('x', 1., 1)['point_count'], 1)
        original.assert_not_called()
        self.assertFalse(install_pointcloud_stats(SimpleNamespace()))

    def test_worker_flags_are_true_dense_lossy_only(self):
        from models.utils.loss.actual_encoder import _SparsePCGCActualEncoder
        args = SimpleNamespace(test_route=True, compress='SparsePCGC',
            heuristic_guidance_mode='ana_den6_online', sparsepcgc_mode='dense_lossy')
        encoder = _SparsePCGCActualEncoder(args)
        command, _ = encoder._worker_args()
        self.assertIn('--exact-morton-stats', command)
        self.assertIn('--exact-unique-loader', command)
        args.test_route = False
        command, _ = encoder._worker_args()
        self.assertNotIn('--exact-morton-stats', command)
        self.assertNotIn('--exact-unique-loader', command)
        args.test_route = True
        args.sparsepcgc_skip_decode = False
        command, _ = encoder._worker_args()
        self.assertNotIn('--exact-morton-stats', command)
        self.assertNotIn('--exact-unique-loader', command)
        args.sparsepcgc_skip_decode = True
        args.sparsepcgc_mode = 'dense_lossless'
        command, _ = encoder._worker_args()
        self.assertNotIn('--exact-morton-stats', command)
        self.assertNotIn('--exact-unique-loader', command)


if __name__ == '__main__':
    unittest.main()
