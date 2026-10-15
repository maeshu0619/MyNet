import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.utils.config.args import parse_pugan_args
from models.utils.pointcloud.geometry_octree import (
    quantize_geometry_octree, build_geometry_octree_context,
    align_geometry_context, geometry_step,
    geometry_actual_cache_fingerprint, geometry_cache_key,
)
from models.utils.pointcloud.geometry_octree_proposals import attach_geometry_octree_guidance
from models.utils.pointcloud.sparsepcgc_voxel import quantize_sparsepcgc_coords, restore_points_from_voxel_coords
from models.utils.compression.gpcc_tmc3 import GPCCGeometryEncoder
from models.utils.compression.proxy_octree import SoftOctreeRateProxy, ProxyOctreeConfig
from models.modules.geometry_octree_structure import GeometryOctreeStructureAnalysis
from models.modules.structure_actuator import StructureRepairActuator


def args_for(codec):
    return SimpleNamespace(compress=codec, qs=2., gpcc_effective_qs=2., gpcc_prequantize=True,
                           # Deliberately incompatible SparsePCGC flags must be irrelevant.
                           sparsepcgc_voxel_size=17., sparsepcgc_pos_quantscale=4,
                           sparsepcgc_dequantize_center=True, heuristic_guidance_mode='ana_den6_online')


class GeometryCodecRoutesTest(unittest.TestCase):
    def test_gpcc_quantization_matches_real_encoder(self):
        args = args_for('G-PCC')
        points = torch.tensor([[[-3.4, -0.9, 0.1, 4.8, 7.0],
                                [0.1, 3.9, 1.1, 5.3, 2.0],
                                [5.1, 6.1, 4.9, 9.0, 1.0]]])
        coords, meta = quantize_geometry_octree(points, args, return_metadata=True)
        encoder = GPCCGeometryEncoder('', '', '', effective_qs=2.)
        np.testing.assert_array_equal(encoder._quantized_coords(points[0]),
                                     np.unique(coords[0].T.numpy(), axis=0))
        restored, info = restore_points_from_voxel_coords(coords, meta, args, unique=False)
        torch.testing.assert_close(quantize_geometry_octree(restored, args), coords)
        self.assertFalse(info['restore_center'])

    def test_octattention_quantization_matches_model_sequence(self):
        args = args_for('OctAttention')
        points = torch.rand((1, 3, 70), generator=torch.Generator().manual_seed(5)) * 30 - 7.3
        coords = quantize_geometry_octree(points, args)
        expected = np.unique(np.round((points[0].T.numpy().astype(np.float64) -
                                      points[0].T.numpy().astype(np.float64).min(0)) / 2).astype(np.int64), axis=0)
        np.testing.assert_array_equal(np.unique(coords[0].T.numpy(), axis=0), expected)
        proxy = SoftOctreeRateProxy(ProxyOctreeConfig(qs=2., max_depth=12))
        actual, _, _ = proxy._build_hard_octattention_sequence(points[0])
        reference, _ = proxy._build_octattention_sequence_from_qpts(expected)
        np.testing.assert_array_equal(actual, reference)

    def test_conventional_dispatch_never_calls_sparsepcgc_quantizer(self):
        points = torch.rand(1, 3, 80) * 20
        for codec in ('G-PCC', 'OctAttention'):
            args = args_for(codec)
            with patch('models.utils.pointcloud.sparsepcgc_voxel.canonical_sparsepcgc_voxel_coords', side_effect=AssertionError('SparsePCGC')):
                coords, meta = quantize_sparsepcgc_coords(points, args, return_metadata=True)
            torch.testing.assert_close(coords, quantize_geometry_octree(points, args))
            restored, _ = restore_points_from_voxel_coords(coords, meta, args, unique=False)
            torch.testing.assert_close(restored, meta['global_offset_tensor'] + coords * meta['effective_qs_tensor'])

    def test_unique_node_context_is_not_truncated_raw_point_context(self):
        args = args_for('G-PCC')
        points = torch.rand((1, 3, 300)) * 5
        ctx = build_geometry_octree_context(points, args)
        unique = torch.unique(ctx['global_voxel_coords'][0].T, dim=0).T[None]
        aligned = align_geometry_context(ctx, unique)
        self.assertLess(unique.shape[-1], points.shape[-1])
        torch.testing.assert_close(aligned['global_voxel_coords'], unique)
        analyzer = GeometryOctreeStructureAnalysis(args)
        xyz = ctx['global_offset'] + unique * ctx['global_qs']
        features = analyzer._prebuilt_octree_context(xyz, full_octree_context=aligned)
        self.assertEqual(features.shape, (1, 8, unique.shape[-1]))
        self.assertEqual(analyzer.feature_dim, 40)

    def test_pools_are_legal_reproducible_and_codec_namespaced(self):
        points = torch.randint(0, 64, (1, 3, 500), generator=torch.Generator().manual_seed(21)).float()
        signatures = []
        for codec in ('G-PCC', 'OctAttention'):
            args = args_for(codec)
            ctx = build_geometry_octree_context(points, args)
            payload = attach_geometry_octree_guidance(ctx, args, device=points.device)['ana_den6_ranked_candidate_guidance']
            again = attach_geometry_octree_guidance(ctx, args, device=points.device)['ana_den6_ranked_candidate_guidance']
            self.assertEqual(payload['cache_signature'], again['cache_signature'])
            self.assertEqual(payload['actual_candidate_encode_count'], 0)
            occupied = set(map(tuple, ctx['global_voxel_coords'][0].T.tolist()))
            for op, pool in payload['operation_edit_units'].items():
                self.assertGreater(len(pool), 0)
                for candidate in pool:
                    self.assertTrue(all(tuple(p) in occupied for p in candidate['remove_coords']))
                    self.assertTrue(all(tuple(p) not in occupied for p in candidate['add_coords']))
            signatures.append(payload['cache_signature'])
        self.assertNotEqual(*signatures)

    def test_compression_alias_selects_backend_and_keeps_actual_enabled(self):
        for codec, backend in [('G-PCC', 'gpcc_surrogate'), ('OctAttention', 'octattention_surrogate')]:
            parser = argparse.ArgumentParser()
            parser.add_argument('--trainORtest', default='train')
            with patch('sys.argv', ['test', '--compression', codec]):
                args = parse_pugan_args(parser, '20261016', 'test')
            self.assertEqual(args.compression_loss_backend, backend)
            self.assertEqual(args.method_com, codec)
            self.assertFalse(args.disable_actual_codec_during_train)
            self.assertFalse(args.use_amp)
            self.assertFalse(args.heuristic_guidance_network_only_inference)
            self.assertTrue(args.repair_output_voxel_restored_points)
            self.assertEqual(StructureRepairActuator._algorithmic_amount_bin_values_from_args(args),
                             list(args.geometry_octree_amount_bin_values))
            actuator = StructureRepairActuator(in_channels=8, hidden_dim=8, args=args)
            self.assertEqual(actuator.algorithmic_amount_selector_head[-1].out_channels,
                             len(args.geometry_octree_amount_bin_values) + 1)
            if codec == 'OctAttention':
                self.assertTrue(args.octattention_actualcode)

    def test_cross_codec_backend_and_unscaled_fp16_capture_are_rejected(self):
        for extra in (['--compression_loss_backend', 'sparsepcgc_surrogate'],
                      ['--compression_loss_backend', 'octattention_surrogate'],
                      ['--use_amp', 'true']):
            parser = argparse.ArgumentParser()
            parser.add_argument('--trainORtest', default='train')
            with patch('sys.argv', ['test', '--compression', 'G-PCC'] + extra):
                with self.assertRaises(ValueError):
                    parse_pugan_args(parser, '20261016', 'test')

    def test_gpcc_without_prequantization_uses_unit_grid(self):
        args = args_for('G-PCC')
        args.gpcc_prequantize = False
        self.assertEqual(geometry_step(args), 1.)

    def test_actual_and_feature_caches_are_codec_and_setting_specific(self):
        gpcc, oa = args_for('G-PCC'), args_for('OctAttention')
        self.assertNotEqual(geometry_actual_cache_fingerprint(gpcc, 'frame'),
                            geometry_actual_cache_fingerprint(oa, 'frame'))
        self.assertNotEqual(geometry_cache_key(gpcc, 'frame'), geometry_cache_key(oa, 'frame'))
        original = geometry_actual_cache_fingerprint(gpcc, 'frame')
        gpcc.gpcc_effective_qs = 3.
        self.assertNotEqual(original, geometry_actual_cache_fingerprint(gpcc, 'frame'))
        original = geometry_actual_cache_fingerprint(oa, 'frame')
        oa.octattention_actualcode = True
        self.assertNotEqual(original, geometry_actual_cache_fingerprint(oa, 'frame'))


if __name__ == '__main__':
    unittest.main()
