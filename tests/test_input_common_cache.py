from types import SimpleNamespace
import unittest

import torch

from models.network import Network
from models.modules.octree_structure import OctreeStructureAnalysis
from train import (
    _clone_input_common_cache_value_to_cpu,
    _clone_input_common_cache_value_to_device,
    _episode_input_common_cache_fetch,
    _episode_input_common_cache_store,
    _estimate_input_common_cache_bytes,
)


class InputCommonCacheTest(unittest.TestCase):
    def test_tensor_alias_is_stored_once_and_preserved(self):
        coords = torch.arange(96, dtype=torch.long).reshape(1, 3, 32)
        context = {
            "global_voxel_coords": coords,
            "full_global_voxel_coords": coords,
            "full_occupied_voxel_coords": coords,
            "nested": {"coords": coords},
        }
        cached = _clone_input_common_cache_value_to_cpu(context)
        expected_bytes = coords.numel() * coords.element_size()
        self.assertEqual(_estimate_input_common_cache_bytes(cached), expected_bytes)
        self.assertIs(
            cached["global_voxel_coords"],
            cached["full_global_voxel_coords"],
        )
        restored = _clone_input_common_cache_value_to_device(cached, device="cpu")
        self.assertIs(
            restored["global_voxel_coords"],
            restored["full_occupied_voxel_coords"],
        )
        self.assertTrue(torch.equal(restored["global_voxel_coords"], coords))

    def test_warm_fetch_is_value_identical(self):
        args = SimpleNamespace(
            episode_input_common_cache=True,
            episode_input_common_cache_max_entries=2,
            episode_input_common_cache_max_memory_mb=1,
            _episode_input_common_cache_auto_max_entries=2,
        )
        coords = torch.arange(48, dtype=torch.long).reshape(1, 3, 16)
        value = {"global_voxel_coords": coords, "alias": coords}
        _episode_input_common_cache_store(args, "frame", value)
        restored = _episode_input_common_cache_fetch(
            args, "frame", device=torch.device("cpu"), section="test"
        )
        self.assertIsNotNone(restored)
        self.assertTrue(torch.equal(restored["global_voxel_coords"], coords))
        self.assertIs(restored["global_voxel_coords"], restored["alias"])
        self.assertEqual(args._episode_input_common_cache_stats["test"]["hit"], 1)

    def test_static_node_cache_is_cpu_and_value_identical(self):
        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace(
            cache_max_entries=4,
            cache_max_memory_mb=1,
            static_node_cache_cpu=True,
            qs=2,
            sparsepcgc_voxel_size=1.0,
            sparsepcgc_pos_quantscale=1,
            sparsepcgc_quant_mode="round_voxel_then_pos",
            sparsepcgc_dequantize_center=False,
            fused_feat_dim=64,
        )
        network.cache_enabled = True
        network.input_cache = __import__("collections").OrderedDict()
        network._input_cache_bytes = 0
        network._input_cache_working_set_bypassed = 0
        network.expected_input_cache_entries = 1
        coords = torch.arange(48, dtype=torch.long).reshape(1, 3, 16)
        features = torch.randn(1, 5, 16)
        state = {
            "voxel_coords": coords,
            "coords_alias": coords,
            "node_features": features,
            "nested": {"features_alias": features},
        }
        network._put_static_node_cache("frame", "full_octree_context", state)
        self.assertEqual(len(network.input_cache), 1)
        stored = next(iter(network.input_cache.values()))["state"]
        self.assertEqual(stored["voxel_coords"].device.type, "cpu")
        self.assertIs(stored["voxel_coords"], stored["coords_alias"])
        restored = network._get_static_node_cache(
            "frame", "full_octree_context", torch.device("cpu")
        )
        self.assertTrue(torch.equal(restored["voxel_coords"], coords))
        self.assertTrue(torch.equal(restored["node_features"], features))
        self.assertIs(
            restored["node_features"],
            restored["nested"]["features_alias"],
        )

    def test_fixed_octree_cache_is_bitwise_identical(self):
        args = SimpleNamespace(
            compress="SparsePCGC",
            sparsepcgc_effective_qs=1.0,
            sparsepcgc_voxel_size=1.0,
            sparsepcgc_pos_quantscale=1,
            octree_ctx_level=5,
            octree_ctx_dim=8,
            structure_geo_k=8,
            structure_geo_max_points=0,
            proxy_max_depth=12,
            octree_diag_levels="4,6,8,10,12",
            structure_neighbor_query_chunk=3,
            force_full_cloud_canonical_voxel_basis=True,
            heuristic_guidance_mode="network_only_codec_policy",
            leaf_pattern_feature_integration=True,
            structure_fixed_cache_max_entries=4,
            structure_fixed_cache_max_memory_mb=16,
        )
        rows = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [2, 2, 2],
                [3, 2, 2],
                [2, 3, 2],
                [4, 4, 4],
            ],
            dtype=torch.long,
        )
        coords = rows.transpose(0, 1).unsqueeze(0)
        context = {
            "global_voxel_coords": coords,
            "global_morton_keys": torch.arange(rows.shape[0]),
            "octree_context_scope": "full_cloud",
        }
        module = OctreeStructureAnalysis(args)
        cold = module(
            coords.float(),
            full_octree_context=context,
            octree_input_mode="full_cloud",
            cache_key="frame",
        )
        warm = module(
            coords.float(),
            full_octree_context=context,
            octree_input_mode="full_cloud",
            cache_key="frame",
        )
        self.assertFalse(cold["fixed_structure_cache_hit"])
        self.assertTrue(warm["fixed_structure_cache_hit"])
        for key in (
            "features",
            "cause_targets",
            "oct_ctx",
            "geo_stats",
            "quant_stats",
            "network_only_fixed_features",
            "network_k_fixed_features",
        ):
            self.assertTrue(torch.equal(cold[key], warm[key]), key)


if __name__ == "__main__":
    unittest.main()
