from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from models.network import Network


class _FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, points, feat=None):
        self.calls += 1
        base = points.mean(dim=1, keepdim=True)
        fused = base.repeat(1, 64, 1)
        channel_offset = torch.linspace(
            0.0, 1.0, 64, device=points.device, dtype=points.dtype
        ).view(1, 64, 1)
        fused = fused + channel_offset
        return fused, fused


def _network():
    network = Network.__new__(Network)
    nn.Module.__init__(network)
    network.args = SimpleNamespace(
        point_transformer_node_features=True,
        point_transformer_node_feature_dim=8,
        point_transformer_node_feature_scale=0.25,
        point_transformer_feature_gate_max=0.25,
        point_transformer_feature_warmup_steps=100,
        point_transformer_feature_cache=True,
        point_transformer_feature_cache_max_entries=4,
        point_transformer_feature_cache_max_memory_mb=16,
        encoder_0grad=True,
        encoder_pre_downsample_max_points=8,
        encoder_raw_downsample_factor=2.0,
        encoder_pre_downsample_voxel_scale=1.0,
        encoder_pre_downsample_growth=1.5,
        encoder_pre_downsample_max_iters=8,
        compress="SparsePCGC",
        sparsepcgc_voxel_size=1.0,
        sparsepcgc_pos_quantscale=1,
    )
    network.point_transformer_node_features = True
    network.point_transformer_node_feature_dim = 8
    network.point_transformer_feature_gate = nn.Parameter(
        torch.full((1, 8, 1), 0.1)
    )
    network.point_transformer_feature_adapter = nn.Sequential(
        nn.Conv1d(64, 16, 1),
        nn.SiLU(inplace=True),
        nn.Conv1d(16, 8, 1),
        nn.Tanh(),
    )
    network.encoder = _FrozenEncoder()
    network.point_transformer_feature_cache = {}
    # Production uses OrderedDict.move_to_end/popitem(last=False).
    from collections import OrderedDict
    network.point_transformer_feature_cache = OrderedDict()
    network._point_transformer_feature_cache_bytes = 0
    network._point_transformer_feature_cache_hits = 0
    network._point_transformer_feature_cache_misses = 0
    network._point_transformer_feature_cache_working_set_bypassed = 0
    network.input_cache = OrderedDict()
    network._input_cache_bytes = 0
    return network


class PointTransformerNodeFeatureTest(unittest.TestCase):
    def test_feature_gate_is_bounded_and_warmed_up_smoothly(self):
        network = _network()
        reference = torch.zeros(1, 16, 4)

        network.args._global_train_step = 0
        start_gate, start_warmup = network._point_transformer_fusion_gate(reference)
        self.assertLess(start_warmup, 0.001)
        self.assertLess(float(start_gate.detach().abs().max()), 0.001)

        network.args._global_train_step = 99
        with torch.no_grad():
            network.point_transformer_feature_gate.fill_(10.0)
        final_gate, final_warmup = network._point_transformer_fusion_gate(reference)
        self.assertAlmostEqual(final_warmup, 1.0)
        self.assertLessEqual(float(final_gate.detach().abs().max()), 0.25)

    def test_auto_warmup_tracks_the_exploration_curriculum(self):
        network = _network()
        network.args.point_transformer_feature_gate_max = 0.10
        network.args.point_transformer_feature_warmup_steps = 0
        network.args._total_train_steps_estimate = 10240
        network.args.repair_exploration_fraction = 0.9
        reference = torch.zeros(1, 16, 4)

        network.args._global_train_step = 1999
        early_gate, early_warmup = network._point_transformer_fusion_gate(reference)
        self.assertLess(early_warmup, 0.15)
        self.assertLess(float(early_gate.detach().abs().max()), 0.015)

        network.args._global_train_step = 9215
        final_gate, final_warmup = network._point_transformer_fusion_gate(reference)
        self.assertAlmostEqual(final_warmup, 1.0)
        self.assertLessEqual(float(final_gate.detach().abs().max()), 0.10)

    def test_frozen_encoder_features_reach_trainable_node_adapter_and_cache(self):
        torch.manual_seed(5)
        network = _network()
        points = torch.stack((
            torch.arange(16, dtype=torch.float32),
            torch.zeros(16),
            torch.zeros(16),
        )).unsqueeze(0)
        counts = torch.tensor([16])

        first, first_debug = network._point_transformer_features_for_nodes(
            points,
            counts,
            cache_key="frame0.ply",
            source="full_octree_context",
        )
        self.assertEqual(tuple(first.shape), (1, 8, 16))
        self.assertTrue(first.requires_grad)
        self.assertTrue(first_debug["flows_to_actuator"])
        self.assertEqual(network.encoder.calls, 1)
        first.square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in network.point_transformer_feature_adapter.parameters()
        ))

        second, second_debug = network._point_transformer_features_for_nodes(
            points,
            counts,
            cache_key="frame0.ply",
            source="full_octree_context",
        )
        self.assertEqual(network.encoder.calls, 1)
        self.assertEqual(second_debug["cache_hits_this_forward"], 1)
        self.assertTrue(torch.allclose(first.detach(), second.detach(), atol=2e-4, rtol=0.0))

        # 巨大な静的Node cacheだけをbypass/clearしても、再利用可能な固定
        # Point Transformer特徴まで誤って捨てない。
        network.clear_input_cache()
        third, third_debug = network._point_transformer_features_for_nodes(
            points,
            counts,
            cache_key="frame0.ply",
            source="full_octree_context",
        )
        self.assertEqual(network.encoder.calls, 1)
        self.assertEqual(third_debug["cache_hits_this_forward"], 1)
        self.assertTrue(torch.allclose(second.detach(), third.detach(), atol=2e-4, rtol=0.0))

    def test_cache_is_bypassed_when_sequential_working_set_cannot_fit(self):
        torch.manual_seed(5)
        network = _network()
        network.expected_input_cache_entries = 100
        points = torch.stack((
            torch.arange(16, dtype=torch.float32),
            torch.zeros(16),
            torch.zeros(16),
        )).unsqueeze(0)
        counts = torch.tensor([16])

        network._point_transformer_features_for_nodes(
            points,
            counts,
            cache_key="frame0.ply",
            source="full_octree_context",
        )
        network._point_transformer_features_for_nodes(
            points,
            counts,
            cache_key="frame0.ply",
            source="full_octree_context",
        )

        self.assertEqual(network.encoder.calls, 2)
        self.assertEqual(len(network.point_transformer_feature_cache), 0)
        self.assertEqual(
            network._point_transformer_feature_cache_working_set_bypassed, 2
        )


if __name__ == "__main__":
    unittest.main()
