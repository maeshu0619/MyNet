"""SparsePCGC actual teacherの符号・no-op回帰テスト。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import models.modules.heuristic_guidance as heuristic_guidance_module
from models.modules.heuristic_guidance import (
    build_heuristic_guidance,
    release_exact_guidance_cache,
    resolve_profile,
)
from models.modules.octree_structure import OctreeStructureAnalysis
from models.modules.structure_actuator import StructureRepairActuator
from models.network import Network
from models.utils.loss.compression import CompressionLossMixin
from models.utils.loss.sparsepcgc_teacher_worker import (
    _encoder_reported_cuda_cleanup,
)
from models.utils.training.utils import surrogate_compression_plot_metric
from models.utils.training.train_flow import prepare_full_cloud_input_pcd
from models.utils.pointcloud.ana_den6_reference import (
    _coord_hash,
    _current_den6_sha256,
    attach_ana_den6_reference_anchor,
)
from models.utils.pointcloud.ana_den6_online import attach_ana_den6_online_guidance


class _ActualCodecFixture(CompressionLossMixin):
    """外部codecを起動せず、actual deltaの扱いだけを検証するfixture。"""

    def __init__(self, gt_xyz: torch.Tensor, edit_record_bits: float = 0.0) -> None:
        self.gt_xyz = gt_xyz
        self.edit_record_bits = float(edit_record_bits)
        self.cached_gt = None
        self.last_compression_debug = {}

    def _get_actuator_voxel_state(self, _args, _device):
        return {"estimated_edit_record_bits": self.edit_record_bits}

    def _get_cached_actual_gt(self, _cache_key):
        return self.cached_gt

    def _store_cached_actual_gt(self, _cache_key, value):
        self.cached_gt = value

    def _encode_actual_batch(self, _args, xyz, final_w=None):
        is_gt = xyz is self.gt_xyz
        bit = 100.0 if is_gt else 110.0
        return {
            "codec": "sparsepcgc",
            "bit": bit,
            "bpp": bit / max(float(xyz.shape[-1]), 1.0),
            "bpn": bit / 10.0,
            "single": 5.0,
            "node": 10.0,
            "point_count": int(xyz.shape[-1]),
            "sparsepcgc_worker_launch_count": 1,
            "sparsepcgc_worker_request_count": 1,
            "sparsepcgc_actual_result_cache_hit": False,
        }

    @staticmethod
    def _is_sparsepcgc_context(_args, **_kwargs):
        return True

    def _store_compression_terms(self, **_kwargs):
        pass

    @staticmethod
    def _should_verbose_step(_args):
        return False

    @staticmethod
    def _log_compression_grad_probe(*_args, **_kwargs):
        pass

    @staticmethod
    def _scalar(value):
        return float(value.detach()) if torch.is_tensor(value) else float(value)


class _EveryStepFixture(CompressionLossMixin):
    def __init__(self) -> None:
        self.actual_calls = 0

    @staticmethod
    def _actual_codec_disabled_for_train(_args):
        return False

    @staticmethod
    def _is_sparsepcgc_context(_args, **_kwargs):
        return True

    def _get_actual_codec_with_proxy_fallback(self, *_args, **kwargs):
        self.actual_calls += 1
        ref = kwargs["gen_xyz"]
        tensor = ref.new_zeros(())
        return tensor, tensor, tensor, tensor, {}, {}

    @staticmethod
    def _get_compression_loss_proxy(*_args, **_kwargs):
        raise AssertionError("SparsePCGC actual_every_step must not fall back to proxy")


class SparsePCGCActualSemanticsTest(unittest.TestCase):
    def test_worker_does_not_repeat_encoder_cuda_cleanup(self):
        self.assertTrue(_encoder_reported_cuda_cleanup({
            "cuda_after_cleanup_reserved_gb": 0.0
        }))
        self.assertFalse(_encoder_reported_cuda_cleanup({
            "cuda_reserved_gb": 1.0
        }))

    def test_actual_batch_preserves_wrapper_runtime_breakdown(self):
        class Encoder:
            codec_name = "sparsepcgc"

            def encode_bits(self, pts):
                return {
                    "bit": 80.0,
                    "bpp": 20.0,
                    "bpn": 8.0,
                    "single": 2.0,
                    "node": 10.0,
                    "point_count": int(pts.shape[-1]),
                    "encode_time": 4.0,
                    "sparsepcgc_input_prepare_time": 0.1,
                    "sparsepcgc_ply_write_time": 0.2,
                    "sparsepcgc_worker_roundtrip_time": 3.7,
                }

        class Fixture(CompressionLossMixin):
            def _get_actual_encoder(self, _args):
                return Encoder()

            def _attach_octree_aux_stats(self, _args, _pts, stats):
                return stats

        result = Fixture()._encode_actual_batch(
            SimpleNamespace(), torch.zeros((1, 3, 4))
        )
        self.assertEqual(result["bit"], 80.0)
        self.assertEqual(result["sparsepcgc_input_prepare_time"], 0.1)
        self.assertEqual(result["sparsepcgc_ply_write_time"], 0.2)
        self.assertEqual(result["sparsepcgc_worker_roundtrip_time"], 3.7)

    def test_cpu_guidance_cache_does_not_retain_exact_manifest(self):
        """固定Tensor cacheへ巨大なframe別candidate manifestを保持しない。"""
        module = heuristic_guidance_module
        key = ("memory-regression", "torch.float32", 3)
        exact = {
            "operation_edit_units": {
                "Prune": [{"payload": "x" * 1024}],
            },
        }
        guidance = {
            "candidate_mask": {"Prune": torch.ones(3, dtype=torch.bool)},
            "numpy_debug": np.zeros((128,), dtype=np.float32),
            "exact_candidate_guidance": exact,
        }
        args = SimpleNamespace(
            heuristic_guidance_cpu_tensor_cache_entries=2,
            heuristic_guidance_cpu_tensor_cache_max_memory_mb=1,
        )
        try:
            module._store_exact_guidance_cpu(key, guidance, exact, args)
            stored = module._EXACT_GUIDANCE_CPU_CACHE[key]["guidance"]
            self.assertNotIn("exact_candidate_guidance", stored)
            self.assertEqual(
                module._guidance_tensor_bytes(stored["numpy_debug"]),
                int(stored["numpy_debug"].nbytes),
            )
        finally:
            removed = module._EXACT_GUIDANCE_CPU_CACHE.pop(key, None)
            if isinstance(removed, dict):
                module._EXACT_GUIDANCE_CPU_CACHE_BYTES -= int(
                    removed.get("bytes", 0)
                )

    def test_full_cloud_input_preserves_every_point(self):
        points = torch.arange(2 * 17 * 6, dtype=torch.float32).reshape(2, 17, 6)
        prepared = prepare_full_cloud_input_pcd(points, use_cuda=False)
        self.assertEqual(tuple(prepared.shape), (2, 6, 17))
        self.assertTrue(torch.equal(prepared, points.permute(0, 2, 1)))

    def test_surrogate_plot_uses_actual_ste_backward_proxy_not_actual_delta(self):
        loss_obj = SimpleNamespace(
            last_compression_debug={
                "rate_proxy_delta": -9.0,
                "actual_total_bit_percent": -9.0,
                "proxy_surrogate": {"loss_bit": 2.5},
            }
        )
        plotted = surrogate_compression_plot_metric(loss_obj, torch.tensor(-9.0), torch.device("cpu"))
        self.assertEqual(float(plotted), 2.5)

    def test_surrogate_plot_is_missing_without_a_backward_surrogate(self):
        loss_obj = SimpleNamespace(
            last_compression_debug={"rate_proxy_delta": -9.0, "actual_total_bit_percent": -9.0}
        )
        plotted = surrogate_compression_plot_metric(loss_obj, torch.tensor(-9.0), torch.device("cpu"))
        self.assertIsNone(plotted)

    def test_release_step_transient_state_drops_only_bridge_references(self):
        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace()
        network.actuator = torch.nn.Identity()
        network.actuator.debug_tensors = {"large": torch.ones(8)}
        network.cost_attributor = torch.nn.Identity()
        network.cost_attributor.debug_tensors = {"large": torch.ones(8)}
        network.policy_module = torch.nn.Identity()
        network.policy_module.debug_tensors = {"large": torch.ones(8)}
        network.debug_tensors = {"large": torch.ones(8)}
        network.last_structure_debug = {"large": torch.ones(8)}
        network.last_encoder_debug = {"large": torch.ones(8)}
        network.last_actuator_voxel_state = {"coords": torch.ones(8)}
        network.last_actuator_soft_terms = {"soft": torch.ones(8)}
        network.last_k_proposal_terms = {"large": torch.ones(8)}
        network.last_single_plan_student_terms = {"large": torch.ones(8)}
        network.last_k_all_actual_debug = {"large": torch.ones(8)}
        network.input_cache = {"reusable": {"state": torch.ones(8)}}

        network.release_step_transient_state()

        self.assertEqual(network.debug_tensors, {})
        self.assertEqual(network.last_structure_debug, {})
        self.assertEqual(network.last_encoder_debug, {})
        self.assertIsNone(network.last_actuator_voxel_state)
        self.assertEqual(network.last_actuator_soft_terms, {})
        self.assertIsNone(network.last_k_proposal_terms)
        self.assertIsNone(network.last_single_plan_student_terms)
        self.assertEqual(network.last_k_all_actual_debug, {})
        self.assertEqual(network.actuator.debug_tensors, {})
        self.assertEqual(network.cost_attributor.debug_tensors, {})
        self.assertEqual(network.policy_module.debug_tensors, {})
        self.assertIn("reusable", network.input_cache)
        self.assertIsNone(network.args._last_actuator_voxel_state)
        self.assertEqual(network.args._last_actuator_soft_terms, {})

    def test_noop_guard_does_not_replace_worse_actual_delta(self):
        gt_xyz = torch.zeros((1, 3, 4), dtype=torch.float32)
        gen_xyz = torch.ones((1, 3, 4), dtype=torch.float32)
        fixture = _ActualCodecFixture(gt_xyz)
        args = SimpleNamespace(
            sparsepcgc_actual_use_actuator_voxel_state=False,
            sparsepcgc_edit_record_bits_enabled=False,
            compression_loss_delta=True,
            sparsepcgc_policy_actual_noop_guard=True,
            sparsepcgc_policy_actual_noop_guard_margin=0.0,
            enable_sparsepcgc_exact_occupancy_loss=False,
            sparsepcgc_exact_teacher_loss_weight=0.0,
            sparsepcgc_exact_teacher_grad_weight=1.0,
            sparsepcgc_exact_fallback_weight=0.2,
        )

        loss, *_ = fixture._get_compression_loss_actual_codec(
            args,
            gen_xyz=gen_xyz,
            gt_xyz=gt_xyz,
            final_w=None,
            cache_key="fixture",
            use_proxy_surrogate=False,
        )

        self.assertAlmostEqual(float(loss), 10.0)
        self.assertTrue(fixture.last_compression_debug["policy_actual_noop_guard_used"])
        self.assertAlmostEqual(
            float(fixture.last_compression_debug["actual_total_bit_percent"]),
            10.0,
        )

    def test_mvub_profile_uses_saved_den6_actual_row(self):
        profile = resolve_profile(SimpleNamespace(dataname="MVUB", sparsepcgc_scale_m=8))
        self.assertEqual((profile.add_share, profile.prune_share, profile.adjust_share), (0.50, 0.40, 0.10))
        self.assertEqual(profile.total_ratio, 0.0025)

    def test_raw_objective_excludes_edit_record_bits_like_den6(self):
        gt_xyz = torch.zeros((1, 3, 4), dtype=torch.float32)
        gen_xyz = torch.ones((1, 3, 4), dtype=torch.float32)
        fixture = _ActualCodecFixture(gt_xyz, edit_record_bits=50.0)
        args = SimpleNamespace(
            sparsepcgc_actual_use_actuator_voxel_state=False,
            sparsepcgc_edit_record_bits_enabled=True,
            sparsepcgc_actual_bit_objective="raw",
            compression_loss_delta=True,
            sparsepcgc_policy_actual_noop_guard=False,
            sparsepcgc_policy_actual_noop_guard_margin=0.0,
            enable_sparsepcgc_exact_occupancy_loss=False,
            sparsepcgc_exact_teacher_loss_weight=0.0,
            sparsepcgc_exact_teacher_grad_weight=1.0,
            sparsepcgc_exact_fallback_weight=0.2,
        )
        loss, *_ = fixture._get_compression_loss_actual_codec(
            args, gen_xyz=gen_xyz, gt_xyz=gt_xyz, final_w=None, cache_key="raw_fixture", use_proxy_surrogate=False
        )
        self.assertAlmostEqual(float(loss), 10.0)
        self.assertAlmostEqual(float(fixture.last_compression_debug["actual_total_bit_percent"]), 60.0)
        self.assertAlmostEqual(float(fixture.last_compression_debug["actual_objective_percent"]), 10.0)

    def test_billed_objective_includes_edit_record_bits(self):
        gt_xyz = torch.zeros((1, 3, 4), dtype=torch.float32)
        gen_xyz = torch.ones((1, 3, 4), dtype=torch.float32)
        fixture = _ActualCodecFixture(gt_xyz, edit_record_bits=50.0)
        args = SimpleNamespace(
            sparsepcgc_actual_use_actuator_voxel_state=False,
            sparsepcgc_edit_record_bits_enabled=True,
            sparsepcgc_actual_bit_objective="billed",
            compression_loss_delta=True,
            sparsepcgc_policy_actual_noop_guard=False,
            sparsepcgc_policy_actual_noop_guard_margin=0.0,
            enable_sparsepcgc_exact_occupancy_loss=False,
            sparsepcgc_exact_teacher_loss_weight=0.0,
            sparsepcgc_exact_teacher_grad_weight=1.0,
            sparsepcgc_exact_fallback_weight=0.2,
        )
        loss, *_ = fixture._get_compression_loss_actual_codec(
            args, gen_xyz=gen_xyz, gt_xyz=gt_xyz, final_w=None, cache_key="billed_fixture", use_proxy_surrogate=False
        )
        self.assertAlmostEqual(float(loss), 60.0)

    def test_sparsepcgc_actual_every_step_ignores_periodic_interval(self):
        fixture = _EveryStepFixture()
        xyz = torch.zeros((1, 3, 2), dtype=torch.float32)
        args = SimpleNamespace(
            compression_loss_backend="sparsepcgc_actual_ste",
            trainORtest="train",
            sparsepcgc_actual_every_step=True,
            actual_eval_interval=20,
            _global_train_step=7,
        )
        fixture.get_compression_loss(args, xyz, xyz, final_w=None)
        self.assertEqual(fixture.actual_calls, 1)

    def test_den6_online_rejects_multi_plan_actual_encode_during_train(self):
        fixture = _EveryStepFixture()
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            _den6_online_training_step_active=True,
        )
        with self.assertRaisesRegex(RuntimeError, "複数候補actual encodeを禁止"):
            fixture._encode_actual_many(args, [torch.zeros((3, 1))])

    def test_den6_online_v7_cache_contains_gt_terms_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "frame.ply"
            input_file.write_bytes(b"synthetic-ply-identity")
            args = SimpleNamespace(
                heuristic_guidance_mode="ana_den6_online",
                heuristic_guidance_online_cache_dir=str(root / "cache"),
                heuristic_guidance_online_memory_entries=2,
                _current_input_file=str(input_file),
                dataname="8i",
                sparsepcgc_voxel_size=1.0,
                sparsepcgc_pos_quantscale=1,
                sparsepcgc_scale_ae=0,
                sparsepcgc_scale_sr=2,
                sparsepcgc_scale_m=8,
                sparsepcgc_mode="dense_lossy",
                heuristic_guidance_require_exact_single_plan_teacher=False,
            )
            context = {
                "global_voxel_coords": torch.tensor(
                    [[[1, 0, 2], [0, 0, 1], [0, 0, 0]]], dtype=torch.long
                )
            }
            attached = attach_ana_den6_online_guidance(
                context, args, device=torch.device("cpu")
            )
            payload = attached["ana_den6_ranked_candidate_guidance"]
            self.assertEqual(payload["proposal_policy"], "one_where_amount_action_per_step")
            self.assertEqual(payload["gt_geometry_terms"]["unique_voxel_count"], 3)
            for forbidden in (
                "operation_candidate_shortlists", "initial_heuristic_plan",
                "selected_candidate_ids", "actual_generated_loss", "surrogate_target",
            ):
                self.assertNotIn(forbidden, payload)

    def test_den6_online_restores_exact_edit_unit_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "frame.ply"
            input_file.write_bytes(b"synthetic-ply-identity")
            import hashlib
            input_sha256 = hashlib.sha256(input_file.read_bytes()).hexdigest()
            setting_id = "vs1_pq1_ae0_sr2_m8"
            edit = {
                "candidate_id": "P0",
                "operation": "Prune",
                "remove_coords": [[1, 0, 0]],
                "add_coords": [],
                "pool_rank": 0,
                "rank_score": 1.0,
            }
            payload = {
                "schema_version": "ana_den6_online_one_pattern_cache_v6",
                "source": "ana_den6_exact_one_pattern_anchor_online_v6",
                "input_file": str(input_file),
                "input_sha256": input_sha256,
                "dataset": "8i",
                "setting_id": setting_id,
                "scale_m": 8,
                "scale_ae": 0,
                "scale_sr": 2,
                "total_ratio": 0.0025,
                "operation_shares": {"Add": 0.4, "Prune": 0.4, "Adjust": 0.2},
                "operation_priority": ["Prune", "Add", "Adjust"],
                "operation_candidate_shortlists": {
                    "Add": [{**edit, "candidate_id": "A0", "operation": "Add", "remove_coords": [], "add_coords": [[2, 0, 0]]}],
                    "Prune": [edit],
                    "Adjust": [{**edit, "candidate_id": "M0", "operation": "Adjust", "add_coords": [[0, 1, 0]]}],
                },
                "initial_heuristic_plan": {
                    "available": True,
                    "candidate_ids": ["P0", "A0", "M0"],
                    "operation_counts": {"Add": 1, "Prune": 1, "Adjust": 1},
                    "metadata": {"operation_order": "Prune>Add>Adjust"},
                },
            }
            cache = root / "cache"
            cache.mkdir()
            torch.save(payload, cache / f"8i_frame_{setting_id}_fixture.pt")
            args = SimpleNamespace(
                heuristic_guidance_mode="ana_den6_online",
                heuristic_guidance_online_cache_dir=str(cache),
                heuristic_guidance_online_memory_entries=2,
                heuristic_guidance_online_compact_reserve_factor=1.0,
                heuristic_guidance_require_exact_single_plan_teacher=True,
                heuristic_guidance_teacher_bootstrap_steps=200,
                _den6_online_training_step_active=True,
                _global_train_step=0,
                _current_input_file=str(input_file),
                dataname="8i",
                sparsepcgc_voxel_size=1.0,
                sparsepcgc_pos_quantscale=1,
                sparsepcgc_scale_ae=0,
                sparsepcgc_scale_sr=2,
                sparsepcgc_scale_m=8,
                sparsepcgc_mode="dense_lossy",
            )
            context = {
                "global_voxel_coords": torch.tensor(
                    [[[1, 0, 2], [0, 0, 1], [0, 0, 0]]], dtype=torch.long
                )
            }
            attached = attach_ana_den6_online_guidance(
                context, args, device=torch.device("cpu")
            )
            teacher = attached["ana_den6_ranked_candidate_guidance"]
            self.assertEqual(
                teacher["source"], "ana_den6_exact_single_plan_teacher_online_v8"
            )
            self.assertEqual(teacher["proposal_policy"], "one_where_amount_action_per_step")
            self.assertFalse(teacher["teacher_bootstrap_active"])
            self.assertEqual(teacher["teacher_bootstrap_steps"], 0)
            self.assertIn("operation_edit_units", teacher)
            self.assertNotIn("operation_candidate_shortlists", teacher)
            guidance = build_heuristic_guidance(
                {
                    "occupancy_nll_proxy": torch.zeros((1, 1, 3)),
                    "global_voxel_coords": context["global_voxel_coords"],
                    "ana_den6_ranked_candidate_guidance": teacher,
                },
                SimpleNamespace(
                    heuristic_guidance_mode="ana_den6_online",
                    heuristic_guidance_enabled=True,
                    dataname="8i",
                    sparsepcgc_scale_m=8,
                    _current_subtree_id="",
                    heuristic_guidance_tensor_cache_entries=2,
                ),
            )
            self.assertEqual(
                guidance["formula_basis"],
                "ana_den6_exact_single_plan_teacher_online_v8",
            )
            self.assertIn("candidate_mask", guidance)
            self.assertIn("candidate_tensor_map", guidance)
            self.assertEqual(int(guidance["candidate_mask"]["Prune"].sum()), 1)
            # GPU/Step cacheを解放しても、固定CPU座標写像から同じTensorを復元する。
            release_exact_guidance_cache(args)
            guidance_warm = build_heuristic_guidance(
                {
                    "occupancy_nll_proxy": torch.zeros((1, 1, 3)),
                    "global_voxel_coords": context["global_voxel_coords"],
                    "ana_den6_ranked_candidate_guidance": teacher,
                },
                args,
            )
            self.assertTrue(guidance_warm["cpu_tensor_cache_hit"])
            for operation in ("Add", "Prune", "Adjust"):
                self.assertTrue(torch.equal(
                    guidance["candidate_mask"][operation],
                    guidance_warm["candidate_mask"][operation],
                ))
                self.assertTrue(torch.equal(
                    guidance["candidate_tensor_map"][operation][
                        "static_compatible"
                    ],
                    guidance_warm["candidate_tensor_map"][operation][
                        "static_compatible"
                    ],
                ))

    def test_den6_online_v7_builds_single_proposal_prior_without_candidate_pool(self):
        like = torch.tensor([[[0.1, 0.8, 0.3]]], dtype=torch.float32)
        structure = {
            name: like.clone()
            for name in (
                "occupancy_nll_proxy", "node_proxy", "single_proxy", "lowprob_proxy",
                "context_proxy", "quant_proxy", "sparse_proxy", "outlier_proxy", "shape_proxy",
            )
        }
        structure["ana_den6_ranked_candidate_guidance"] = {
            "source": "ana_den6_gt_terms_single_proposal_online_v7",
            "cache_signature": "fixture-v7",
            "dataset": "8i",
            "scale_m": 8,
        }
        guidance = build_heuristic_guidance(
            structure,
            SimpleNamespace(
                heuristic_guidance_mode="ana_den6_online",
                heuristic_guidance_enabled=True,
                dataname="8i",
                sparsepcgc_scale_m=8,
                heuristic_guidance_total_ratio_percent=-1.0,
                heuristic_guidance_operation_shares="",
                heuristic_guidance_operation_heuristics="",
            ),
        )
        self.assertEqual(
            guidance["formula_basis"],
            "ana_den6_single_proposal_network_residual_online_v7",
        )
        self.assertEqual(guidance["proposal_policy"], "one_where_amount_action_per_step")
        self.assertNotIn("candidate_mask", guidance)
        self.assertNotIn("candidate_tensor_map", guidance)

    def test_den6_online_eval_guidance_is_network_only_and_has_no_teacher_data(self):
        structure = {"occupancy_nll_proxy": torch.zeros((1, 1, 5))}
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            heuristic_guidance_enabled=True,
            _heuristic_guidance_network_only_forward=True,
            dataname="8i",
            sparsepcgc_scale_m=8,
        )
        guidance = build_heuristic_guidance(structure, args)
        self.assertEqual(
            guidance["formula_basis"],
            "network_only_where_amount_action_inference_v1",
        )
        self.assertEqual(guidance["proposal_policy"], "one_network_plan_without_den6_or_cache")
        self.assertEqual(guidance["where_prior"], {})
        self.assertEqual(guidance["amount_prior"], {})
        self.assertEqual(guidance["action_gate_prior"], {})
        self.assertNotIn("exact_candidate_guidance", guidance)
        self.assertNotIn("candidate_mask", guidance)
        self.assertNotIn("candidate_tensor_map", guidance)

        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        actuator.args = args
        accepted = actuator._heuristic_guidance_state({"heuristic_guidance": guidance})
        self.assertIs(accepted, guidance)
        like = torch.randn((1, 1, 5))
        self.assertTrue(torch.equal(
            actuator._heuristic_where_logit_bias(guidance, "Prune", like),
            torch.zeros_like(like),
        ))

    def test_network_eval_forward_does_not_attach_den6_guidance(self):
        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            heuristic_guidance_enabled=True,
            heuristic_guidance_network_only_inference=True,
            full_cloud_activation_checkpoint=True,
            encoder_0grad=False,
        )
        network.cost_attributor = SimpleNamespace()
        network.policy_module = SimpleNamespace()
        network.training = False
        sentinel = ("network-only-early-return",)
        context = {"global_voxel_coords": torch.zeros((1, 3, 4), dtype=torch.long)}
        with patch(
            "models.network.attach_ana_den6_online_guidance",
            side_effect=AssertionError("eval must not call den6"),
        ), patch.object(
            Network, "_maybe_fast_full_cloud_oracle_forward", return_value=sentinel,
        ):
            result = network.forward(
                torch.zeros((1, 3, 4)),
                None,
                full_octree_context=context,
                octree_input_mode="full_cloud",
            )
        self.assertIs(result, sentinel)
        self.assertTrue(network.args._heuristic_guidance_network_only_forward)

    def test_den6_online_batch_aggregate_preserves_worker_request_counter(self):
        gt_xyz = torch.zeros((1, 3, 4), dtype=torch.float32)
        fixture = _ActualCodecFixture(gt_xyz)
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            _den6_online_training_step_active=True,
            sparsepcgc_actual_use_actuator_voxel_state=False,
            sparsepcgc_edit_record_bits_enabled=False,
            compression_loss_delta=True,
            sparsepcgc_policy_actual_noop_guard=False,
            sparsepcgc_policy_actual_noop_guard_margin=0.0,
            enable_sparsepcgc_exact_occupancy_loss=False,
            sparsepcgc_exact_teacher_loss_weight=0.0,
            sparsepcgc_exact_teacher_grad_weight=1.0,
            sparsepcgc_exact_fallback_weight=0.2,
            _global_train_step=0,
        )
        fixture._get_compression_loss_actual_codec(
            args,
            gen_xyz=torch.ones_like(gt_xyz),
            gt_xyz=gt_xyz,
            final_w=None,
            cache_key="online-counter",
            use_proxy_surrogate=False,
        )
        self.assertEqual(fixture._den6_online_actual_audit["edited"], 1)
        self.assertEqual(fixture._den6_online_actual_audit["worker_request_count"], 1)

    def test_den6_online_policy_gradient_sign_changes_with_actual_outcome(self):
        def gradient_for_objective(objective):
            network = Network.__new__(Network)
            torch.nn.Module.__init__(network)
            network.args = SimpleNamespace(
                heuristic_guidance_mode="ana_den6_online",
                _current_input_file="fixture.ply",
                sparsepcgc_scale_ae=0,
                sparsepcgc_scale_sr=2,
                sparsepcgc_scale_m=8,
                heuristic_guidance_online_reward_ema=0.1,
                heuristic_guidance_online_policy_weight=1.0,
                heuristic_guidance_online_entropy_weight=0.0,
            )
            network._den6_online_objective_baseline = __import__("collections").OrderedDict()
            log_prob = torch.tensor(0.0, requires_grad=True)
            network.last_actuator_voxel_state = {
                "den6_online_policy_log_prob": log_prob,
                "den6_online_policy_entropy": log_prob.new_zeros(()),
            }
            loss = network.discrete_policy_loss(torch.tensor(float(objective)))
            loss.backward()
            return float(log_prob.grad)

        # Gradient descent subtracts grad: improvement must increase log-prob,
        # while a worse actual compression result must decrease it.
        self.assertLess(gradient_for_objective(-1.0), 0.0)
        self.assertGreater(gradient_for_objective(1.0), 0.0)

    def test_den6_online_policy_uses_per_input_best_actual_baseline(self):
        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            _current_input_file="fixture.ply",
            sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2,
            sparsepcgc_scale_m=8,
            heuristic_guidance_online_policy_weight=1.0,
            heuristic_guidance_online_entropy_weight=0.0,
            heuristic_guidance_online_reward_scale=1.0,
            heuristic_guidance_online_advantage_clip=10.0,
        )
        network._den6_online_objective_baseline = __import__("collections").OrderedDict()

        def gradient(objective):
            log_prob = torch.tensor(0.0, requires_grad=True)
            network.last_actuator_voxel_state = {
                "den6_online_policy_log_prob": log_prob,
                "den6_online_policy_entropy": log_prob.new_zeros(()),
            }
            network.discrete_policy_loss(torch.tensor(float(objective))).backward()
            return float(log_prob.grad)

        self.assertLess(gradient(-4.0), 0.0)   # no-op 0%より改善
        self.assertGreater(gradient(-3.0), 0.0)  # 過去最良-4%より悪化
        self.assertLess(gradient(-4.5), 0.0)   # 過去最良を更新

    def test_den6_geometry_credit_updates_amount_only_inside_compression_guard(self):
        """圧縮改善を保ったGeometry改善だけを離散Amountへ返す。"""
        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            _current_input_file="fixture.ply",
            sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2,
            sparsepcgc_scale_m=8,
            heuristic_guidance_online_policy_weight=1.0,
            heuristic_guidance_online_entropy_weight=0.0,
            heuristic_guidance_online_reward_scale=1.0,
            heuristic_guidance_online_advantage_clip=10.0,
            heuristic_guidance_online_geometry_policy_weight=1.0,
            heuristic_guidance_online_geometry_compression_tolerance=0.25,
            heuristic_guidance_online_geometry_advantage_clip=1.0,
        )
        key = "fixture.ply|0|2|8"
        network._den6_online_objective_baseline = __import__("collections").OrderedDict(
            [(key, -4.0)]
        )
        network._den6_online_geometry_baseline = __import__("collections").OrderedDict(
            [(key, 0.01)]
        )
        amount_log_prob = torch.tensor(0.0, requires_grad=True)
        policy_log_prob = torch.tensor(0.0, requires_grad=True)
        network.last_actuator_voxel_state = {
            "den6_online_policy_log_prob": policy_log_prob,
            "den6_online_policy_entropy": policy_log_prob.new_zeros(()),
            "den6_online_amount_log_prob": amount_log_prob,
        }
        loss = network.discrete_policy_loss(
            torch.tensor(-4.0), geometry=torch.tensor(0.009)
        )
        amount_grad = torch.autograd.grad(loss, amount_log_prob, retain_graph=True)[0]
        self.assertLess(float(amount_grad), 0.0)
        self.assertTrue(network.last_discrete_policy_debug["geometry_policy_guard_passed"])

        rejected_amount_log_prob = torch.tensor(0.0, requires_grad=True)
        rejected_policy_log_prob = torch.tensor(0.0, requires_grad=True)
        network.last_actuator_voxel_state = {
            "den6_online_policy_log_prob": rejected_policy_log_prob,
            "den6_online_policy_entropy": rejected_policy_log_prob.new_zeros(()),
            "den6_online_amount_log_prob": rejected_amount_log_prob,
        }
        rejected = network.discrete_policy_loss(
            torch.tensor(-3.0), geometry=torch.tensor(0.008)
        )
        rejected_grad = torch.autograd.grad(
            rejected, rejected_amount_log_prob, allow_unused=True
        )[0]
        self.assertIsNone(rejected_grad)
        self.assertFalse(network.last_discrete_policy_debug["geometry_policy_guard_passed"])

    def test_den6_sequence_baseline_credits_new_frames_immediately(self):
        """未知frameが続いてもsequence EMAでActualとGeometryを比較する。"""
        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            _current_input_file="/dataset/longdress/frame_0001.ply",
            sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2,
            sparsepcgc_scale_m=8,
            heuristic_guidance_online_policy_weight=1.0,
            heuristic_guidance_online_entropy_weight=0.0,
            heuristic_guidance_online_reward_scale=1.0,
            heuristic_guidance_online_advantage_clip=10.0,
            heuristic_guidance_online_reward_ema=0.1,
            heuristic_guidance_online_geometry_policy_weight=1.0,
            heuristic_guidance_online_geometry_compression_tolerance=0.25,
            heuristic_guidance_online_geometry_advantage_clip=1.0,
        )
        network._den6_online_objective_baseline = __import__("collections").OrderedDict()
        network._den6_online_geometry_baseline = __import__("collections").OrderedDict()

        first_policy = torch.tensor(0.0, requires_grad=True)
        first_amount = torch.tensor(0.0, requires_grad=True)
        network.last_actuator_voxel_state = {
            "den6_online_policy_log_prob": first_policy,
            "den6_online_policy_entropy": first_policy.new_zeros(()),
            "den6_online_amount_log_prob": first_amount,
        }
        network.discrete_policy_loss(
            torch.tensor(-4.0), geometry=torch.tensor(0.010)
        )

        network.args._current_input_file = "/dataset/longdress/frame_0002.ply"
        second_policy = torch.tensor(0.0, requires_grad=True)
        second_amount = torch.tensor(0.0, requires_grad=True)
        network.last_actuator_voxel_state = {
            "den6_online_policy_log_prob": second_policy,
            "den6_online_policy_entropy": second_policy.new_zeros(()),
            "den6_online_amount_log_prob": second_amount,
        }
        second = network.discrete_policy_loss(
            torch.tensor(-4.0), geometry=torch.tensor(0.009)
        )
        amount_grad = torch.autograd.grad(second, second_amount)[0]
        self.assertLess(float(amount_grad), 0.0)
        self.assertEqual(
            network.last_discrete_policy_debug["objective_baseline_source"],
            "sequence",
        )
        self.assertEqual(
            network.last_discrete_policy_debug["geometry_policy_baseline_source"],
            "sequence",
        )

    def test_den6_sequence_baseline_reverses_gradient_for_worse_new_frame(self):
        """同一sequenceの未出frameでもActualの良化・悪化を同じ正例にしない。"""
        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            _current_input_file="/dataset/loot/frame_0001.ply",
            sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2,
            sparsepcgc_scale_m=8,
            heuristic_guidance_online_policy_weight=1.0,
            heuristic_guidance_online_entropy_weight=0.0,
            heuristic_guidance_online_reward_scale=1.0,
            heuristic_guidance_online_advantage_clip=10.0,
            heuristic_guidance_online_reward_ema=0.1,
            heuristic_guidance_online_geometry_policy_weight=0.0,
        )
        network._den6_online_objective_baseline = __import__("collections").OrderedDict()
        network.last_actuator_voxel_state = {
            "den6_online_policy_log_prob": torch.tensor(0.0, requires_grad=True),
            "den6_online_policy_entropy": torch.tensor(0.0),
        }
        network.discrete_policy_loss(torch.tensor(-4.0))

        network.args._current_input_file = "/dataset/loot/frame_0002.ply"
        worse_log_prob = torch.tensor(0.0, requires_grad=True)
        network.last_actuator_voxel_state = {
            "den6_online_policy_log_prob": worse_log_prob,
            "den6_online_policy_entropy": worse_log_prob.new_zeros(()),
        }
        worse_loss = network.discrete_policy_loss(torch.tensor(-3.0))
        worse_grad = torch.autograd.grad(worse_loss, worse_log_prob)[0]

        network.args._current_input_file = "/dataset/loot/frame_0003.ply"
        better_log_prob = torch.tensor(0.0, requires_grad=True)
        network.last_actuator_voxel_state = {
            "den6_online_policy_log_prob": better_log_prob,
            "den6_online_policy_entropy": better_log_prob.new_zeros(()),
        }
        better_loss = network.discrete_policy_loss(torch.tensor(-5.0))
        better_grad = torch.autograd.grad(better_loss, better_log_prob)[0]
        self.assertGreater(float(worse_grad), 0.0)
        self.assertLess(float(better_grad), 0.0)

    def test_reproduction_reference_matches_saved_mvub_plan(self):
        path = Path(__file__).resolve().parents[1] / "tools" / "ana_den6_reproduce.py"
        spec = importlib.util.spec_from_file_location("ana_den6_reproduce_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(module.REFERENCE_SHARES[("MVUB", 8)], (0.50, 0.40, 0.10))

    def test_train_accepts_den6_reproduction_until_hard_anchor_is_attached(self):
        structure = {"occupancy_nll_proxy": torch.zeros((1, 1, 4), dtype=torch.float32)}
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_reproduce",
            heuristic_guidance_enabled=True,
        )
        guidance = build_heuristic_guidance(structure, args)
        self.assertTrue(guidance["enabled"])

    def test_den6_manifest_reconstructs_verified_hard_voxel_set(self):
        """保存PLYではなくden6 candidate列から最終集合を再構築する。"""
        initial = torch.tensor(
            [[[0, 1], [0, 0], [0, 0]]], dtype=torch.long
        )
        expected = torch.tensor(
            [[[1, 2], [1, 0], [0, 0]]], dtype=torch.long
        )
        with tempfile.TemporaryDirectory() as temporary:
            input_file = Path(temporary) / "input.ply"
            input_file.write_bytes(b"synthetic canonical input")
            manifest_file = Path(temporary) / "plan.json"
            manifest_file.write_text(
                json.dumps(
                    {
                        "schema_version": "ana_den6_mixed_plan_manifest_v1",
                        "den6_sha256": _current_den6_sha256(),
                        "input_file": str(input_file),
                        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
                        "setting_id": "native_vs1_pq1_ae0_sr2_m8",
                        "selected_operation_counts": {"Add": 1, "Prune": 1, "Adjust": 1},
                        "selected_candidates": [
                            {"operation": "Prune", "remove_coords": [[0, 0, 0]], "add_coords": []},
                            {"operation": "Adjust", "remove_coords": [[1, 0, 0]], "add_coords": [[1, 1, 0]]},
                            {"operation": "Add", "remove_coords": [], "add_coords": [[2, 0, 0]]},
                        ],
                        "final_voxel_hash": _coord_hash(expected),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                heuristic_guidance_mode="ana_den6_reproduce",
                heuristic_guidance_den6_plan_manifest=str(manifest_file),
                _current_input_file=str(input_file),
                sparsepcgc_scale_ae=0,
                sparsepcgc_scale_sr=2,
                sparsepcgc_scale_m=8,
            )
            result = attach_ana_den6_reference_anchor(
                {"full_global_voxel_coords": initial}, args, device=torch.device("cpu")
            )
            other_input = Path(temporary) / "other.ply"
            other_input.write_bytes(b"different frame")
            args._current_input_file = str(other_input)
            with self.assertRaisesRegex(RuntimeError, "入力PLYが一致しない"):
                attach_ana_den6_reference_anchor(
                    {"full_global_voxel_coords": initial}, args, device=torch.device("cpu")
                )
        self.assertTrue(torch.equal(result["actual_oracle_override_final_voxel_coords"], expected))
        self.assertEqual(result["actual_oracle_override_add_count"], 1)
        self.assertEqual(result["actual_oracle_override_drop_count"], 1)
        self.assertEqual(result["actual_oracle_override_move_count"], 1)
        self.assertEqual(result["ana_den6_reference_anchor_source"], "ana_den6_candidate_plan_manifest")


    def test_den6_residual_uses_exact_ranked_pool_not_proxy_formula(self):
        """den6 v2 poolをsource/target候補へ写像し、proxy式へfallbackしない。"""
        coords = torch.tensor(
            [[[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=torch.long
        )
        exact = {
            "source": "ana_den6_exact_ranked_candidate_pool_v2",
            "manifest_sha256": "synthetic",
            "dataset": "8I",
            "scale_m": 8,
            "total_ratio": 0.0025,
            "operation_shares": {"Add": 0.4, "Prune": 0.4, "Adjust": 0.2},
            "operation_heuristics": {
                "Add": "geometry_safe_rate",
                "Prune": "subtree_collapse",
                "Adjust": "hotspot_cluster",
            },
            "ranked_candidate_pools": {
                "Prune": [
                    {"operation": "Prune", "pool_rank": 0, "remove_coords": [[0, 0, 0]], "add_coords": []},
                    {"operation": "Prune", "pool_rank": 1, "remove_coords": [[1, 0, 0]], "add_coords": []},
                ],
                "Add": [
                    {"operation": "Add", "pool_rank": 0, "remove_coords": [], "add_coords": [[1, 1, 0]]},
                ],
                "Adjust": [
                    {"operation": "Adjust", "pool_rank": 0, "remove_coords": [[0, 1, 0]], "add_coords": [[0, 1, 1]]},
                ],
            },
        }
        structure = {
            "occupancy_nll_proxy": torch.zeros((1, 1, 3), dtype=torch.float32),
            "global_voxel_coords": coords,
            "ana_den6_ranked_candidate_guidance": exact,
        }
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_residual",
            heuristic_guidance_enabled=True,
            dataname="8i",
            sparsepcgc_scale_m=8,
            _current_subtree_id="",
            heuristic_guidance_tensor_cache_entries=2,
        )
        guidance = build_heuristic_guidance(structure, args)
        self.assertEqual(
            guidance["formula_basis"],
            "ana_den6_exact_ranked_editcandidate_pool_v2",
        )
        self.assertEqual(int(guidance["candidate_mask"]["Prune"].sum()), 2)
        self.assertGreater(int(guidance["candidate_mask"]["Add"].sum()), 0)
        self.assertEqual(int(guidance["candidate_mask"]["Adjust"].sum()), 1)
        self.assertIn("target_direction_sparse", guidance)

        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        actuator.args = args
        actuator.neighbor_offsets = torch.tensor(
            [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ],
            dtype=torch.float32,
        )
        direction = actuator._fit_heuristic_target_direction_prior(
            guidance, "Adjust", structure["occupancy_nll_proxy"]
        )
        self.assertEqual(tuple(direction.shape), (1, 26, 3))
        self.assertEqual(int((direction > 0).sum()), 1)

    def test_den6_residual_rejects_missing_exact_candidate_pool(self):
        structure = {"occupancy_nll_proxy": torch.zeros((1, 1, 4), dtype=torch.float32)}
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_residual",
            heuristic_guidance_enabled=True,
        )
        with self.assertRaisesRegex(RuntimeError, "exact candidate guidance"):
            build_heuristic_guidance(structure, args)

    def test_den6_online_rejects_legacy_v5_candidate_shortlist(self):
        """online modeへ旧Poolを直接注入する抜け道も拒否する。"""
        coords = torch.tensor(
            [[[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=torch.long
        )
        exact = {
            "source": "ana_den6_exact_compact_candidate_shortlist_online_v5",
            "manifest_sha256": "synthetic-v5",
            "dataset": "8i",
            "scale_m": 8,
            "total_ratio": 0.0025,
            "operation_shares": {"Add": 0.4, "Prune": 0.4, "Adjust": 0.2},
            "operation_candidate_shortlists": {
                "Prune": [{"operation": "Prune", "pool_rank": 0, "remove_coords": [[0, 0, 0]], "add_coords": []}],
                "Add": [{"operation": "Add", "pool_rank": 0, "remove_coords": [], "add_coords": [[1, 1, 0]]}],
                "Adjust": [{"operation": "Adjust", "pool_rank": 0, "remove_coords": [[0, 1, 0]], "add_coords": [[0, 1, 1]]}],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "candidate Pool"):
            build_heuristic_guidance(
                {
                    "occupancy_nll_proxy": torch.zeros((1, 1, 3), dtype=torch.float32),
                    "global_voxel_coords": coords,
                    "ana_den6_ranked_candidate_guidance": exact,
                },
                SimpleNamespace(
                    heuristic_guidance_mode="ana_den6_online",
                    heuristic_guidance_enabled=True,
                    dataname="8i",
                    sparsepcgc_scale_m=8,
                    _current_subtree_id="",
                    heuristic_guidance_tensor_cache_entries=2,
                ),
            )

    def test_den6_exact_residual_hard_plan_matches_anchor_when_residual_is_zero(self):
        """Network残差0ではden6のpool順・衝突回避・最終Voxel集合を再現する。"""
        coords = torch.tensor(
            [[[0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]],
            dtype=torch.long,
        )
        expected = torch.tensor(
            [[[0, 1, 0, 1, 2, 2], [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 1]]],
            dtype=torch.long,
        )
        exact = {
            "source": "ana_den6_exact_ranked_candidate_pool_v2",
            "manifest_sha256": "fixture",
            "dataset": "8I",
            "scale_m": 8,
            # UVGと同様、既定の8i学習bin外にある0.50% anchorを再生する。
            "total_ratio": 0.005,
            "operation_shares": {"Add": 0.4, "Prune": 0.4, "Adjust": 0.2},
            "operation_heuristics": {},
            "operation_priority": ["Add", "Prune", "Adjust"],
            "plan_variants": 6,
            "selected_operation_counts": {"Add": 1, "Prune": 1, "Adjust": 1},
            "anchor_operation_counts": {"Add": 1, "Prune": 1, "Adjust": 1},
            "final_voxel_hash": _coord_hash(expected),
            "ranked_candidate_pools": {
                "Add": [{"candidate_id": "a0", "operation": "Add", "pool_rank": 0, "remove_coords": [], "add_coords": [[2, 0, 1]]}],
                "Prune": [
                    {"candidate_id": "p0", "operation": "Prune", "pool_rank": 0, "remove_coords": [[0, 0, 0]], "add_coords": []},
                    {"candidate_id": "p1", "operation": "Prune", "pool_rank": 1, "remove_coords": [[0, 1, 0]], "add_coords": []},
                ],
                "Adjust": [{"candidate_id": "m0", "operation": "Adjust", "pool_rank": 0, "remove_coords": [[1, 0, 0]], "add_coords": [[2, 0, 0]]}],
            },
        }
        like = torch.zeros((1, 1, coords.shape[-1]), dtype=torch.float32)
        guidance = build_heuristic_guidance(
            {
                "occupancy_nll_proxy": like,
                "global_voxel_coords": coords,
                "ana_den6_ranked_candidate_guidance": exact,
            },
            SimpleNamespace(
                heuristic_guidance_mode="ana_den6_residual",
                heuristic_guidance_enabled=True,
                dataname="8i",
                sparsepcgc_scale_m=8,
                _current_subtree_id="",
                heuristic_guidance_tensor_cache_entries=2,
            ),
        )
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        actuator.args = SimpleNamespace(
            _global_train_step=0,
            heuristic_guidance_anchor_steps=200,
            heuristic_guidance_network_residual_weight=0.5,
        )
        result = actuator._build_exact_den6_residual_plan(
            guidance,
            coords,
            torch.tensor([[[0.0010]]]),
            torch.tensor([[[0.0010]]]),
            torch.tensor([[[0.0005]]]),
            torch.zeros((1, 1, coords.shape[-1])),
            torch.zeros((1, 1, coords.shape[-1])),
            torch.zeros((1, 26, coords.shape[-1])),
            torch.zeros((1, coords.shape[-1], 26)),
        )
        self.assertIsNotNone(result)
        final_coords, debug = result
        self.assertEqual(debug["selected_counts"], {"Add": 1, "Prune": 1, "Adjust": 1})
        self.assertAlmostEqual(debug["amount_bin_ratio"], 0.005, places=9)
        self.assertEqual(_coord_hash(final_coords), _coord_hash(expected))
        expected_sorted = torch.unique(
            expected[0].transpose(0, 1), dim=0, sorted=True
        ).transpose(0, 1).unsqueeze(0)
        self.assertTrue(torch.equal(final_coords, expected_sorted))
        self.assertTrue(debug["static_candidate_compatibility_used"])
        shadow_teacher = debug["single_plan_shadow_teacher"]
        self.assertEqual(shadow_teacher["plan_key"], debug["plan_hash"])
        self.assertEqual(len(shadow_teacher["candidates"]), 3)
        self.assertEqual(
            {row["operation"] for row in shadow_teacher["candidates"]},
            {"Add", "Prune", "Adjust"},
        )

        # anchor後は3つの既存Amount headの出力を共通残差へ潰さず、
        # operation別の残差として保持する。
        actuator.args._global_train_step = 1000
        actuator.args.heuristic_guidance_online_amount_log_sigma = 0.0
        actuator.args.heuristic_guidance_online_amount_residual_scale = 0.10
        add_ratio = torch.tensor([[[0.0018]]], requires_grad=True)
        prune_ratio = torch.tensor([[[0.0007]]], requires_grad=True)
        adjust_ratio = torch.tensor([[[0.0003]]], requires_grad=True)
        residual_result = actuator._build_exact_den6_residual_plan(
            guidance,
            coords,
            add_ratio,
            prune_ratio,
            adjust_ratio,
            torch.zeros((1, 1, coords.shape[-1])),
            torch.zeros((1, 1, coords.shape[-1])),
            torch.zeros((1, 26, coords.shape[-1])),
            torch.zeros((1, coords.shape[-1], 26)),
        )
        self.assertIsNotNone(residual_result)
        operation_residuals = residual_result[1]["operation_amount_mean_log_residuals"]
        self.assertEqual(set(operation_residuals), {"Add", "Prune", "Adjust"})
        self.assertEqual(len({round(value, 7) for value in operation_residuals.values()}), 3)
        operation_gradients = torch.autograd.grad(
            residual_result[1]["policy_log_prob"],
            (add_ratio, prune_ratio, adjust_ratio),
        )
        self.assertTrue(all(torch.isfinite(value).all() for value in operation_gradients))
        self.assertTrue(all(float(value.abs().sum()) > 0.0 for value in operation_gradients))
        self.assertEqual(
            len({round(float(value.reshape(-1)[0]), 7) for value in operation_gradients}),
            3,
        )

        # 固定validationで許可されたresidual weight拡大により、同じPool内で
        # den6順位差をNetwork scoreが逆転できることを確認する。
        prune_map = guidance["candidate_tensor_map"]["Prune"]
        prune_map["rank_score"] = torch.tensor([0.55, 0.45])
        prune_map["source_index"] = torch.tensor([0, 2], dtype=torch.long)
        drop_preference = torch.zeros((1, 1, coords.shape[-1]))
        drop_preference[0, 0, 0] = -10.0
        drop_preference[0, 0, 2] = 10.0
        actuator.args.heuristic_guidance_network_residual_weight = 0.05
        actuator.args.heuristic_guidance_network_residual_weight_max = 0.25
        actuator.args._heuristic_guidance_network_residual_weight_current = 0.05
        low_autonomy = actuator._build_exact_den6_residual_plan(
            guidance,
            coords,
            add_ratio,
            prune_ratio,
            adjust_ratio,
            drop_preference,
            torch.zeros((1, 1, coords.shape[-1])),
            torch.zeros((1, 26, coords.shape[-1])),
            torch.zeros((1, coords.shape[-1], 26)),
        )
        actuator.args._heuristic_guidance_network_residual_weight_current = 0.25
        high_autonomy = actuator._build_exact_den6_residual_plan(
            guidance,
            coords,
            add_ratio,
            prune_ratio,
            adjust_ratio,
            drop_preference,
            torch.zeros((1, 1, coords.shape[-1])),
            torch.zeros((1, 26, coords.shape[-1])),
            torch.zeros((1, coords.shape[-1], 26)),
        )
        self.assertIn("p0", low_autonomy[1]["selected_candidate_ids"])
        self.assertIn("p1", high_autonomy[1]["selected_candidate_ids"])
        self.assertNotEqual(low_autonomy[1]["plan_hash"], high_autonomy[1]["plan_hash"])

    def test_den6_anchor_amounts_are_operation_specific_at_step_zero(self):
        """8i m=8の0.25%を旧5% Prune候補で上書きしない。"""
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        actuator.args = SimpleNamespace(
            _global_train_step=0,
            heuristic_guidance_anchor_steps=200,
            heuristic_guidance_amount_residual_fraction=0.50,
            heuristic_guidance_amount_min_residual=0.0001,
            heuristic_guidance_amount_grad_scale=1.0,
        )
        guidance = {
            "amount_prior": {"Add": 0.0010, "Prune": 0.0010, "Adjust": 0.0005},
        }
        network_ratio = torch.tensor([[[0.05]]], dtype=torch.float32, requires_grad=True)
        values = {
            operation: actuator._apply_heuristic_amount_guidance(
                network_ratio, guidance, operation, 0.30
            )
            for operation in ("Add", "Prune", "Adjust")
        }
        self.assertAlmostEqual(float(values["Add"].detach()), 0.0010, places=7)
        self.assertAlmostEqual(float(values["Prune"].detach()), 0.0010, places=7)
        self.assertAlmostEqual(float(values["Adjust"].detach()), 0.0005, places=7)
        values["Prune"].backward()
        self.assertTrue(torch.isfinite(network_ratio.grad).all())
        self.assertGreater(float(network_ratio.grad.abs().sum()), 0.0)

    def test_den6_online_amounts_remain_nonzero_after_old_200_step_boundary(self):
        """旧bootstrap境界を越えても3操作のAmount priorと勾配を消さない。"""
        guidance = {
            "amount_prior": {"Add": 0.0010, "Prune": 0.0010, "Adjust": 0.0005},
        }
        for step in (120, 200, 201, 600, 1000):
            actuator = StructureRepairActuator.__new__(StructureRepairActuator)
            actuator.args = SimpleNamespace(
                heuristic_guidance_mode="ana_den6_online",
                _global_train_step=step,
                _total_train_steps_estimate=1000,
            )
            for operation in ("Add", "Prune", "Adjust"):
                network_ratio = torch.tensor(
                    [[[0.15]]], dtype=torch.float32, requires_grad=True
                )
                value = actuator._apply_heuristic_amount_guidance(
                    network_ratio, guidance, operation, 0.30
                )
                self.assertGreater(float(value.detach()), 0.0)
                self.assertLessEqual(float(value.detach()), 0.00401)
                value.backward()
                self.assertTrue(torch.isfinite(network_ratio.grad).all())
                self.assertGreater(float(network_ratio.grad.abs().sum()), 0.0)

    def test_full_cloud_static_node_input_cache_reuses_detached_state(self):
        """Episode cache must reuse only fixed Node/Voxel inputs, never a graph."""
        network = Network.__new__(Network)
        network.args = SimpleNamespace(
            cache_max_entries=2,
            cache_max_memory_mb=1,
            static_node_cache_cpu=False,
            qs=2,
            sparsepcgc_voxel_size=1.0,
            sparsepcgc_pos_quantscale=1,
            sparsepcgc_quant_mode="round_voxel_then_pos",
            sparsepcgc_dequantize_center=False,
            fused_feat_dim=4,
            out_dim=4,
        )
        network.cache_enabled = True
        network.input_cache = __import__("collections").OrderedDict()
        network._input_cache_bytes = 0
        state = {
            "voxel_coords": torch.zeros((1, 3, 2), dtype=torch.long),
            "node_xyz": torch.zeros((1, 3, 2), dtype=torch.float32),
            "node_features": torch.ones((1, 4, 2), dtype=torch.float32, requires_grad=True),
            "node_mask": torch.ones((1, 2), dtype=torch.bool),
            "node_counts": torch.tensor([2], dtype=torch.long),
            "restore_meta": {"global_qs": torch.tensor([1.0])},
            "restore_info": {"restore_output_points": 2},
            "source": "full_octree_context",
        }
        network._put_static_node_cache("frame.ply", "full_octree_context", state)
        cached = network._get_static_node_cache(
            "frame.ply", "full_octree_context", torch.device("cpu")
        )
        self.assertIsNotNone(cached)
        self.assertFalse(cached["node_features"].requires_grad)
        self.assertIs(cached, network._get_static_node_cache(
            "frame.ply", "full_octree_context", torch.device("cpu")
        ))
        stats = network.input_cache_stats()
        self.assertEqual(stats["entries"], 1)
        self.assertGreater(stats["bytes"], 0)

    def test_chunked_neighbor_occupancy_is_bitwise_exact(self):
        """Chunking must query the same 26 coordinates and return identical means."""
        torch.manual_seed(7)
        coords = torch.randint(-9, 10, (257, 3), dtype=torch.long)
        coords[100:130] = coords[:30]
        args = SimpleNamespace(
            compress="SparsePCGC",
            sparsepcgc_effective_qs=1.0,
            sparsepcgc_voxel_size=1.0,
            sparsepcgc_pos_quantscale=1,
            octree_ctx_level=5,
            octree_ctx_dim=8,
            structure_geo_k=8,
            structure_geo_max_points=2048,
            proxy_max_depth=12,
            octree_diag_levels="4,6,8,10,12",
            structure_neighbor_query_chunk=7,
        )
        module = OctreeStructureAnalysis(args)
        unique_coords = torch.unique(coords, dim=0, sorted=True)
        targets = coords[:, None, :] + module.neighbor_offsets.view(1, -1, 3)
        expected = module._coords_membership(
            targets.reshape(-1, 3), unique_coords
        ).view(coords.shape[0], -1).float().mean(dim=1)
        actual = module._neighbor_occupancy_chunked(coords, unique_coords)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
