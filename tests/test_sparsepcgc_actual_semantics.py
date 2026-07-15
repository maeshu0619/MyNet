"""SparsePCGC actual teacherの符号・no-op回帰テスト。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import tempfile
import unittest

import torch

from models.modules.heuristic_guidance import build_heuristic_guidance, resolve_profile
from models.modules.structure_actuator import StructureRepairActuator
from models.utils.loss.compression import CompressionLossMixin
from models.utils.pointcloud.ana_den6_reference import (
    _coord_hash,
    _current_den6_sha256,
    attach_ana_den6_reference_anchor,
)


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


if __name__ == "__main__":
    unittest.main()
