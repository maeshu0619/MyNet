import ast
import gzip
import inspect
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import models.network as network_module
from models.modules.network_k_proposal_policy import NetworkKProposalPolicy
from models.utils.loss.compression import CompressionLossMixin
from models.utils.loss.k_proposal_distillation import (
    KProposalSetLoss,
    OfflineKProposalTeacherStore,
)


def _args(**overrides):
    values = dict(
        sparsepcgc_scale_ae=0,
        sparsepcgc_scale_sr=2,
        sparsepcgc_scale_m=8,
        sparsepcgc_voxel_size=1.0,
        sparsepcgc_pos_quantscale=1.0,
        sparsepcgc_psnr_resolution=1023,
        sparsepcgc_native_bit_depth=10,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class NetworkKProposalPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.policy = NetworkKProposalPolicy(
            25, hidden_dim=16, proposal_count=8, shortlist_size=256
        ).eval()
        self.features = torch.randn(1, 25, 4096)
        self.fixed = torch.rand(1, 6, 4096)
        self.coords = torch.randint(0, 512, (1, 3, 4096))

    def _forward(self):
        return self.policy(
            self.features, _args(), training=False,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )

    def test_inference_module_has_no_forbidden_dependency(self):
        source = inspect.getsource(NetworkKProposalPolicy)
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        forbidden = ("den4", "den5", "den6", "teacher", "cache", "actual_encoder")
        self.assertFalse(any(any(term in name for term in forbidden) for name in imports))

    def test_missing_fixed_features_fails_closed(self):
        with self.assertRaises(RuntimeError):
            self.policy(
                self.features, _args(), training=False,
                fixed_features=None, voxel_coords=self.coords,
            )

    def test_training_teacher_source_augments_but_preserves_natural_shortlist(self):
        teacher_coord = self.coords[0, :, 3000].tolist()
        output = self.policy(
            self.features,
            _args(_network_k_training_teacher_coords=[teacher_coord]),
            training=True,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )
        self.assertEqual(output["natural_shortlist_indices"].shape[1], 256)
        self.assertTrue(bool((output["shortlist_indices"] == 3000).any()))

    def test_inference_rejects_teacher_shortlist(self):
        with self.assertRaises(RuntimeError):
            self.policy(
                self.features,
                _args(_network_k_training_teacher_coords=[[0, 0, 0]]),
                training=False,
                fixed_features=self.fixed,
                voxel_coords=self.coords,
            )

    def test_add_target_set_logits_use_reachable_sources_without_source_label(self):
        target = (self.coords[0, :, 3000] + torch.tensor([1, 0, 0])).tolist()
        output = self.policy(
            self.features,
            _args(
                network_k_offline_dataset="offline-only",
                _network_k_training_teacher_target_coords=[target],
            ),
            training=True,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )
        self.assertEqual(tuple(output["slot_target_logits"].shape), (1, 8, 3, 1))
        self.assertTrue(torch.isfinite(output["slot_target_logits"][:, :, 1]).all())
        output["slot_target_logits"][:, :, 1].sum().backward()
        self.assertGreater(float(self.policy.shared_direction_head.weight.grad.abs().sum()), 0.0)

    def test_deterministic_specialized_slots_and_counter_contract(self):
        first = self._forward()
        second = self._forward()
        for key in ("slot_logits", "total_ratio", "shares", "priorities", "critic_score"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(torch.equal(first["selected_slot"], second["selected_slot"]))
        masks = first["compact_plans"]["selected_shortlist_mask"][0]
        hashes = {bytes(mask.detach().cpu().numpy()) for mask in masks}
        self.assertGreater(len(hashes), 1)
        expected = {
            "shared_encoder_forward_count": 1,
            "shared_basis_forward_count": 1,
            "proposal_count": 8,
            "critic_batch_count": 1,
            "selected_plan_count": 1,
            "den6_call_count": 0,
            "cache_reference_count": 0,
            "teacher_reference_count": 0,
            "sparsepcgc_probe_count": 0,
            "candidate_actual_encode_count": 0,
        }
        for key, value in expected.items():
            self.assertEqual(first[key], value, key)

    def test_one_token_changes_its_mode_not_other_slots(self):
        before = self._forward()
        with torch.no_grad():
            self.policy.plan_tokens[3].add_(2.0)
        after = self._forward()
        for key in ("ratio_logits", "shares", "order_logits", "variant_logits"):
            unchanged = [
                torch.equal(before[key][:, slot], after[key][:, slot])
                for slot in range(8) if slot != 3
            ]
            self.assertTrue(all(unchanged), key)
            self.assertFalse(torch.equal(before[key][:, 3], after[key][:, 3]), key)

    def test_codec_conditioning_changes_proposal_set(self):
        first = self._forward()["slot_logits"]
        second = self.policy(
            self.features,
            _args(sparsepcgc_scale_ae=1, sparsepcgc_scale_sr=0),
            training=False,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )["slot_logits"]
        self.assertFalse(torch.equal(first, second))

    def test_ratio_and_order_theta_change_executable_plan_hash(self):
        kwargs = dict(
            training=False, fixed_features=self.fixed, voxel_coords=self.coords,
        )
        before = self.policy(
            self.features, _args(network_k_debug_plan_hash=True), **kwargs
        )
        before_hash = before["executable_plans"]["plan_hash"][0][0]
        with torch.no_grad():
            self.policy.slot_ratio_bias[0].fill_(-8.0)
            self.policy.slot_ratio_bias[0, 4] = 8.0
            self.policy.slot_order_bias[0].fill_(-8.0)
            self.policy.slot_order_bias[0, 5] = 8.0
        after = self.policy(
            self.features, _args(network_k_debug_plan_hash=True), **kwargs
        )
        self.assertNotEqual(before_hash, after["executable_plans"]["plan_hash"][0][0])

    def test_inactive_threshold_temperature_and_enable_heads_are_removed(self):
        for name in ("threshold_head", "temperature_head", "enable_head", "priority_head"):
            self.assertFalse(hasattr(self.policy, name), name)

    def test_legacy_k_checkpoint_dead_heads_do_not_break_non_strict_load(self):
        state = dict(self.policy.state_dict())
        hidden = self.policy.amount_head.weight.shape[1]
        state["amount_head.weight"] = torch.zeros(1, hidden)
        state["amount_head.bias"] = torch.zeros(1)
        for name in ("enable_head", "priority_head", "threshold_head", "temperature_head"):
            state[name + ".weight"] = torch.zeros(3, hidden)
            state[name + ".bias"] = torch.zeros(3)
        self.policy.load_state_dict(state, strict=False)

    def test_selected_amount_matches_collision_resolved_compact_counts(self):
        output = self._forward()
        selected = int(output["selected_slot"][0])
        expected = output["compact_plans"]["accepted_count"][0, selected].long()
        terms = output["selected_policy_terms"]
        actual = torch.round(
            terms["total_ratio"][0, 0, 0]
            * terms["shares"][0, :, 0]
            * float(self.features.shape[-1])
        ).long()
        self.assertTrue(torch.equal(actual, expected), (actual, expected))

    def test_forbidden_runtime_functions_are_not_reached(self):
        error = RuntimeError("forbidden path reached")
        with mock.patch.object(network_module, "attach_ana_den6_online_guidance", side_effect=error), \
             mock.patch.object(network_module, "attach_ana_den6_reference_anchor", side_effect=error), \
             mock.patch.object(network_module, "build_heuristic_guidance", side_effect=error):
            result = self._forward()
        self.assertEqual(result["proposal_count"], 8)

    def test_actual_reward_reverses_k_critic_selection_gradient(self):
        self.policy.train()
        output = self.policy(
            self.features, _args(), training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        log_probability = output["critic_selection_log_prob"]
        parameter = self.policy.critic_gain_head.weight
        improving = torch.autograd.grad(
            -(+1.0) * log_probability, parameter, retain_graph=True
        )[0]
        worsening = torch.autograd.grad(-(-1.0) * log_probability, parameter)[0]
        self.assertGreater(float(improving.abs().sum()), 0.0)
        self.assertTrue(torch.allclose(improving, -worsening, atol=1e-7, rtol=1e-5))

    def test_all_actual_exploration_updates_each_theta_family_without_teacher(self):
        self.policy.train()
        output = self.policy(
            self.features,
            _args(
                network_k_all_actual_enabled=True,
                network_k_all_actual_temperature=1.0,
                network_k_offline_dataset="参照してはいけない",
            ),
            training=True,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )
        self.assertTrue(output["all_actual_exploration"])
        self.assertIsNone(output["slot_direction_logits"])
        advantage = torch.linspace(-1.0, 1.0, 8).view(1, 8)
        loss = -(advantage * output["slot_policy_log_prob"]).mean()
        loss.backward()
        groups = {
            "ratio": self.policy.amount_head.weight.grad,
            "share": self.policy.share_head.weight.grad,
            "order": self.policy.order_head.weight.grad,
            "variant": self.policy.variant_head.weight.grad,
            "coefficient": self.policy.coefficient_head.weight.grad,
            "where": self.policy.shared_basis_head.weight.grad,
            "direction": self.policy.direction_delta_head.weight.grad,
        }
        for name, gradient in groups.items():
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.detach().abs().sum()), 0.0, name)

    def test_all_actual_flag_does_not_make_inference_random(self):
        args = _args(network_k_all_actual_enabled=True)
        first = self.policy(
            self.features, args, training=False,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        second = self.policy(
            self.features, args, training=False,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        for key in ("ratio_class", "shares", "order_class", "variant_class", "slot_logits"):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_all_actual_relative_reward_has_correct_actor_sign(self):
        log_prob = torch.zeros(1, 8, requires_grad=True)
        predicted_gain = torch.zeros(1, 8, 1, requires_grad=True)
        dummy = SimpleNamespace(
            training=True,
            args=_args(
                heuristic_guidance_mode="network_k_proposal_policy",
                network_k_all_actual_enabled=True,
            ),
            last_k_proposal_terms={
                "slot_policy_log_prob": log_prob,
                "predicted_plan_gain": predicted_gain,
                "slot_policy_entropy": torch.ones(1, 8, requires_grad=True),
                "selected_slot": torch.zeros(1, dtype=torch.long),
            },
            last_k_all_actual_debug={},
        )
        compression_loss = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, 1.5, 0.5, -0.5]])
        total = network_module.Network.k_proposal_all_actual_loss(dummy, compression_loss)
        actor_grad = torch.autograd.grad(total, log_prob, retain_graph=True)[0]
        # 最も改善したslotは勾配降下でlog-probが増え、最悪slotは減る。
        self.assertLess(float(actor_grad[0, 0]), 0.0)
        self.assertGreater(float(actor_grad[0, 4]), 0.0)
        self.assertTrue(total.requires_grad)

    def test_set_loss_updates_proposal_and_critic_heads(self):
        self.policy.train()
        output = self.policy(
            self.features, _args(), training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        descriptor = output["compact_plans"]["descriptor"].detach()
        teacher = {
            "mode_descriptor": descriptor.roll(1, dims=1),
            "actual_gain": torch.linspace(-0.5, 4.0, 8).view(1, 8),
            "geometry": torch.linspace(0.1, 0.8, 8).view(1, 8),
            "mode_mask": torch.ones(1, 8, dtype=torch.bool),
            "voxel_relative_value": torch.rand(1, 3, output["slot_logits"].shape[-1]),
            "theta": {
                "ratio_class": output["ratio_class"].detach().roll(1, dims=1),
                "total_ratio": output["total_ratio"].detach().squeeze(-1).roll(1, dims=1),
                "share": output["shares"].detach().roll(1, dims=1),
                "order_class": output["order_class"].detach().roll(1, dims=1),
                "variant_class": torch.zeros(1, 8, dtype=torch.long),
                "mask": torch.ones(1, 8, dtype=torch.bool),
            },
        }
        losses = KProposalSetLoss()(output, teacher)
        self.assertTrue(losses["total"].requires_grad)
        losses["total"].backward()
        groups = {
            "token": self.policy.plan_tokens.grad,
            "amount": self.policy.amount_head.weight.grad,
            "share": self.policy.share_head.weight.grad,
            "where": self.policy.shared_basis_head.weight.grad,
            "critic": self.policy.critic_gain_head.weight.grad,
        }
        for name, gradient in groups.items():
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.detach().abs().sum()), 0.0, name)
        self.assertEqual(set(losses["raw"]), set(KProposalSetLoss.DEFAULT_WEIGHTS))

    def test_offline_teacher_join_is_explicit_and_add_target_is_not_fabricated(self):
        output = self._forward()
        descriptor = output["compact_plans"]["descriptor"][0, 0].tolist()
        shortlist_index = int(output["shortlist_indices"][0, 0])
        coord = self.coords[0, :, shortlist_index].tolist()
        mode = {
            "plan_key": "diagnostic",
            "descriptor": descriptor,
            "actual_gain_percent": 2.0,
            "geometry": {"D1_loss_db": 0.2, "D2_loss_db": 0.3},
            "member_count": 1,
        }
        payload = {
            "schema_version": "mynet_kproposal_mode_dataset_v1",
            "offline_only": True,
            "voxel_target_semantics": "rank_weighted_relative_value_not_causal_gain",
            "split": {"train": "sha|setting"},
            "states": {
                "sha|setting": {
                    "mode_medoids": [mode],
                    "voxel_relative_values": [
                        {"operation": "Prune", "coord": coord, "rank_weighted_relative_value": 0.9},
                        {"operation": "Add", "coord": coord, "rank_weighted_relative_value": 0.9},
                    ],
                }
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json.gz") as stream:
            with gzip.open(stream.name, "wt", encoding="utf-8") as compressed:
                json.dump(payload, compressed)
            store = OfflineKProposalTeacherStore(stream.name)
            teacher = store.teacher_for_output(
                "sha|setting", output, self.coords, split="train"
            )
        self.assertTrue(bool(teacher["voxel_value_mask"][0, 0].any()))
        self.assertFalse(bool(teacher["voxel_value_mask"][0, 1].any()))
        self.assertFalse(teacher["add_where_teacher_available"])
        self.assertEqual(teacher["voxel_target_semantics"], "rank_weighted_relative_value_not_causal_gain")


class KAllActualEvaluatorTest(unittest.TestCase):
    def test_evaluates_exactly_k_completed_network_plans_and_reuses_selected(self):
        class DummyLoss(CompressionLossMixin):
            def __init__(self):
                self.encoded = 0

            def _get_cached_actual_gt(self, cache_key):
                return {"bit": 1000.0}

            def _encode_actual_many(self, args, xyz_list):
                self.encoded += len(xyz_list)
                return [
                    {"bit": 1000.0 - 10.0 * index, "point_count": int(xyz.shape[-1]),
                     "actual_finished": True}
                    for index, xyz in enumerate(xyz_list)
                ]

            def _voxel_state_to_codec_xyz(self, args, voxel_state, like_xyz):
                return like_xyz, {"used": True}

        class Plan:
            operation_order = torch.zeros(1, 8, 3, dtype=torch.long)

        dummy = DummyLoss()
        args = _args(
            heuristic_guidance_mode="network_k_proposal_policy",
            network_k_all_actual_enabled=True,
            network_k_proposal_count=8,
        )
        proposal = {
            "executable_plan_batch": Plan(),
            "selected_slot": torch.tensor([3]),
            "predicted_plan_gain": torch.zeros(1, 8, 1),
        }
        voxel_state = {"initial_voxel_coords": torch.zeros(1, 3, 4, dtype=torch.long)}
        with mock.patch(
            "models.utils.loss.compression.apply_selected_executable_plan",
            return_value=(
                torch.zeros(1, 3, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.bool),
            ),
        ):
            result = dummy.evaluate_network_k_plans_actual(
                args,
                proposal_output=proposal,
                voxel_state=voxel_state,
                gt_xyz=torch.zeros(1, 3, 4),
                cache_key="state",
            )
        self.assertEqual(dummy.encoded, 8)
        self.assertEqual(result["edited_actual_encode_count"], 8)
        self.assertEqual(result["proposal_actual_encode_count"], 8)
        self.assertEqual(result["candidate_actual_encode_count"], 0)
        self.assertEqual(result["baseline_actual_encode_count"], 0)
        self.assertTrue(result["baseline_scalar_cache_hit"])
        self.assertEqual(result["selected_stats"]["bit"], 970.0)
        self.assertEqual(tuple(result["actual_compression_percent"].shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
