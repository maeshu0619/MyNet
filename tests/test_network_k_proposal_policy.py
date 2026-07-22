import ast
import gzip
import inspect
import json
import tempfile
import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

import models.network as network_module
from models.modules.network_k_proposal_policy import NetworkKProposalPolicy
from models.utils.loss.compression import CompressionLossMixin
from models.utils.loss.k_proposal_distillation import (
    KProposalSetLoss,
    OfflineKProposalTeacherStore,
)
from models.utils.surrogate.pretrain import _build_surrogate_pretrain_canonical_context


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
        self.fixed = torch.rand(1, 17, 4096)
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

    def test_old_k_checkpoint_keeps_existing_trunk_shapes(self):
        state = {
            key: value.clone()
            for key, value in self.policy.state_dict().items()
            if not key.startswith("fixed_codec_basis_head.")
        }
        restored = NetworkKProposalPolicy(
            25, hidden_dim=16, proposal_count=8, shortlist_size=256
        )
        incompatible = restored.load_state_dict(state, strict=False)
        self.assertIn(
            "fixed_codec_basis_head.weight", incompatible.missing_keys
        )
        self.assertTrue(torch.equal(
            restored.shared_local_trunk[0].weight,
            self.policy.shared_local_trunk[0].weight,
        ))
        self.assertTrue(torch.equal(
            restored.shared_global_trunk[0].weight,
            self.policy.shared_global_trunk[0].weight,
        ))

    def test_octree_pattern_channels_seed_operation_specific_codec_basis(self):
        policy = NetworkKProposalPolicy(
            25, hidden_dim=16, proposal_count=8, shortlist_size=256
        ).eval()
        with torch.no_grad():
            for parameter in policy.shared_local_trunk.parameters():
                parameter.zero_()
            policy.shared_basis_head.weight.zero_()
            policy.shared_basis_head.bias.zero_()
        fixed = torch.zeros_like(self.fixed)
        fixed[:, 6] = 0.75
        output = policy(
            self.features, _args(), training=False,
            fixed_features=fixed, voxel_coords=self.coords,
        )
        prune_direct = output["shared_basis"][:, 0, 0]
        add_direct = output["shared_basis"][:, 1, 0]
        self.assertTrue(torch.allclose(prune_direct, torch.full_like(prune_direct, 0.1875)))
        self.assertTrue(torch.allclose(add_direct, torch.zeros_like(add_direct)))

    def test_where_policy_logits_are_scale_normalized_without_rank_change(self):
        output = self._forward()
        policy_logits = output["policy_base_slot_logits"]
        self.assertTrue(torch.isfinite(policy_logits).all())
        self.assertLess(float(policy_logits.mean(dim=3).abs().max()), 1e-4)
        self.assertGreater(float(output["where_policy_entropy"].mean()), 0.1)
        raw_order = output["slot_logits"].argsort(dim=3)
        normalized_order = policy_logits.argsort(dim=3)
        self.assertTrue(torch.equal(raw_order, normalized_order))

    def test_categorical_logits_remain_explorable_after_raw_scale_growth(self):
        with torch.no_grad():
            self.policy.slot_ratio_bias.fill_(-100.0)
            self.policy.slot_ratio_bias[:, 0] = 100.0
        output = self._forward()
        self.assertTrue(bool((output["ratio_class"] == 0).all()))
        normalized = output["ratio_policy_entropy"] / torch.log(torch.tensor(5.0))
        self.assertGreater(float(normalized.min()), 0.20)

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

    def test_training_actual_replay_source_augments_but_inference_rejects_it(self):
        output = self.policy(
            self.features, _args(), training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
            replay_source_indices=(3000, 3001),
        )
        self.assertEqual(output["natural_shortlist_indices"].shape[1], 256)
        self.assertTrue(bool((output["shortlist_indices"] == 3000).any()))
        with self.assertRaises(RuntimeError):
            self.policy(
                self.features, _args(), training=False,
                fixed_features=self.fixed, voxel_coords=self.coords,
                replay_source_indices=(3000,),
            )

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

    def test_temperature_anneal_waits_for_positive_state_experience(self):
        self.policy.train()
        common = dict(
            network_k_all_actual_enabled=True,
            network_k_all_actual_temperature=1.25,
            network_k_all_actual_temperature_min=0.50,
            network_k_all_actual_anneal_steps=5000,
            network_k_min_positive_before_anneal=8,
            _global_train_step=4000,
        )
        blocked = self.policy(
            self.features,
            _args(_network_k_positive_experience_count=0, **common),
            training=True,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )
        annealed = self.policy(
            self.features,
            _args(
                _network_k_positive_experience_count=8,
                _network_k_anneal_unlock_step=3000,
                **common,
            ),
            training=True,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )
        self.assertTrue(blocked["exploration_anneal_blocked"])
        self.assertAlmostEqual(float(blocked["exploration_temperature"]), 1.25, places=5)
        self.assertFalse(annealed["exploration_anneal_blocked"])
        self.assertAlmostEqual(float(annealed["exploration_anneal_progress"]), 0.2, places=5)
        self.assertAlmostEqual(float(annealed["exploration_temperature"]), 1.10, places=5)

    def test_all_actual_training_changes_all_eight_executable_patterns(self):
        self.policy.train()
        args = _args(
            network_k_all_actual_enabled=True,
            network_k_all_actual_temperature=1.0,
            network_k_all_actual_temperature_min=0.25,
            network_k_all_actual_anneal_steps=500,
            network_k_all_actual_coefficient_std=0.15,
            network_k_all_actual_direction_std=0.10,
            network_k_share_lattice_step=0.05,
            network_k_target_domain="neighbor26_empty",
            network_k_offline_dataset="",
            _global_train_step=0,
        )
        first = self.policy(
            self.features, args, training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        second = self.policy(
            self.features, args, training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        first_hashes = first["executable_plans"]["plan_hash"][0]
        second_hashes = second["executable_plans"]["plan_hash"][0]
        self.assertEqual(len(set(first_hashes)), 8)
        self.assertEqual(len(set(second_hashes)), 8)
        self.assertGreaterEqual(
            sum(left != right for left, right in zip(first_hashes, second_hashes)), 6
        )

    def test_cache_free_coverage_slots_rotate_theta_and_cover_share_lattice(self):
        self.policy.train()
        common = dict(
            network_k_all_actual_enabled=True,
            network_k_coverage_enabled=True,
            network_k_coverage_slots=4,
            network_k_coverage_share_stride=37,
            network_k_offline_dataset="",
        )
        torch.manual_seed(91)
        first = self.policy(
            self.features, _args(_global_train_step=40, _network_k_state_visit=0, **common), training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        torch.manual_seed(91)
        second = self.policy(
            self.features, _args(_global_train_step=80, _network_k_state_visit=1, **common), training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        self.assertEqual(first["coverage_slot_count"], 4)
        self.assertEqual(first["coverage_sequence_step"], 40)
        self.assertEqual(second["coverage_sequence_step"], 123)
        self.assertEqual(first["coverage_theta_index"].tolist(), [120, 120, 121, 122, 123, 124, 125, 126])
        self.assertEqual(second["coverage_theta_index"].tolist(), [369, 369, 370, 371, 372, 373, 374, 375])
        self.assertTrue(torch.equal(first["ratio_class"][0, 0:1], first["ratio_class"][0, 1:2]))
        self.assertTrue(torch.allclose(first["shares"][0, 0:1], first["shares"][0, 1:2]))
        self.assertNotEqual(
            int(first["coverage_permuted_theta_index"][2]),
            int(first["coverage_permuted_theta_index"][3]),
        )
        self.assertTrue(torch.equal(
            first["ratio_class"][0, :4],
            first["coverage_permuted_theta_index"][:4].remainder(5),
        ))
        first_lattice = first["coverage_share_lattice_index"][:4]
        expected_share = self.policy.share_lattice.index_select(0, first_lattice)
        self.assertTrue(torch.allclose(first["shares"][0, :4], expected_share))
        visited = {int((theta * 37) % 30780) for theta in range(30780)}
        self.assertEqual(len(visited), 30780)

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

    def test_real_policy_actor_scale_is_finite_after_where_normalization(self):
        self.policy.train()
        args = _args(
            heuristic_guidance_mode="network_k_proposal_policy",
            network_k_all_actual_enabled=True,
            network_k_coverage_enabled=True,
            network_k_coverage_slots=4,
            network_k_entropy_floor_target=0.20,
            network_k_entropy_floor_weight=0.25,
        )
        output = self.policy(
            self.features, args, training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        dummy = SimpleNamespace(
            training=True,
            args=args,
            last_k_proposal_terms=output,
            last_k_all_actual_debug={},
        )
        actual = torch.tensor([[0.10, -0.20, 0.05, 0.30, -0.10, 0.0, 0.20, -0.05]])
        total = network_module.Network.k_proposal_all_actual_loss(dummy, actual)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertLess(abs(dummy.last_k_all_actual_debug["actor_raw"]), 100.0)
        self.assertGreater(
            dummy.last_k_all_actual_debug["coverage_pair_where_contrast_raw"],
            0.0,
        )
        self.assertTrue(torch.isfinite(self.policy.shared_basis_head.weight.grad).all())

    def test_local_gain_uses_executed_clean_scores_not_order_descriptor(self):
        self.policy.train()
        output = self.policy(
            self.features,
            _args(network_k_all_actual_enabled=True),
            training=True,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )
        local_gain = output["predicted_local_gain_all"]
        self.assertEqual(tuple(local_gain.shape), (1, 8))
        self.assertTrue(local_gain.requires_grad)
        self.assertFalse(torch.allclose(local_gain, torch.full_like(local_gain, 1.5)))
        local_gain.sum().backward()
        self.assertIsNotNone(self.policy.shared_basis_head.weight.grad)
        self.assertGreater(float(self.policy.shared_basis_head.weight.grad.abs().sum()), 0.0)

    def test_all_worsening_plans_never_receive_positive_advantage(self):
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
        worsening = torch.tensor([[0.01, 0.10, 0.20, 0.40, 0.80, 0.05, 0.30, 0.60]])
        total = network_module.Network.k_proposal_all_actual_loss(dummy, worsening)
        combined = torch.tensor(dummy.last_k_all_actual_debug["combined_advantage"])
        self.assertTrue(bool((combined <= 0.0).all()))
        self.assertEqual(dummy.last_k_all_actual_debug["worsening_reinforced_count"], 0)
        actor_grad = torch.autograd.grad(total, log_prob)[0]
        self.assertTrue(bool((actor_grad >= 0.0).all()))

    def test_worsening_scheduled_slots_do_not_collapse_on_policy_distribution(self):
        theta_log_prob = torch.zeros(1, 8, requires_grad=True)
        spatial_log_prob = torch.zeros(1, 8, requires_grad=True)
        log_prob = theta_log_prob + spatial_log_prob
        predicted_gain = torch.zeros(1, 8, 1, requires_grad=True)
        dummy = SimpleNamespace(
            training=True,
            args=_args(
                heuristic_guidance_mode="network_k_proposal_policy",
                network_k_all_actual_enabled=True,
            ),
            last_k_proposal_terms={
                "slot_policy_log_prob": log_prob,
                "theta_policy_log_prob": theta_log_prob,
                "spatial_policy_log_prob": spatial_log_prob,
                "predicted_plan_gain": predicted_gain,
                "slot_policy_entropy": torch.ones(1, 8, requires_grad=True),
                "selected_slot": torch.zeros(1, dtype=torch.long),
                "coverage_slot_mask": torch.tensor(
                    [True, True, True, True, False, False, False, False]
                ),
            },
            last_k_all_actual_debug={},
        )
        compression = torch.tensor([[1.0, 0.8, 0.6, 0.4, -0.2, 0.1, 0.2, 0.3]])
        total = network_module.Network.k_proposal_all_actual_loss(dummy, compression)
        theta_grad, spatial_grad = torch.autograd.grad(
            total, (theta_log_prob, spatial_log_prob)
        )
        self.assertTrue(torch.equal(theta_grad[0, :4], torch.zeros(4)))
        self.assertTrue(bool((spatial_grad[0, :4] > 0.0).all()))
        self.assertLess(float(theta_grad[0, 4]), 0.0)

    def test_training_shortlist_moves_over_full_input_without_affecting_inference(self):
        self.policy.train()
        args0 = _args(
            network_k_all_actual_enabled=True,
            network_k_shortlist_exploration_fraction=0.5,
            _global_train_step=0,
        )
        args1 = _args(
            network_k_all_actual_enabled=True,
            network_k_shortlist_exploration_fraction=0.5,
            _global_train_step=1,
        )
        torch.manual_seed(11)
        first = self.policy(
            self.features, args0, training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        torch.manual_seed(11)
        second = self.policy(
            self.features, args1, training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        self.assertFalse(torch.equal(
            first["natural_shortlist_indices"],
            second["natural_shortlist_indices"],
        ))
        self.policy.eval()
        inferred0 = self.policy(
            self.features, args0, training=False,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        inferred1 = self.policy(
            self.features, args1, training=False,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        self.assertTrue(torch.equal(
            inferred0["natural_shortlist_indices"],
            inferred1["natural_shortlist_indices"],
        ))

    def test_positive_actual_experience_enters_mode_balanced_elite_replay(self):
        self.policy.train()
        output = self.policy(
            self.features,
            _args(network_k_all_actual_enabled=True),
            training=True,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )
        dummy = SimpleNamespace(
            training=True,
            args=_args(
                heuristic_guidance_mode="network_k_proposal_policy",
                network_k_all_actual_enabled=True,
                network_k_elite_enabled=True,
                network_k_elite_replay_capacity=64,
                network_k_elite_replay_count=8,
                network_k_elite_replay_weight=0.25,
                network_k_elite_min_gain_percent=1e-6,
            ),
            last_k_proposal_terms=output,
            last_k_all_actual_debug={},
            _network_k_actual_replay={},
            _network_k_previous_plan_hashes={},
        )
        dummy._store_network_k_actual_experience = MethodType(
            network_module.Network._store_network_k_actual_experience, dummy
        )
        dummy._network_k_elite_replay_loss = MethodType(
            network_module.Network._network_k_elite_replay_loss, dummy
        )
        dummy._network_k_positive_replay_source_indices = MethodType(
            network_module.Network._network_k_positive_replay_source_indices, dummy
        )
        actual_loss = torch.tensor([[-2.0, 0.4, 0.2, 0.1, 0.5, 0.3, 0.7, 0.6]])
        total = network_module.Network.k_proposal_all_actual_loss(
            dummy, actual_loss, state_key="known-state"
        )
        self.assertTrue(total.requires_grad)
        debug = dummy.last_k_all_actual_debug
        self.assertEqual(debug["replay_store"]["positive"], 1)
        self.assertGreaterEqual(debug["replay_elite"]["elite_count"], 1)
        self.assertAlmostEqual(debug["replay_elite"]["best_replay_gain"], 2.0, places=5)
        self.assertEqual(debug["replay_store"]["policy_mean_l2_delta"], -1.0)
        replay_sources = dummy._network_k_positive_replay_source_indices("known-state")
        self.assertTrue(replay_sources)

        shifted_probability = output["ratio_probability"].detach().roll(1, dims=2)
        output["ratio_probability"] = shifted_probability
        network_module.Network.k_proposal_all_actual_loss(
            dummy, actual_loss, state_key="known-state"
        )
        self.assertGreater(
            dummy.last_k_all_actual_debug["replay_store"]["policy_mean_l2_delta"], 0.0
        )

    def test_surrogate_pretrain_builds_required_full_cloud_canonical_basis(self):
        context = _build_surrogate_pretrain_canonical_context(
            torch.rand(1, 3, 128), _args()
        )
        self.assertEqual(context["octree_input_mode"], "full_cloud")
        self.assertEqual(context["canonical_source"], "surrogate_pretrain")
        self.assertTrue(torch.is_tensor(context["global_voxel_coords"]))
        self.assertTrue(torch.equal(
            context["global_voxel_coords"], context["full_global_voxel_coords"]
        ))

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
                self.aux_indices = None

            def _get_cached_actual_gt(self, cache_key):
                return {"bit": 1000.0}

            def _encode_actual_many(self, args, xyz_list, attach_aux_indices=None):
                self.encoded += len(xyz_list)
                self.aux_indices = tuple(attach_aux_indices or ())
                return [
                    {"bit": 1000.0 - 10.0 * index, "point_count": int(xyz.shape[-1]),
                     "actual_finished": True,
                     "actual_aux_stats_attached": index in self.aux_indices}
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
        self.assertEqual(dummy.aux_indices, (3,))
        self.assertEqual(result["proposal_aux_stats_count"], 1)
        self.assertEqual(tuple(result["actual_compression_percent"].shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
