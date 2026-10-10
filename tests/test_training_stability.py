import os
from collections import OrderedDict
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory

import torch

from models.utils.training.actual_compression_guard import (
    apply_actual_compression_guard,
    update_network_autonomy_from_guard,
)
from models.utils.training.compression_primary_loss import (
    build_compression_primary_loss,
)
from models.utils.training.lr_control import step_scheduler_with_floor
from models.utils.training.checkpointing import save_episode_checkpoint
from models.utils.training.optim_amp import clip_model_gradients
from models.utils.training.convergence_control import (
    TrainingConvergenceMonitor,
    exploration_schedule_step_estimate,
)
from models.utils.training.train_flow import backward_only_scaled_loss
from models.utils.training.train_runtime import (
    _balance_actual_operation_head_gradients,
    _capture_den6_policy_only_gradients,
    _probe_den6_policy_source_gradients,
    _restore_den6_policy_only_gradients,
    fixed_full_cloud_validation_records,
)
from models.modules.structure_actuator import (
    StructureRepairActuator,
    adaptive_exploration_candidate_changes,
    evidence_driven_post_window_multiplier,
    exploration_candidate_change_budget,
    policy_exploration_multiplier,
    smooth_exploration_phase,
)


class _Writer:
    def __init__(self):
        self.lines = []

    def write(self, value):
        self.lines.append(str(value))


class _Loss:
    compression_surrogate = None
    surrogate_optimizer = None


class _Plot:
    @staticmethod
    def epi_loss_return():
        return 1.0


class TrainingStabilityTest(unittest.TestCase):
    def test_source_gradient_probe_separates_actor_and_proposal_without_pollution(self):
        class _Actuator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.den6_candidate_actor = torch.nn.Linear(2, 1, bias=False)
                self.den6_candidate_proposal = torch.nn.Linear(2, 1, bias=False)
                self.den6_candidate_critic = torch.nn.Linear(2, 2, bias=False)
                self.den6_plan_critic = torch.nn.Linear(2, 2, bias=False)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actuator = _Actuator()

        model = _Model()
        value = torch.ones((1, 2))
        actor_value = model.actuator.den6_candidate_actor(value).sum()
        proposal_value = model.actuator.den6_candidate_proposal(value).sum()
        result = _probe_den6_policy_source_gradients(model, {
            "actual": actor_value + proposal_value,
            "proxy": actor_value,
        })
        self.assertGreater(result["actual_actor"], 0.0)
        self.assertGreater(result["actual_proposal"], 0.0)
        self.assertGreater(result["proxy_actor"], 0.0)
        self.assertEqual(result["proxy_proposal"], 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_source_gradient_probe_skips_reentrant_checkpoint_training_graph(self):
        class _Actuator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.args = SimpleNamespace(
                    heuristic_guidance_mode="ana_den6_online",
                    full_cloud_activation_checkpoint=True,
                )
                self.den6_candidate_actor = torch.nn.Linear(2, 1, bias=False)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actuator = _Actuator()

        model = _Model()
        value = torch.ones((1, 2))
        actor_value = model.actuator.den6_candidate_actor(value).sum()
        result = _probe_den6_policy_source_gradients(
            model, {"actual": actor_value}
        )
        self.assertEqual(result, {"probe_skipped_reentrant_checkpoint": 1.0})
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_candidate_score_fixed_scale_does_not_amplify_tiny_span(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator.args = SimpleNamespace(heuristic_guidance_network_score_scale=1.0)
        tiny = torch.tensor([0.0010, 0.0015, 0.0020], requires_grad=True)
        calibrated = actuator._normalize_candidate_network_score(tiny)
        self.assertLess(float(calibrated.detach().abs().max()), 0.001)
        calibrated.sum().backward()
        self.assertTrue(torch.isfinite(tiny.grad).all())

    def test_candidate_score_unmapped_entries_are_neutral(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator.args = SimpleNamespace(heuristic_guidance_network_score_scale=1.0)
        raw = torch.tensor([-2.0, -1.0, 0.0])
        scored = torch.tensor([True, True, False])
        calibrated = actuator._normalize_candidate_network_score(raw, scored)
        self.assertAlmostEqual(float(calibrated[2]), 0.0, places=7)

    def test_candidate_score_pool_scale_prevents_large_raw_saturation(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator.args = SimpleNamespace(
            heuristic_guidance_network_score_scale=1.0,
            heuristic_guidance_network_score_temperature=1.0,
        )
        raw = torch.tensor(
            [-120000.0, -60000.0, 0.0, 60000.0, 120000.0],
            requires_grad=True,
        )
        calibrated, audit = actuator._normalize_candidate_network_score(
            raw, return_audit=True
        )
        self.assertTrue(torch.equal(
            torch.argsort(raw.detach()), torch.argsort(calibrated.detach())
        ))
        self.assertEqual(audit["saturation_rate"], 0.0)
        self.assertGreater(audit["normalization_scale"], 1.0)
        calibrated.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_online_decision_balance_only_clips_large_gradient(self):
        class _Actuator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.drop_head = torch.nn.Linear(1, 1, bias=False)
                self.add_head = torch.nn.Linear(1, 1, bias=False)
                self.add_voxel_head = torch.nn.Linear(1, 1, bias=False)
                self.move_voxel_head = torch.nn.Linear(1, 1, bias=False)
                self.drop_amount_head = torch.nn.Linear(1, 1, bias=False)
                self.add_amount_head = torch.nn.Linear(1, 1, bias=False)
                self.move_amount_head = torch.nn.Linear(1, 1, bias=False)
                self.algorithmic_amount_selector_head = torch.nn.Linear(1, 1, bias=False)
                self.operation_gate_head = torch.nn.Linear(1, 1, bias=False)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actuator = _Actuator()
                self.policy_module = torch.nn.Linear(1, 1, bias=False)

        model = _Model()
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        model.actuator.drop_head.weight.grad.fill_(0.2)
        model.actuator.algorithmic_amount_selector_head.weight.grad.fill_(100.0)
        model.actuator.operation_gate_head.weight.grad.fill_(0.01)
        debug = _balance_actual_operation_head_gradients(
            SimpleNamespace(
                repair_balance_operation_head_grads=True,
                heuristic_guidance_mode="ana_den6_online",
                repair_operation_head_grad_target=1.0,
                grad_scale_operation_amount=200.0,
                minimal_loss_objective=True,
                repair_online_decision_grad_max_norm=1.0,
            ),
            model,
            {},
        )
        self.assertEqual(debug["den6_online_where_grad_balance_status"], "preserved")
        self.assertEqual(debug["den6_online_amount_grad_balance_status"], "clipped")
        self.assertEqual(debug["den6_online_action_grad_balance_status"], "preserved")
        self.assertAlmostEqual(float(model.actuator.drop_head.weight.grad.norm()), 0.2, places=6)
        self.assertAlmostEqual(
            float(model.actuator.algorithmic_amount_selector_head.weight.grad.norm()),
            1.0,
            places=6,
        )

    def test_exploration_schedule_is_independent_from_extended_training_length(self):
        args = SimpleNamespace(episodes=384, exploration_schedule_episodes=256)
        self.assertEqual(exploration_schedule_step_estimate(args, 40), 256 * 40)

    def test_zero_exploration_schedule_uses_legacy_episode_length(self):
        args = SimpleNamespace(episodes=384, exploration_schedule_episodes=0)
        self.assertEqual(exploration_schedule_step_estimate(args, 40), 384 * 40)

    def test_auto_exploration_schedule_tracks_convergence_limit(self):
        args = SimpleNamespace(
            episodes=256,
            exploration_schedule_episodes=0,
            train_until_converged=True,
            convergence_min_episodes=256,
            convergence_max_episodes=384,
        )
        self.assertEqual(exploration_schedule_step_estimate(args, 40), 384 * 40)

    def test_exploration_tail_lands_without_a_slope_jump(self):
        fraction = 0.9
        tail = 0.25
        boundary = fraction * (1.0 - tail)
        epsilon = 1e-5
        left_slope = (
            smooth_exploration_phase(boundary, fraction, tail)
            - smooth_exploration_phase(boundary - epsilon, fraction, tail)
        ) / epsilon
        right_slope = (
            smooth_exploration_phase(boundary + epsilon, fraction, tail)
            - smooth_exploration_phase(boundary, fraction, tail)
        ) / epsilon
        end_slope = (
            smooth_exploration_phase(fraction, fraction, tail)
            - smooth_exploration_phase(fraction - epsilon, fraction, tail)
        ) / epsilon
        self.assertAlmostEqual(left_slope, right_slope, places=3)
        self.assertAlmostEqual(end_slope, 0.0, places=3)

    def test_constant_policy_exploration_does_not_change_with_training_phase(self):
        values = [
            policy_exploration_multiplier(
                training=True,
                schedule_mode="constant",
                constant_multiplier=0.25,
                annealed_phase=phase,
            )
            for phase in (0.0, 0.5, 0.99, 1.0)
        ]
        self.assertEqual(values, [0.25, 0.25, 0.25, 0.25])

    def test_policy_exploration_is_disabled_during_evaluation(self):
        self.assertEqual(
            policy_exploration_multiplier(
                training=False,
                schedule_mode="constant",
                constant_multiplier=0.25,
                annealed_phase=0.0,
            ),
            0.0,
        )

    def test_candidate_exploration_budget_contracts_with_actual_quality(self):
        self.assertEqual(adaptive_exploration_candidate_changes(16, 1.0), 16)
        self.assertEqual(adaptive_exploration_candidate_changes(16, 0.5), 8)
        self.assertEqual(adaptive_exploration_candidate_changes(16, 0.25), 4)

    def test_candidate_exploration_budget_never_vanishes_while_enabled(self):
        self.assertEqual(adaptive_exploration_candidate_changes(16, 0.0), 1)
        self.assertEqual(adaptive_exploration_candidate_changes(0, 1.0), 0)

    def test_candidate_exploration_budget_scales_with_executed_plan(self):
        self.assertEqual(
            exploration_candidate_change_budget(96, 800, 0.125, 1.0), 96
        )
        self.assertEqual(
            exploration_candidate_change_budget(96, 400, 0.125, 1.0), 50
        )
        self.assertEqual(
            exploration_candidate_change_budget(96, 400, 0.125, 0.5), 25
        )
        self.assertEqual(
            exploration_candidate_change_budget(96, 0, 0.125, 1.0), 0
        )
        self.assertEqual(
            exploration_candidate_change_budget(96, 400, 0.125, 0.0), 0
        )

    def test_slightly_wider_exploration_budget_remains_bounded(self):
        self.assertEqual(
            exploration_candidate_change_budget(128, 800, 0.175, 1.0), 128
        )
        self.assertEqual(
            exploration_candidate_change_budget(128, 400, 0.175, 1.0), 70
        )
        self.assertEqual(
            exploration_candidate_change_budget(128, 800, 0.175, 0.02), 3
        )

    def test_post_window_exploration_is_evidence_driven_and_tiny(self):
        multiplier, need = evidence_driven_post_window_multiplier(
            0.02,
            64,
            [{"comparisons": 0, "authority": 0.0, "uncertainty": 1.0}],
        )
        self.assertAlmostEqual(multiplier, 0.02)
        self.assertAlmostEqual(need, 1.0)
        multiplier, need = evidence_driven_post_window_multiplier(
            0.02,
            64,
            [{"comparisons": 64, "authority": 1.0, "uncertainty": 0.1}],
        )
        self.assertEqual(multiplier, 0.0)
        self.assertEqual(need, 0.0)

    def test_factorized_exploration_softly_prioritizes_where_coverage(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator.args = SimpleNamespace(
            heuristic_guidance_online_factorized_exploration=True,
            heuristic_guidance_online_factorized_ucb_scale=0.25,
            heuristic_guidance_online_where_exploration_share=0.60,
            heuristic_guidance_online_where_coverage_weight=1.0,
            heuristic_guidance_online_group_evidence_target=64,
            heuristic_guidance_online_group_need_weight=0.75,
            heuristic_guidance_online_group_authority_pairs=8,
            _current_input_file="frame.ply",
            sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=0,
            sparsepcgc_scale_m=8,
        )
        groups = (
            "AddWhere", "PruneWhere", "AdjustWhere", "Amount", "Gate", "Fine"
        )
        actuator._den6_factorized_group_counts = {}
        actuator._den6_factorized_group_uncertainty = {}
        actuator._den6_pending_matched_groups = OrderedDict()
        actuator._den6_actual_group_stats = {
            name: {
                "comparisons": 0,
                "actor_correct": 0.0,
                "critic_correct": 0.0,
                "proposal_comparisons": 0,
                "proposal_correct": 0.0,
            }
            for name in groups
        }
        actuator._den6_fixed_rd_authority_rejected = False
        actuator.train()
        confidence = {
            name: {"uncertainty": 1.0}
            for name in ("Add", "Prune", "Adjust")
        }
        for _ in range(120):
            actuator._select_den6_factorized_exploration_group(confidence)
        counts = actuator._den6_factorized_group_counts
        where_count = sum(counts.get(name, 0) for name in groups[:3])
        self.assertGreaterEqual(where_count, 60)
        self.assertTrue(all(counts.get(name, 0) > 0 for name in groups))

    def test_completed_matched_group_releases_pending_frame(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator._den6_pending_matched_groups = OrderedDict((
            ("frame", {"group": "AddWhere"}),
        ))
        actuator._den6_matched_anchor_plan_memory = OrderedDict((
            ("anchor", {"selected_indices": {}}),
        ))
        self.assertTrue(
            actuator._complete_den6_pending_matched_group("frame", "anchor")
        )
        self.assertFalse(actuator._den6_pending_matched_groups)
        self.assertFalse(actuator._den6_matched_anchor_plan_memory)

    def test_annealed_policy_exploration_remains_available_for_ablation(self):
        self.assertAlmostEqual(
            policy_exploration_multiplier(
                training=True,
                schedule_mode="annealed",
                constant_multiplier=0.25,
                annealed_phase=0.6,
            ),
            0.4,
        )

    def test_quarter_training_exploration_reaches_zero_at_boundary(self):
        fraction = 0.25
        expected_rows = (
            (0.0, 1.0, 16),
            (0.125, 0.5, 8),
            (0.25, 0.0, 0),
            (0.8, 0.0, 0),
        )
        for progress, expected, expected_budget in expected_rows:
            phase = smooth_exploration_phase(progress, fraction, 0.25)
            multiplier = policy_exploration_multiplier(
                training=True,
                schedule_mode="annealed",
                constant_multiplier=0.50,
                annealed_phase=phase,
            )
            self.assertAlmostEqual(multiplier, expected, places=6)
            budget = (
                adaptive_exploration_candidate_changes(16, multiplier)
                if multiplier > 0.0 else 0
            )
            self.assertEqual(budget, expected_budget)

    def test_exact_online_legacy_noise_is_zero_after_quarter_window(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            repair_policy_exploration_mode="annealed",
            repair_policy_exploration_constant_multiplier=0.50,
            repair_exploration_fraction=0.25,
            repair_exploration_smooth_tail_fraction=0.25,
            _exploration_schedule_steps_estimate=1000,
            _total_train_steps_estimate=1000,
            _global_train_step=0,
            noise_start=0.10,
            noise_end=0.05,
        )
        actuator.train()
        values = []
        for step in (0, 125, 250, 800):
            actuator.args._global_train_step = step
            values.append(actuator._annealed_value("noise_start", "noise_end"))
        self.assertEqual([round(value, 6) for value in values], [0.10, 0.05, 0.0, 0.0])

    def test_constant_exact_online_phase_also_freezes_legacy_random_mix(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            repair_policy_exploration_mode="constant",
            repair_policy_exploration_constant_multiplier=0.25,
            repair_exploration_fraction=0.9,
            repair_exploration_smooth_tail_fraction=0.25,
            _exploration_schedule_steps_estimate=1000,
            _total_train_steps_estimate=1000,
            _global_train_step=0,
        )
        actuator.train()
        phases = []
        values = []
        for step in (0, 500, 999):
            actuator.args._global_train_step = step
            phases.append(actuator._behavior_exploration_phase())
            actuator.args.noise_start = 0.10
            actuator.args.noise_end = 0.02
            values.append(actuator._annealed_value("noise_start", "noise_end"))
        self.assertEqual(phases, [0.75, 0.75, 0.75])
        self.assertEqual([round(value, 6) for value in values], [0.04, 0.04, 0.04])

        actuator.eval()
        self.assertEqual(actuator._behavior_exploration_phase(), 1.0)
        self.assertAlmostEqual(actuator._annealed_value("noise_start", "noise_end"), 0.02)

    def test_backward_only_policy_term_preserves_gradient_and_zeroes_forward(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        raw_policy_loss = parameter.square() + 3.0
        neutral = backward_only_scaled_loss(raw_policy_loss, scale=10.0)
        self.assertEqual(float(neutral.detach()), 0.0)
        neutral.backward()
        self.assertAlmostEqual(float(parameter.grad), 40.0)

    def test_convergence_requires_post_minimum_stable_evidence(self):
        args = SimpleNamespace(
            train_until_converged=True,
            episodes=4,
            convergence_min_episodes=0,
            convergence_window_episodes=4,
            convergence_patience_episodes=3,
            convergence_guard_cooldown_episodes=2,
            convergence_actual_compression_target=-3.5,
            convergence_total_loss_slope_max=0.02,
            convergence_total_loss_half_delta_max=0.12,
            convergence_compression_slope_max=0.01,
            convergence_compression_half_delta_max=0.08,
            convergence_fixed_objective_slope_max=0.001,
            convergence_fixed_objective_half_delta_max=0.01,
            convergence_geometry_relative_worsening_max=0.0025,
            repair_exploration_fraction=0.9,
            _total_train_steps_estimate=40,
        )
        monitor = TrainingConvergenceMonitor(args)
        converged = []
        for episode in range(1, 8):
            event = monitor.update(
                {
                    "episode": episode,
                    "total_loss": -10.0,
                    "compression_loss_L_com": -11.0,
                    "full_cloud_val_fixed_objective": -2.4,
                    "full_cloud_val_actual_percent": -3.6,
                    "full_cloud_val_geometry": 0.025,
                    "full_cloud_val_sample_signature": "fixed-a",
                    "optimizer_success_ok": True,
                    "geometry_ok": True,
                    "safety_ok": True,
                },
                guard_event={},
                global_step=episode * 10,
            )
            converged.append(event["converged"])
        self.assertFalse(any(converged[:6]))
        self.assertTrue(converged[6])

    def test_convergence_resets_after_fixed_validation_level_shift(self):
        args = SimpleNamespace(
            train_until_converged=True,
            episodes=1,
            convergence_min_episodes=0,
            convergence_window_episodes=4,
            convergence_patience_episodes=2,
            convergence_guard_cooldown_episodes=0,
            convergence_actual_compression_target=-3.5,
            convergence_total_loss_slope_max=0.02,
            convergence_total_loss_half_delta_max=0.12,
            convergence_compression_slope_max=0.01,
            convergence_compression_half_delta_max=0.08,
            convergence_fixed_objective_slope_max=0.001,
            convergence_fixed_objective_half_delta_max=0.01,
            convergence_geometry_relative_worsening_max=0.0025,
            repair_exploration_fraction=0.0,
            _total_train_steps_estimate=1,
        )
        monitor = TrainingConvergenceMonitor(args)
        event = None
        for episode, fixed_objective in enumerate((-2.4, -2.4, -2.4, -2.2), 1):
            event = monitor.update(
                {
                    "episode": episode,
                    "total_loss": -10.0,
                    "compression_loss_L_com": -11.0,
                    "full_cloud_val_fixed_objective": fixed_objective,
                    "full_cloud_val_actual_percent": -3.6,
                    "full_cloud_val_geometry": 0.025,
                    "full_cloud_val_sample_signature": "fixed-a",
                    "optimizer_success_ok": True,
                    "geometry_ok": True,
                    "safety_ok": True,
                },
                guard_event={},
                global_step=episode,
            )
        self.assertFalse(event["stable_now"])
        self.assertIn("fixed_objective_slope", event["reasons"])
        self.assertEqual(event["stable_episodes"], 0)

    def test_stochastic_train_level_shift_is_not_convergence_evidence(self):
        args = SimpleNamespace(
            train_until_converged=True,
            episodes=1,
            convergence_min_episodes=0,
            convergence_window_episodes=4,
            convergence_patience_episodes=1,
            convergence_guard_cooldown_episodes=0,
            convergence_actual_compression_target=-3.5,
            convergence_fixed_objective_slope_max=0.001,
            convergence_fixed_objective_half_delta_max=0.01,
            convergence_geometry_relative_worsening_max=0.0025,
        )
        monitor = TrainingConvergenceMonitor(args)
        event = None
        for episode, train_compression in enumerate((-2.0, -2.2, -3.2, -3.8), 1):
            event = monitor.update(
                {
                    "episode": episode,
                    "total_loss": train_compression * 3.0,
                    "compression_loss_L_com": train_compression,
                    "full_cloud_val_fixed_objective": -2.4,
                    "full_cloud_val_actual_percent": -3.6,
                    "full_cloud_val_geometry": 0.025,
                    "full_cloud_val_sample_signature": "fixed-a",
                    "optimizer_success_ok": True,
                    "geometry_ok": True,
                    "safety_ok": True,
                },
                guard_event={},
                global_step=episode,
            )
        self.assertTrue(event["stable_now"])
        self.assertTrue(event["converged"])
        self.assertNotIn("compression_slope", event["reasons"])

    def _guard_args(self):
        return SimpleNamespace(
            compression_loss_backend="sparsepcgc_surrogate",
            actual_compression_guard=True,
            actual_guard_require_fixed_validation=True,
            actual_guard_require_full_state_restore=True,
            checkpoint_full_cloud_min_count=1,
            actual_guard_min_fresh=1,
            actual_guard_improvement_epsilon=1e-6,
            actual_guard_tolerance=0.01,
            actual_guard_patience=1,
            actual_guard_restore_best=True,
            actual_guard_max_restores=1,
            actual_guard_max_restore_age_episodes=16,
            actual_guard_rebase_stale_best=True,
            actual_guard_decay_lr=False,
            actual_guard_lr_decay=0.5,
            min_main_lr=1e-6,
            min_surrogate_lr=1e-6,
            _global_train_step=3,
            _sparsepcgc_full_cloud_sequence_baseline_memory={"seq": {"baseline": -3.0}},
        )

    def test_guard_uses_fixed_validation_only(self):
        args = self._guard_args()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        event = apply_actual_compression_guard(
            args=args,
            model=model,
            loss=_Loss(),
            optimizer=optimizer,
            writer=_Writer(),
            guard_state={},
            checkpoint_metrics={
                "checkpoint_eligible": True,
                "checkpoint_actual_source": "fresh",
                "checkpoint_actual_delta": -3.0,
                "checkpoint_actual_count": 1,
            },
            ckpt_dir=".",
            episode=0,
        )
        self.assertEqual(event["action"], "skipped")
        self.assertEqual(event["reason"], "fixed_full_cloud_validation_required")

    def test_guard_restores_optimizer_scheduler_scaler_and_runtime_memory(self):
        torch.manual_seed(4)
        args = self._guard_args()
        writer = _Writer()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        mapping = {"state": 3}
        runtime_state = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "mutable_mappings": {"network_k_state_visit_counts": mapping},
        }

        # Adam momentを作った同じ時点をbestとして保存する。
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        saved_weight = model.weight.detach().clone()
        saved_lr = optimizer.param_groups[0]["lr"]

        with TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "0.pth")
            torch.save(model.state_dict(), model_path)
            guard_state = {}
            best_event = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=writer,
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -3.0,
                    "checkpoint_actual_count": 1,
                    "full_cloud_val_sample_signature": "fixed-set-a",
                },
                ckpt_dir=directory,
                episode=0,
                runtime_state=runtime_state,
            )
            self.assertTrue(best_event["training_state_saved"])

            with torch.no_grad():
                model.weight.add_(10.0)
            optimizer.param_groups[0]["lr"] = 0.123
            scheduler.step()
            mapping["state"] = 99
            args._sparsepcgc_full_cloud_sequence_baseline_memory = {
                "seq": {"baseline": 99.0}
            }

            rollback = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=writer,
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -2.0,
                    "checkpoint_actual_count": 1,
                    "full_cloud_val_sample_signature": "fixed-set-a",
                },
                ckpt_dir=directory,
                episode=1,
                runtime_state=runtime_state,
            )
            self.assertEqual(rollback["action"], "rollback")
            self.assertTrue(rollback["training_state_restored"])
            self.assertTrue(torch.equal(model.weight.detach(), saved_weight))
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], saved_lr)
            self.assertEqual(mapping["state"], 3)
            self.assertEqual(
                args._sparsepcgc_full_cloud_sequence_baseline_memory["seq"]["baseline"],
                -3.0,
            )

    def test_guard_rejects_changed_validation_set(self):
        args = self._guard_args()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        guard_state = {}
        with TemporaryDirectory() as directory:
            torch.save(model.state_dict(), os.path.join(directory, "0.pth"))
            first = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=_Writer(),
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -3.0,
                    "checkpoint_actual_count": 1,
                    "full_cloud_val_sample_signature": "fixed-set-a",
                },
                ckpt_dir=directory,
                episode=0,
            )
            self.assertEqual(first["action"], "new_best")
            changed = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=_Writer(),
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -4.0,
                    "checkpoint_actual_count": 1,
                    "full_cloud_val_sample_signature": "different-set",
                },
                ckpt_dir=directory,
                episode=1,
            )
            self.assertEqual(changed["action"], "skipped")
            self.assertEqual(changed["reason"], "fixed_validation_signature_changed")

    def test_guard_does_not_restore_the_same_best_twice(self):
        args = self._guard_args()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        guard_state = {}
        with TemporaryDirectory() as directory:
            torch.save(model.state_dict(), os.path.join(directory, "0.pth"))
            first = apply_actual_compression_guard(
                args=args, model=model, loss=_Loss(), optimizer=optimizer,
                writer=_Writer(), guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -3.0,
                    "checkpoint_actual_count": 1,
                    "full_cloud_val_fixed_objective": -2.0,
                    "full_cloud_val_sample_signature": "fixed-set-a",
                },
                ckpt_dir=directory, episode=0,
                runtime_state={"optimizer": optimizer},
            )
            self.assertEqual(first["action"], "new_best")

            # A local rebase must not silently re-arm another model restore.
            for episode in (1, 2, 3):
                torch.save(
                    model.state_dict(), os.path.join(directory, f"{episode}.pth")
                )
                event = apply_actual_compression_guard(
                    args=args, model=model, loss=_Loss(), optimizer=optimizer,
                    writer=_Writer(), guard_state=guard_state,
                    checkpoint_metrics={
                        "checkpoint_eligible": True,
                        "checkpoint_actual_source": "full_cloud",
                        "checkpoint_actual_delta": -2.0 + 0.1 * episode,
                        "checkpoint_actual_count": 1,
                        "full_cloud_val_fixed_objective": -1.0 + 0.1 * episode,
                        "full_cloud_val_sample_signature": "fixed-set-a",
                    },
                    ckpt_dir=directory, episode=episode,
                    runtime_state={"optimizer": optimizer},
                )
                if episode == 1:
                    self.assertEqual(event["action"], "rollback")
                else:
                    self.assertEqual(event["action"], "rebase_stale")
                    self.assertEqual(event["reason"], "restore_budget_exhausted")
                    self.assertEqual(event["restore_count"], 1)

    def test_disabled_scheduler_does_not_decay_emulator_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        args = SimpleNamespace(lr_scheduler_enabled=False, min_main_lr=1e-6)
        event = step_scheduler_with_floor(scheduler, optimizer, args)
        self.assertFalse(event["scheduler_stepped"])
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_plateau_scheduler_requires_finite_fixed_rd_metric(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=0
        )
        args = SimpleNamespace(
            lr_scheduler_enabled=True,
            lr_scheduler_mode="plateau",
            min_main_lr=0.0,
        )
        skipped = step_scheduler_with_floor(
            scheduler, optimizer, args, metric=None
        )
        self.assertFalse(skipped["scheduler_stepped"])
        step_scheduler_with_floor(scheduler, optimizer, args, metric=1.0)
        reduced = step_scheduler_with_floor(scheduler, optimizer, args, metric=1.1)
        self.assertTrue(reduced["scheduler_stepped"])
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)

    def test_best_pth_uses_deterministic_fixed_rd_objective(self):
        model = torch.nn.Linear(1, 1, bias=False)
        args = SimpleNamespace(compression_loss_backend="sparsepcgc_actual")
        trackers = {}
        with TemporaryDirectory() as directory:
            with torch.no_grad():
                model.weight.fill_(1.0)
            _, _, trackers = save_episode_checkpoint(
                model, directory, _Plot(), _Writer(), 0, float("inf"),
                args=args,
                stage="joint",
                checkpoint_metrics={
                    "total_loss": 1.0,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -3.0,
                    "checkpoint_actual_count": 5,
                    "full_cloud_val_actual_count": 5,
                    "full_cloud_val_fixed_objective": -2.0,
                    "checkpoint_eligible": True,
                    "geometry_ok": True,
                    "safety_ok": True,
                },
                best_trackers=trackers,
            )
            with torch.no_grad():
                model.weight.fill_(2.0)
            _, _, trackers = save_episode_checkpoint(
                model, directory, _Plot(), _Writer(), 1, 0.0,
                args=args,
                stage="joint",
                checkpoint_metrics={
                    "total_loss": 0.5,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -3.5,
                    "checkpoint_actual_count": 5,
                    "full_cloud_val_actual_count": 5,
                    "full_cloud_val_fixed_objective": -1.5,
                    "checkpoint_eligible": True,
                    "geometry_ok": True,
                    "safety_ok": True,
                },
                best_trackers=trackers,
            )
            best = torch.load(os.path.join(directory, "best.pth"), map_location="cpu")
            self.assertAlmostEqual(float(best["weight"]), 1.0)
            self.assertEqual(trackers["best_pth_source"], "best_fixed_rd")
    def test_fixed_validation_records_are_disjoint_and_ignore_moving_window(self):
        dataset = SimpleNamespace(
            all_files=[
                "/tmp/frame_0001.ply", "/tmp/frame_0002.ply", "/tmp/frame_0003.ply",
                "/tmp/frame_0004.ply", "/tmp/frame_0005.ply",
            ],
            files=["/tmp/frame_0001.ply", "/tmp/frame_0002.ply"],
        )
        args = SimpleNamespace(train_frames_per_sequence=3)
        first, created = fixed_full_cloud_validation_records(
            args, [("sequence", dataset)], 2
        )
        self.assertTrue(created)
        self.assertEqual(
            [path for _, path in first],
            ["/tmp/frame_0004.ply", "/tmp/frame_0005.ply"],
        )
        dataset.files = ["/tmp/frame_0003.ply", "/tmp/frame_0001.ply"]
        second, created_again = fixed_full_cloud_validation_records(
            args, [("sequence", dataset)], 2
        )
        self.assertFalse(created_again)
        self.assertEqual([path for _, path in first], [path for _, path in second])

    def test_network_autonomy_grows_only_on_fixed_validation_new_best(self):
        args = SimpleNamespace(
            heuristic_guidance_network_residual_weight=0.05,
            heuristic_guidance_network_residual_weight_max=0.10,
            heuristic_guidance_network_residual_weight_increment=0.025,
        )
        first = update_network_autonomy_from_guard(args, {"action": "new_best"})
        self.assertAlmostEqual(first["current"], 0.075)
        unchanged = update_network_autonomy_from_guard(
            args, {"action": "within_tolerance"}
        )
        self.assertAlmostEqual(unchanged["current"], 0.075)
        second = update_network_autonomy_from_guard(args, {"action": "new_best"})
        self.assertAlmostEqual(second["current"], 0.10)
        capped = update_network_autonomy_from_guard(args, {"action": "new_best"})
        self.assertAlmostEqual(capped["current"], 0.10)
        rolled_back = update_network_autonomy_from_guard(
            args, {"action": "rollback", "rd_improved": False}
        )
        self.assertAlmostEqual(rolled_back["current"], 0.075)

        # RDが改善しても固定圧縮目標未達では裁量を広げない。
        args._heuristic_guidance_network_residual_weight_current = 0.05
        held = update_network_autonomy_from_guard(
            args,
            {"action": "new_best", "rd_improved": True, "actual_delta": -3.49},
        )
        self.assertAlmostEqual(held["current"], 0.05)
        self.assertFalse(held["compression_target_met"])

    def test_network_autonomy_ignores_sub_noise_improvements(self):
        args = SimpleNamespace(
            heuristic_guidance_network_residual_weight=0.05,
            heuristic_guidance_network_residual_weight_max=0.15,
            heuristic_guidance_network_residual_weight_increment=0.025,
            actual_guard_autonomy_compression_target=-3.5,
            actual_guard_autonomy_min_improvement=0.01,
        )
        first = update_network_autonomy_from_guard(
            args, {"action": "new_best", "rd_improved": True, "actual_delta": -3.60}
        )
        tiny = update_network_autonomy_from_guard(
            args, {"action": "new_best", "rd_improved": True, "actual_delta": -3.605}
        )
        accumulated = update_network_autonomy_from_guard(
            args, {"action": "new_best", "rd_improved": True, "actual_delta": -3.611}
        )
        self.assertAlmostEqual(first["current"], 0.075)
        self.assertAlmostEqual(tiny["current"], 0.075)
        self.assertFalse(tiny["meaningful_improvement"])
        self.assertAlmostEqual(accumulated["current"], 0.10)

    def test_guard_does_not_accept_geometry_unsafe_compression_best(self):
        args = self._guard_args()
        args.cp_lambda_geom = 50.0
        args.cp_tau_geom = 0.0
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        guard_state = {}
        with TemporaryDirectory() as directory:
            unsafe = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=_Writer(),
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": False,
                    "checkpoint_ineligible_reason": "fixed_validation_geometry_or_safety_failed",
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -4.0,
                    "checkpoint_actual_count": 1,
                    "full_cloud_val_geometry": 0.02,
                    "full_cloud_val_fixed_objective": 4.0,
                    "full_cloud_val_sample_signature": "fixed-set-a",
                    "geometry_ok": False,
                    "safety_ok": False,
                },
                ckpt_dir=directory,
                episode=0,
            )
            self.assertNotEqual(unsafe["action"], "new_best")
            self.assertTrue(unsafe["unsafe_candidate"])
            self.assertIsNone(guard_state["best_path"])

            safe = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=_Writer(),
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -3.8,
                    "checkpoint_actual_count": 1,
                    "full_cloud_val_geometry": 0.003,
                    "full_cloud_val_fixed_objective": -2.6,
                    "full_cloud_val_sample_signature": "fixed-set-a",
                    "geometry_ok": True,
                    "safety_ok": True,
                },
                ckpt_dir=directory,
                episode=1,
            )
            self.assertEqual(safe["action"], "new_best")
            self.assertTrue(safe["rd_improved"])

    def test_sparsepcgc_geometry_penalty_is_continuous(self):
        args = SimpleNamespace(
            compression_loss_backend="sparsepcgc_surrogate",
            w_com=1.0,
            cp_tau_geom=0.0,
            cp_lambda_geom=100.0,
            compression_primary_aux_target_ratio=0.25,
            compression_primary_aux_balance_min_scale=0.0,
            compression_primary_aux_balance_max_scale=1.0,
        )
        main = torch.tensor(-4.0, requires_grad=True)
        geom = torch.tensor(0.003, requires_grad=True)
        total, _, debug = build_compression_primary_loss(
            args,
            terms={"main": main},
            L_com=main,
            L_geom=geom,
            L_actuator=torch.tensor(0.0),
            global_train_step=0,
            stage_factors={},
        )
        self.assertAlmostEqual(debug["cp_P_geom"], 0.003, places=7)
        self.assertAlmostEqual(debug["cp_geom_block_raw"], 0.3, places=6)
        total.backward()
        self.assertGreater(float(geom.grad), 0.0)

    def test_gradient_clip_is_shared_by_fp32_and_amp_paths(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.full_like(model.weight, 20.0)
        debug = clip_model_gradients(model, 10.0)
        self.assertTrue(debug["train_grad_clip_applied"])
        self.assertAlmostEqual(debug["train_grad_total_norm_before_clip"], 20.0)
        self.assertLessEqual(float(model.weight.grad.norm()), 10.00001)

    def test_den6_policy_gradient_isolation_removes_unrelated_ste_gradient(self):
        head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(2.0)
        model = SimpleNamespace(actuator=SimpleNamespace(drop_head=head))
        value = head(torch.ones(1, 1)).sum()
        policy_loss = value.square()
        snapshot = _capture_den6_policy_only_gradients(policy_loss, model)
        (policy_grad,) = torch.autograd.grad(
            policy_loss, (head.weight,), retain_graph=True
        )
        (policy_loss + 100.0 * value).backward()
        self.assertNotAlmostEqual(
            float(head.weight.grad), float(policy_grad), places=4
        )
        debug = _restore_den6_policy_only_gradients(snapshot)
        self.assertTrue(debug["den6_policy_only_gradient_isolation"])
        self.assertAlmostEqual(float(head.weight.grad), float(policy_grad), places=6)

    def test_disabled_gradient_clip_does_not_change_gradient(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.full_like(model.weight, 20.0)
        debug = clip_model_gradients(model, 0.0)
        self.assertFalse(debug["train_grad_clip_applied"])
        self.assertAlmostEqual(float(model.weight.grad.norm()), 20.0)


if __name__ == "__main__":
    unittest.main()
