import ast
import inspect
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import torch

import models.network as network_module
from models.modules.network_only_codec_policy import NetworkOnlyCodecPolicy
from models.modules.structure_actuator import StructureRepairActuator


def _args(**overrides):
    values = dict(
        sparsepcgc_scale_ae=0,
        sparsepcgc_scale_sr=2,
        sparsepcgc_scale_m=8,
        sparsepcgc_voxel_size=1.0,
        sparsepcgc_pos_quantscale=1.0,
        sparsepcgc_psnr_resolution=1023,
        sparsepcgc_native_bit_depth=10,
        network_only_exploration_steps=0,
        _global_train_step=10,
        network_only_where_gumbel_scale=0.0,
        network_only_action_exploration_floor=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class NetworkOnlyCodecPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.policy = NetworkOnlyCodecPolicy(25, hidden_dim=16).eval()
        self.features = torch.randn(1, 25, 127)
        self.fixed = torch.rand(1, 6, 127)

    def _forward(self, args=None):
        return self.policy(
            self.features,
            args or _args(),
            training=False,
            fixed_features=self.fixed,
        )

    def test_policy_module_has_no_legacy_import(self):
        tree = ast.parse(inspect.getsource(network_module.NetworkOnlyCodecPolicy))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any("den" in name or "teacher" in name or "candidate" in name for name in imports))

    def test_forbidden_functions_can_be_monkeypatched_to_raise(self):
        error = RuntimeError("forbidden legacy path reached")
        with mock.patch.object(network_module, "attach_ana_den6_online_guidance", side_effect=error), \
             mock.patch.object(network_module, "attach_ana_den6_reference_anchor", side_effect=error), \
             mock.patch.object(network_module, "build_heuristic_guidance", side_effect=error):
            output = self._forward()
        self.assertEqual(tuple(output["local_cost_maps"].shape), (1, 3, 7, 127))

    def test_codec_settings_condition_policy(self):
        first = self._forward(_args(sparsepcgc_scale_ae=0, sparsepcgc_scale_sr=2))["where_logits"]
        second = self._forward(_args(sparsepcgc_scale_ae=3, sparsepcgc_scale_sr=0))["where_logits"]
        self.assertFalse(torch.equal(first, second))

    def test_amount_share_action_and_direction_control_plan(self):
        with torch.no_grad():
            self.policy.amount_head.bias.fill_(-8.0)
        small = self._forward()
        small_count = int(round(float(small["total_ratio"].item()) * 1_000_000))
        with torch.no_grad():
            self.policy.amount_head.bias.fill_(1.0)
            self.policy.share_head.bias.copy_(torch.tensor([5.0, -5.0, -5.0]))
            self.policy.gate_head.bias.copy_(torch.tensor([5.0, -5.0, -5.0]))
        changed = self._forward()
        large_count = int(round(float(changed["total_ratio"].item()) * 1_000_000))
        self.assertGreater(large_count, small_count)
        self.assertGreater(float(changed["operation_ratios"][0, 0]), float(changed["operation_ratios"][0, 1]))

        with torch.no_grad():
            self.policy.direction_field_head.weight.zero_()
            self.policy.direction_field_head.bias.zero_()
            self.policy.direction_field_head.bias[0] = 1.0
            self.policy.direction_field_head.bias[3] = 2.0
        positive_x = self._forward()["direction_logits"][0, 0].mean(dim=1).argmax()
        with torch.no_grad():
            self.policy.direction_field_head.bias[0] = -1.0
        negative_x = self._forward()["direction_logits"][0, 0].mean(dim=1).argmax()
        self.assertNotEqual(int(positive_x), int(negative_x))

    def test_policy_gradient_reward_sign(self):
        selected_logit = torch.tensor(0.2, requires_grad=True)
        log_probability = torch.log(torch.sigmoid(selected_logit))
        improving_loss = -(+1.0) * log_probability
        improving_grad = torch.autograd.grad(improving_loss, selected_logit, retain_graph=True)[0]
        worsening_loss = -(-1.0) * log_probability
        worsening_grad = torch.autograd.grad(worsening_loss, selected_logit)[0]
        self.assertLess(float(improving_grad), 0.0)
        self.assertGreater(float(worsening_grad), 0.0)

    def test_gumbel_topk_credit_updates_relative_where_ranking(self):
        logits = torch.zeros((1, 1, 4), requires_grad=True)
        selected = torch.tensor([[[True, False, False, False]]])
        log_probability = StructureRepairActuator.network_only_topk_log_prob(
            logits, selected, 0.75
        )
        improving_grad = torch.autograd.grad(
            -(+1.0) * log_probability, logits, retain_graph=True
        )[0]
        worsening_grad = torch.autograd.grad(-(-1.0) * log_probability, logits)[0]
        self.assertLess(float(improving_grad[0, 0, 0]), 0.0)
        self.assertTrue(torch.all(improving_grad[0, 0, 1:] > 0.0))
        self.assertGreater(float(worsening_grad[0, 0, 0]), 0.0)
        self.assertTrue(torch.all(worsening_grad[0, 0, 1:] < 0.0))

    def test_plan_gain_target_is_opposite_of_actual_objective(self):
        dummy = SimpleNamespace(
            args=SimpleNamespace(
                heuristic_guidance_mode="network_only_codec_policy",
                network_only_plan_gain_loss_weight=1.0,
            ),
            last_actuator_voxel_state={},
            last_discrete_policy_debug={},
        )
        bad_prediction = torch.tensor([0.0], requires_grad=True)
        dummy.last_actuator_voxel_state["network_only_predicted_plan_gain"] = bad_prediction
        bad_loss = network_module.Network.network_only_plan_gain_loss(
            dummy, torch.tensor(+0.2)
        )
        bad_grad = torch.autograd.grad(bad_loss, bad_prediction)[0]

        good_prediction = torch.tensor([0.0], requires_grad=True)
        dummy.last_actuator_voxel_state["network_only_predicted_plan_gain"] = good_prediction
        good_loss = network_module.Network.network_only_plan_gain_loss(
            dummy, torch.tensor(-0.2)
        )
        good_grad = torch.autograd.grad(good_loss, good_prediction)[0]

        # Gradient descent lowers the gain for a worsening plan and raises it
        # for an improving plan.
        self.assertGreater(float(bad_grad), 0.0)
        self.assertLess(float(good_grad), 0.0)

    def test_training_samples_one_diverse_plan_and_eval_is_deterministic(self):
        args = _args(network_only_exploration_anneal_steps=2000, _global_train_step=1)
        sampled = []
        for _ in range(8):
            output = self.policy(
                self.features, args, training=True, fixed_features=self.fixed
            )
            sampled.append((
                round(float(output["total_ratio"].detach()), 7),
                tuple(round(float(value), 5) for value in output["shares"].detach().reshape(-1)),
                tuple(int(value) for value in output["priority_order"].detach().reshape(-1)),
                tuple(int(value) for value in output["where_logits"].detach().argmax(dim=2).reshape(-1)),
            ))
            self.assertTrue(output["amount_sample_log_prob"].requires_grad)
            self.assertTrue(output["share_sample_log_prob"].requires_grad)
            self.assertTrue(output["priority_sample_log_prob"].requires_grad)
        self.assertGreater(len(set(sampled)), 1)

        first = self._forward(args)
        second = self._forward(args)
        for key in ("where_logits", "total_ratio", "shares", "priorities", "direction_logits"):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_actual_policy_ema_baseline_preserves_reward_sign(self):
        args = SimpleNamespace(
            heuristic_guidance_mode="network_only_codec_policy",
            heuristic_guidance_online_reward_scale=1.0,
            heuristic_guidance_online_advantage_clip=2.0,
            heuristic_guidance_online_policy_weight=1.0,
            heuristic_guidance_online_entropy_weight=0.0,
            heuristic_guidance_online_reward_ema=0.1,
            network_only_adaptive_entropy_weight=0.0,
            _current_input_file="frame.ply",
            sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2,
            sparsepcgc_scale_m=8,
        )
        key = "frame.ply|0|2|8"
        def gradient(actual):
            logit = torch.tensor(0.2, requires_grad=True)
            dummy = SimpleNamespace(
                args=args,
                last_actuator_voxel_state={
                    "den6_online_policy_log_prob": torch.log(torch.sigmoid(logit)),
                    "den6_online_policy_entropy": torch.tensor(0.0),
                },
                _den6_online_objective_baseline=OrderedDict([(key, 0.0)]),
                last_discrete_policy_debug={},
            )
            loss = network_module.Network.discrete_policy_loss(dummy, torch.tensor(actual))
            return torch.autograd.grad(loss, logit)[0]
        self.assertLess(float(gradient(-0.1)), 0.0)
        self.assertGreater(float(gradient(+0.1)), 0.0)

    def test_worsening_ema_does_not_reward_zero_bit_change(self):
        args = SimpleNamespace(
            heuristic_guidance_mode="network_only_codec_policy",
            heuristic_guidance_online_reward_scale=1.0,
            heuristic_guidance_online_advantage_clip=2.0,
            heuristic_guidance_online_policy_weight=1.0,
            heuristic_guidance_online_entropy_weight=0.0,
            heuristic_guidance_online_reward_ema=0.1,
            network_only_adaptive_entropy_weight=0.0,
            _current_input_file="frame.ply",
            sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2,
            sparsepcgc_scale_m=8,
        )
        key = "frame.ply|0|2|8"
        logit = torch.tensor(0.2, requires_grad=True)
        dummy = SimpleNamespace(
            args=args,
            last_actuator_voxel_state={
                "den6_online_policy_log_prob": torch.log(torch.sigmoid(logit)),
                "den6_online_policy_entropy": torch.tensor(0.0),
            },
            _den6_online_objective_baseline=OrderedDict([(key, +0.2)]),
            last_discrete_policy_debug={},
        )
        loss = network_module.Network.discrete_policy_loss(dummy, torch.tensor(0.0))
        gradient = torch.autograd.grad(loss, logit)[0]
        self.assertEqual(float(gradient), 0.0)
        self.assertEqual(dummy.last_discrete_policy_debug["objective_effective_baseline"], 0.0)


if __name__ == "__main__":
    unittest.main()
