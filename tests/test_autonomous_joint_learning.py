from collections import OrderedDict
from types import SimpleNamespace
import itertools
import unittest

import torch

from models.utils.training.autonomous_joint_learning import (
    OPERATIONS, sample_joint_amounts, sample_where_order, where_prefix_log_prob,
    joint_actual_policy_loss,
)


class JointLearningTest(unittest.TestCase):
    def sample(self, seed=0, phase=1., training=True):
        torch.manual_seed(seed)
        selector = torch.zeros(6, requires_grad=True)
        means = {name: torch.zeros((), requires_grad=True) for name in OPERATIONS}
        gate = torch.zeros(3, requires_grad=True)
        result = sample_joint_amounts(selector, means, gate,
            torch.tensor([0., .001, .009, .01, .015, .03]), .03, phase, training)
        return result, selector, means, gate

    def test_all_six_actions_are_reachable_for_each_operation(self):
        seen = {op: set() for op in OPERATIONS}
        distinct = set()
        for seed in range(100):
            result, _, _, _ = self.sample(seed)
            indexes = tuple(int(result['indices'][op]) for op in OPERATIONS)
            distinct.add(indexes)
            for op in OPERATIONS:
                seen[op].add(int(result['indices'][op]))
                self.assertTrue(0 <= result['ratios'][op] <= .03)
        self.assertTrue(all(values == set(range(6)) for values in seen.values()))
        self.assertGreater(len(distinct), 40)  # not one shared Amount bin

    def test_each_family_can_use_three_percent_simultaneously(self):
        means = {name: torch.tensor(0.) for name in OPERATIONS}
        result = sample_joint_amounts(torch.tensor([-10., -10., -10., -10., -10., 10.]),
            means, torch.zeros(3), torch.tensor([0., .001, .009, .01, .015, .03]), .03, 0., False)
        self.assertAlmostEqual(sum(float(v) for v in result['ratios'].values()), .09, places=6)
        self.assertTrue(all(float(v) > .0299 for v in result['ratios'].values()))

    def test_explicit_zero_fine_noise_is_respected(self):
        means = {name: torch.tensor(0.) for name in OPERATIONS}
        bins = torch.tensor([0., .001, .009, .01, .015, .03])
        torch.manual_seed(3)
        result = sample_joint_amounts(torch.zeros(6), means, torch.zeros(3), bins,
                                      .03, 1., True, fine_sigma=0.)
        for name in OPERATIONS:
            self.assertEqual(float(result['ratios'][name]), float(bins[result['indices'][name]]))

    def test_noop_is_per_operation_not_a_forced_plan_quota(self):
        for seed in range(100):
            result, _, _, _ = self.sample(seed)
            values = [float(v) for v in result['ratios'].values()]
            if 0. in values and any(v > 0 for v in values):
                break
        else:
            self.fail('no independent no-op action was sampled')

    def test_likelihood_updates_all_amount_heads_and_shared_selector(self):
        # Select a seed with three positive actions so Fine is causal for all.
        for seed in range(100):
            result, selector, means, gate = self.sample(seed)
            if all(int(v) > 0 for v in result['indices'].values()):
                break
        loss = -sum(result['log_probs'].values())
        loss.backward()
        self.assertGreater(float(selector.grad.norm()), 0.)
        self.assertGreater(float(gate.grad.norm()), 0.)
        self.assertTrue(all(abs(float(v.grad)) > 0 for v in means.values()))

    def test_eval_and_zero_noise_are_deterministic(self):
        for phase, training in ((0., True), (1., False)):
            a, _, _, _ = self.sample(1, phase, training)
            b, _, _, _ = self.sample(99, phase, training)
            self.assertFalse(a['stochastic'])
            self.assertEqual([int(v) for v in a['indices'].values()], [int(v) for v in b['indices'].values()])
            self.assertTrue(all(float(v) == 0 for v in a['log_probs'].values()))

    def test_prefix_likelihood_matches_enumerated_without_replacement(self):
        logits = torch.tensor([.2, -.3, .7], dtype=torch.float64, requires_grad=True)
        total = logits.new_zeros(())
        for order in itertools.permutations(range(3)):
            order = torch.tensor(order)
            lp = where_prefix_log_prob(logits, order, 2, 1., True)
            manual = (torch.log_softmax(logits, 0)[order[0]]
                      + torch.log_softmax(logits[order[1:]], 0)[0])
            self.assertTrue(torch.allclose(lp, manual))
            total = total + lp.exp()
        self.assertAlmostEqual(float(total), 1., places=12)

    def test_rejected_prefix_candidates_remain_in_likelihood(self):
        logits = torch.tensor([.1, .2, .3], requires_grad=True)
        order = torch.tensor([1, 2, 0])
        lp = where_prefix_log_prob(logits, order, 2, 1., True)
        lp.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertNotEqual(float(logits.grad[1]), 0.)
        self.assertNotEqual(float(lp), float(where_prefix_log_prob(logits, order, 1, 1., True)))

    def test_where_sampling_is_repeatable_and_covers_lower_rank(self):
        logits = torch.tensor([1., .5, 0., -.5])
        first = set()
        for seed in range(50):
            torch.manual_seed(seed)
            order = sample_where_order(logits, 1., True)
            first.add(int(order[0]))
        self.assertEqual(first, set(range(4)))
        self.assertEqual(sample_where_order(logits, 0., True).tolist(), [0, 1, 2, 3])

    def model(self):
        lp = torch.tensor(-2., requires_grad=True)
        feature = torch.randn(18, requires_grad=True)
        debug = dict(_joint_policy_log_prob=lp, _joint_entropy=lp * 0.,
                     autonomous_input_sha256='a'*64, exploration_multiplier=1.,
                     _actor_critic_selected_operation_ratio=torch.tensor(.01),
                     _actor_critic_plan_feature=feature)
        model = SimpleNamespace(args=SimpleNamespace(w_com=3., cp_lambda_geom=150.),
            actuator=SimpleNamespace(den6_plan_critic=torch.nn.Linear(18, 2)),
            last_actuator_voxel_state={'ana_den6_exact_residual_plan_debug': debug})
        return model, lp, feature

    def test_actual_compression_and_geometry_have_correct_cost_gradient_sign(self):
        model, lp, feature = self.model()
        loss = joint_actual_policy_loss(model, torch.tensor(2.), torch.tensor(.01))
        c = torch.autograd.grad(model._den6_policy_source_losses['actual_compression_policy'], lp, retain_graph=True)[0]
        g = torch.autograd.grad(model._den6_policy_source_losses['geometry'], lp, retain_graph=True)[0]
        # Gradient descent lowers the probability of this positive-cost action.
        self.assertGreater(float(c), 0.)
        self.assertGreater(float(g), 0.)
        loss.backward()
        # Critic cannot update action/Where-feature slots through its input.
        self.assertEqual(float(feature.grad[12:].abs().sum()), 0.)
        self.assertGreater(float(feature.grad[[0, 1, 4, 5, 8, 9]].abs().sum()), 0.)

    def test_efficiency_never_rewards_quality_regression(self):
        model, lp, feature = self.model()
        joint_actual_policy_loss(model, torch.tensor(-3.), torch.tensor(.02))
        debug = model.last_actuator_voxel_state['ana_den6_exact_residual_plan_debug']
        debug['_actor_critic_selected_operation_ratio'] = torch.tensor(.005)
        joint_actual_policy_loss(model, torch.tensor(-2.), torch.tensor(.01))
        self.assertFalse(model.last_discrete_policy_debug['efficiency_guard_passed'])
        self.assertEqual(float(model._den6_policy_source_losses['efficiency']), 0.)
        joint_actual_policy_loss(model, torch.tensor(-3.), torch.tensor(.009))
        self.assertTrue(model.last_discrete_policy_debug['efficiency_guard_passed'])

    def test_baselines_are_isolated_by_input(self):
        model, _, _ = self.model()
        joint_actual_policy_loss(model, torch.tensor(-3.), torch.tensor(.02))
        model.last_actuator_voxel_state['ana_den6_exact_residual_plan_debug']['autonomous_input_sha256'] = 'b'*64
        joint_actual_policy_loss(model, torch.tensor(-2.), torch.tensor(.01))
        self.assertEqual(model.last_discrete_policy_debug['objective_baseline_source'], 'zero_control_variate_not_measurement')

    def test_large_actor_gradient_does_not_shrink_small_amount_or_critic_gradients(self):
        from models.utils.training.optim_amp import clip_model_gradients
        from models.utils.training.autonomous_joint_learning import clip_joint_gradients
        model = torch.nn.Module()
        model.args = SimpleNamespace(test_route=True, compress='SparsePCGC',
            heuristic_guidance_mode='ana_den6_online', repair_online_decision_grad_max_norm=1.)
        model.actuator = torch.nn.Module()
        for name in ('den6_candidate_actor', 'den6_candidate_proposal', 'add_amount_head',
                     'den6_candidate_critic', 'subtree_move_source_head'):
            setattr(model.actuator, name, torch.nn.Linear(2, 1))
        model.backbone = torch.nn.Linear(2, 1)
        for p in model.parameters():
            p.grad = torch.full_like(p, .001)
        for p in model.actuator.den6_candidate_actor.parameters():
            p.grad.fill_(1e6)
        for p in model.backbone.parameters():
            p.grad.fill_(1e6)
        expected = model.actuator.add_amount_head.weight.grad.clone()
        expected_critic = model.actuator.den6_candidate_critic.weight.grad.clone()
        report = clip_model_gradients(model, 10.)
        self.assertTrue(torch.equal(model.actuator.add_amount_head.weight.grad, expected))
        self.assertTrue(torch.equal(model.actuator.den6_candidate_critic.weight.grad, expected_critic))
        self.assertLessEqual(report['autonomous_gradient_groups']['actor_proposal']['after'], 1.000001)
        self.assertLessEqual(report['autonomous_gradient_groups']['backbone']['after'], 10.00001)


if __name__ == '__main__':
    unittest.main()
