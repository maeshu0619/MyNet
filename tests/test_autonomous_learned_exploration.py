from types import SimpleNamespace
import math
import unittest
import warnings

import torch

from models.utils.training.autonomous_exploration import (
    exploration_controls, install_learned_exploration, learned_exploration_enabled,
)


class LearnedExplorationTest(unittest.TestCase):
    def settings(self, **values):
        args = dict(compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online',
                    test_route=True, test_route_learned_exploration=True,
                    test_route_amount_uniform_fraction=.25,
                    test_route_structured_where_noise=4., sparsepcgc_amount_exploration_log_sigma=.3)
        args.update(values)
        return SimpleNamespace(**args)

    def actor(self, **values):
        actor = torch.nn.Sequential(torch.nn.Linear(2, 1))
        args = self.settings(**values)
        install_learned_exploration(actor, args)
        return actor, args

    def test_false_and_other_codecs_have_no_new_parameter_or_rng_draw(self):
        for values in ({'test_route': False}, {'compress': 'G-PCC'}, {'compress': 'OctAttention'},
                       {'test_route_learned_exploration': False}):
            actor = torch.nn.Sequential(torch.nn.Linear(2, 1))
            before = torch.get_rng_state().clone()
            names = list(actor.state_dict())
            args = self.settings(**values)
            install_learned_exploration(actor, args)
            self.assertFalse(learned_exploration_enabled(args))
            self.assertEqual(names, list(actor.state_dict()))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_initial_broad_scales_are_independent_of_episode_and_dataset(self):
        actor, args = self.actor()
        for dataset in ('8i', 'MVUB', 'UVG'):
            for step in (0, 9549, 9550, 38200):
                args.dataname, args._global_train_step = dataset, step
                values = exploration_controls(actor, args)
                self.assertEqual(float(values['native_std']), 4.)
                self.assertEqual(float(values['amount_temperature']), 2.)
                self.assertAlmostEqual(float(values['fine_std']), .3)
                self.assertEqual(float(values['uniform_fraction']), .25)

    def test_complete_state_roundtrip_and_explicit_legacy_migration(self):
        actor, args = self.actor()
        actor.exploration_log_scales.data.add_(.15)
        other, _ = self.actor()
        other.load_state_dict(actor.state_dict(), strict=True)
        self.assertTrue(torch.equal(actor.exploration_log_scales, other.exploration_log_scales))
        legacy = {k: v for k,v in actor.state_dict().items() if k != 'exploration_log_scales'}
        fresh, _ = self.actor()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            fresh.load_state_dict(legacy, strict=True)
        self.assertTrue(any('Legacy autonomous checkpoint' in str(w.message) for w in observed))
        self.assertEqual(float(exploration_controls(fresh, args)['native_std']), 4.)
        self.assertTrue(torch.equal(fresh[0].weight, actor[0].weight))
        self.assertNotIn('exploration_log_scales', legacy)  # caller's dict is not mutated

    def test_score_gradient_can_learn_smaller_variance_without_an_episode_target(self):
        actor, args = self.actor()
        torch.manual_seed(9)
        controls = exploration_controls(actor, args)
        distribution = torch.distributions.Normal(torch.tensor(0.), controls['native_std'])
        samples = distribution.sample((40000,))
        # Analytic toy cost, NOT an Actual codec measurement. E[x^2]=sigma^2.
        loss = ((samples.square()-16.).detach()*distribution.log_prob(samples)).mean()
        loss.backward()
        self.assertGreater(float(actor.exploration_log_scales.grad[0]), 0.)
        self.assertTrue(torch.isfinite(actor.exploration_log_scales.grad).all())
        optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
        optimizer.step()
        self.assertLess(float(exploration_controls(actor, args)['native_std']), 4.)

    def test_zero_configured_noise_is_respected_and_bounds_remain_positive(self):
        actor, args = self.actor(test_route_structured_where_noise=0.,
                                sparsepcgc_amount_exploration_log_sigma=0.,
                                test_route_amount_uniform_fraction=0.)
        values = exploration_controls(actor, args)
        for key in ('native_std', 'fine_std', 'uniform_fraction'):
            self.assertEqual(float(values[key]), 0.)
        actor.exploration_log_scales.data.fill_(-100.)
        self.assertGreater(float(exploration_controls(actor, args)['where_temperature']), 0.)

    def test_amount_sampling_and_likelihood_update_temperature_fine_and_mix(self):
        from models.utils.training.autonomous_joint_learning import sample_joint_amounts, OPERATIONS
        actor, args = self.actor()
        values = exploration_controls(actor, args)
        means = {op:torch.tensor(.1, requires_grad=True) for op in OPERATIONS}
        torch.manual_seed(7)
        action = sample_joint_amounts(torch.tensor([-.3,.1,.4], requires_grad=True),
            means, torch.zeros(3, requires_grad=True), torch.tensor([0.,.001,.03]), .03,
            1., True, uniform_fraction=values['uniform_fraction'],
            temperature_override=values['amount_temperature'], fine_std_override=values['fine_std'])
        (-sum(action['log_probs'].values())).backward()
        for index in (2,3,4):
            self.assertNotEqual(float(actor.exploration_log_scales.grad[index]), 0.)
        self.assertTrue(torch.isfinite(actor.exploration_log_scales.grad).all())
        self.assertTrue(all(torch.isfinite(v.grad) for v in means.values()))

    def test_where_temperature_likelihood_has_zero_mean_score(self):
        import itertools
        from models.utils.training.autonomous_joint_learning import where_prefix_log_prob
        logits = torch.tensor([.2,-.3,.7], dtype=torch.float64)
        temperature = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
        total = sum(where_prefix_log_prob(logits, torch.tensor(order), 2, 1., True,
            temperature_override=temperature).exp() for order in itertools.permutations(range(3)))
        self.assertAlmostEqual(float(total), 1., places=12)
        self.assertAlmostEqual(float(torch.autograd.grad(total, temperature)[0]), 0., places=12)

    def test_greedy_evaluation_ignores_the_new_widths_and_has_no_sampling_credit(self):
        from models.utils.training.autonomous_joint_learning import sample_joint_amounts, sample_where_order, OPERATIONS
        actor,args = self.actor()
        values = exploration_controls(actor,args)
        means = {op:torch.tensor(.1) for op in OPERATIONS}
        initial = torch.get_rng_state().clone()
        action = sample_joint_amounts(torch.tensor([-.3,.1,.4]),means,torch.zeros(3),
            torch.tensor([0.,.001,.03]),.03,0.,False,uniform_fraction=values['uniform_fraction'],
            temperature_override=values['amount_temperature'],fine_std_override=values['fine_std'])
        order = sample_where_order(torch.tensor([1.,2.,0.]),0.,False,
                                  temperature_override=values['where_temperature'])
        self.assertEqual(order.tolist(),[1,0,2])
        self.assertTrue(all(float(v)==0. for v in action['log_probs'].values()))
        self.assertTrue(torch.equal(initial,torch.get_rng_state()))

    def test_exploration_coordinate_clip_keeps_other_small_updates(self):
        from models.utils.training.autonomous_joint_learning import clip_joint_gradients
        model = torch.nn.Module();model.args=self.settings()
        model.actuator=torch.nn.Module()
        model.actuator.den6_candidate_actor,args=self.actor()
        for p in model.parameters():p.grad=torch.full_like(p,.001)
        parameter=model.actuator.den6_candidate_actor.exploration_log_scales
        parameter.grad[1]=1e6
        report=clip_joint_gradients(model,10.)
        self.assertEqual(float(parameter.grad[1]),1.)
        self.assertAlmostEqual(float(parameter.grad[0]),.001)
        self.assertIsNotNone(report['autonomous_exploration_clip'])

    def test_better_current_actual_plan_is_not_trained_back_to_an_old_teacher(self):
        from tests.test_autonomous_codec_context import AutonomousRecoveryTest
        from models.utils.training.autonomous_actual_replay import measured_plan_replay
        debug,previous,args=AutonomousRecoveryTest().fixture()
        args.__dict__.update(vars(self.settings()))
        loss,replacement,report=measured_plan_replay(debug,previous,-5.,.01,args)
        self.assertEqual(float(loss),0.)
        self.assertFalse(report['used'])
        self.assertIsNotNone(replacement)
        self.assertEqual(report['reason'],'current_measured_rd_not_worse_than_teacher')

    def test_new_checkpoint_restores_adam_moments_including_widths(self):
        actor,args=self.actor();optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
        sum(p.square().sum() for p in actor.parameters()).backward();optimizer.step()
        target,_=self.actor();target.load_state_dict(actor.state_dict(),strict=True)
        restored=torch.optim.Adam(target.parameters(),lr=1e-4)
        restored.load_state_dict(optimizer.state_dict())
        self.assertTrue(torch.equal(restored.state[target.exploration_log_scales]['exp_avg'],
                                    optimizer.state[actor.exploration_log_scales]['exp_avg']))

    def test_clean_amount_and_noisy_where_use_separate_past_frame_baselines(self):
        from tests.test_autonomous_joint_learning import JointLearningTest
        from models.utils.training.autonomous_joint_learning import joint_actual_policy_loss
        model,_,_=JointLearningTest().model()
        model.args.__dict__.update(vars(self.settings()))
        debug=model.last_actuator_voxel_state['ana_den6_exact_residual_plan_debug']
        debug['joint_credit_family']='where_sampled'
        joint_actual_policy_loss(model,torch.tensor(50.),torch.tensor(.01))
        debug['joint_credit_family']='amount_clean_where'
        joint_actual_policy_loss(model,torch.tensor(-4.),torch.tensor(.01))
        self.assertEqual(model.last_discrete_policy_debug['objective_baseline_source'],
                         'zero_family_control_variate_not_measurement')
        joint_actual_policy_loss(model,torch.tensor(-5.),torch.tensor(.01))
        self.assertAlmostEqual(model.last_discrete_policy_debug['baseline'],-12.+1.5,places=5)


if __name__ == '__main__':
    unittest.main()
