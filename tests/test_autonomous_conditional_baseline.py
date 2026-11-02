import copy
import io
import unittest

import torch

from models.utils.training.autonomous_joint_learning import joint_actual_policy_loss
from tests import test_autonomous_joint_learning as joint_fixture
from tests import test_autonomous_learned_exploration as learned_fixture


class ConditionalWhereBaselineTest(unittest.TestCase):
    def fixture(self):
        model, _, _ = joint_fixture.JointLearningTest().model()
        model.args.__dict__.update(vars(learned_fixture.LearnedExplorationTest().settings()))
        model.args.test_route_where_conditional_baseline = True
        position = torch.tensor(-2., requires_grad=True)
        native = torch.tensor(-1., requires_grad=True)
        debug = model.last_actuator_voxel_state['ana_den6_exact_residual_plan_debug']
        debug.update(_joint_policy_log_prob=position+native,
                     _joint_where_position_log_prob=position,
                     _joint_native_sample_log_prob=native,
                     joint_credit_family='where_sampled')
        return model, debug, position, native

    def observe(self, model, debug, sign, c, g):
        debug['structured_where_coefficients'] = {op:float(sign) for op in ('Add','Prune','Adjust')}
        return joint_actual_policy_loss(model, torch.tensor(c), torch.tensor(g))

    def test_only_past_same_frame_same_native_condition_informs_where(self):
        model, debug, position, native = self.fixture()
        self.observe(model, debug, -1., 50., .02)
        self.assertFalse(model.last_discrete_policy_debug['where_control_variate']['available'])
        self.observe(model, debug, 1., -4., .01)
        self.assertFalse(model.last_discrete_policy_debug['where_control_variate']['available'])
        self.observe(model, debug, -1., 55., .025)
        report = model.last_discrete_policy_debug
        self.assertTrue(report['where_control_variate']['available'])
        self.assertAlmostEqual(report['where_baseline'], 3*50.+150*.02, places=5)
        c = model._den6_policy_source_losses['actual_compression_policy']
        g = model._den6_policy_source_losses['geometry']
        where_c, native_c = torch.autograd.grad(c, (position,native), retain_graph=True)
        where_g, native_g = torch.autograd.grad(g, (position,native))
        self.assertAlmostEqual(float(where_c), 3*(55.-50.), places=4)
        self.assertAlmostEqual(float(native_c), 3*(55.-44.6), places=4)
        self.assertAlmostEqual(float(where_g), 150*(.025-.02), places=5)
        self.assertAlmostEqual(float(native_g), 150*(.025-.019), places=5)

    def test_native_score_is_unchanged_and_legacy_flag_disables_correction(self):
        model, debug, position, native = self.fixture()
        self.observe(model, debug, -1., 50., .02)
        self.observe(model, debug, 1., -4., .01)
        memory = copy.deepcopy(model._den6_online_efficiency_baseline)
        self.observe(model, debug, -1., 55., .025)
        c = model._den6_policy_source_losses['actual_compression_policy']
        changed = torch.autograd.grad(c, (position,native), retain_graph=True)
        model._den6_online_efficiency_baseline = memory
        model.args.test_route_where_conditional_baseline = False
        self.observe(model, debug, -1., 55., .025)
        c = model._den6_policy_source_losses['actual_compression_policy']
        original = torch.autograd.grad(c, (position,native))
        self.assertTrue(torch.equal(changed[1], original[1]))
        self.assertNotEqual(float(changed[0]),float(original[0]))

    def test_clean_amount_trials_preserve_history_without_using_it(self):
        model, debug, _, _ = self.fixture()
        self.observe(model, debug, -1., 50., .02)
        original = copy.deepcopy(next(iter(model._den6_online_efficiency_baseline.values()))['where_native_condition_ema'])
        debug['joint_credit_family'] = 'amount_clean_where'
        self.observe(model, debug, 1., -4., .01)
        self.assertFalse(model.last_discrete_policy_debug['where_control_variate']['enabled'])
        self.assertEqual(next(iter(model._den6_online_efficiency_baseline.values()))['where_native_condition_ema'], original)
        debug['joint_credit_family'] = 'where_sampled'
        debug['autonomous_input_sha256'] = 'b'*64
        self.observe(model, debug, -1., 55., .025)
        self.assertFalse(model.last_discrete_policy_debug['where_control_variate']['available'])

    def test_disabled_routes_and_old_debug_have_no_conditional_history(self):
        for disabled in ({'test_route':False},{'compress':'G-PCC'}, {'compress':'OctAttention'},
                         {'test_route_learned_exploration':False}):
            model, debug, _, _ = self.fixture()
            model.args.__dict__.update(disabled)
            self.observe(model, debug, -1., 50., .02)
            self.assertFalse(model.last_discrete_policy_debug['where_control_variate']['enabled'])
            self.assertNotIn('where_native_condition_ema',next(iter(model._den6_online_efficiency_baseline.values())))
        model, debug, _, _ = self.fixture()
        del debug['_joint_where_position_log_prob']
        self.observe(model, debug, -1., 50., .02)
        self.assertFalse(model.last_discrete_policy_debug['where_control_variate']['enabled'])

    def test_history_roundtrip_and_first_observation_preserve_gradients(self):
        model, debug, position, native = self.fixture()
        self.observe(model, debug, -1., 50., .02)
        original = torch.autograd.grad(model._den6_policy_source_losses['actual_compression_policy'],
                                       (position,native),retain_graph=True)
        first_memory = copy.deepcopy(model._den6_online_efficiency_baseline)
        model._den6_online_efficiency_baseline.clear()
        model.args.test_route_where_conditional_baseline = False
        self.observe(model, debug, -1., 50., .02)
        legacy = torch.autograd.grad(model._den6_policy_source_losses['actual_compression_policy'],
                                     (position,native),retain_graph=True)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(original,legacy)))
        buffer = io.BytesIO();torch.save(first_memory,buffer);buffer.seek(0)
        restored = torch.load(buffer)
        self.assertEqual(restored,first_memory)
        model._den6_online_efficiency_baseline = restored
        model.args.test_route_where_conditional_baseline = True
        self.observe(model, debug, -1., 51., .02)
        self.assertTrue(model.last_discrete_policy_debug['where_control_variate']['available'])
        self.assertAlmostEqual(model.last_discrete_policy_debug['where_baseline'],153.,places=5)

    def test_enumerated_gradient_is_unbiased_but_native_must_keep_unconditional_baseline(self):
        native_logits = torch.tensor([.2,-.1], dtype=torch.float64, requires_grad=True)
        where_logits = torch.tensor([[.2,-.3,.1],[.1,.4,-.2]], dtype=torch.float64, requires_grad=True)
        cost = torch.tensor([[45.,55.,40.],[-5.,-4.,-3.]], dtype=torch.float64)
        pn, pw = native_logits.softmax(0), where_logits.softmax(1)
        probability = pn[:,None]*pw
        objective = (probability*cost).sum()
        target = torch.autograd.grad(objective,(native_logits,where_logits),retain_graph=True)
        baseline = 20.
        conditional = (pw.detach()*cost).sum(1)
        corrected, incorrect = 0., 0.
        for z in range(2):
            for a in range(3):
                weight = probability[z,a].detach()
                corrected = corrected + weight*((cost[z,a]-baseline)*pn[z].log()
                            +(cost[z,a]-conditional[z])*pw[z,a].log())
                incorrect = incorrect + weight*(cost[z,a]-conditional[z])*(pn[z].log()+pw[z,a].log())
        actual = torch.autograd.grad(corrected,(native_logits,where_logits),retain_graph=True)
        wrong = torch.autograd.grad(incorrect,native_logits)[0]
        for a,b in zip(actual,target):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        self.assertGreater(float((wrong-target[0]).abs().max()),1.)
        # Compute exact per-sample Where-gradient second moments as well.
        moment = {'family':0.,'conditional':0.}
        for z in range(2):
            for a in range(3):
                score = -pw[z].detach().clone();score[a] += 1.
                for name,b in (('family',baseline),('conditional',conditional[z])):
                    moment[name] += float(probability[z,a].detach()*(cost[z,a]-b).square()*score.square().sum())
        self.assertLess(moment['conditional'],moment['family'])


if __name__ == '__main__':
    unittest.main()
