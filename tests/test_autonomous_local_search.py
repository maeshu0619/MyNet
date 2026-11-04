"""Distribution, credit and route isolation checks; no synthetic Actual claims."""
import copy
import itertools
import json
import math
from types import SimpleNamespace
import unittest
import zlib

import torch

from models.utils.training.autonomous_local_search import (
    phased_search_enabled, search_stage, policy_identity, policy_memory_key,
    prepare_neighbourhood, neighbourhood_controls, amount_neighbourhood,
    where_neighbourhood,
)
from models.utils.training.autonomous_joint_learning import (
    OPERATIONS, sample_joint_amounts, sample_where_order, where_prefix_log_prob,
)


class LocalSearchTest(unittest.TestCase):
    def fixture(self, step=12000):
        args = SimpleNamespace(compress='SparsePCGC', test_route=True,
            heuristic_guidance_mode='ana_den6_online', test_route_learned_exploration=True,
            test_route_phased_exploration=True, test_route_broad_search_episodes=60,
            test_route_local_search_episodes=120, _global_train_step=step,
            _fixed_episode_steps=100, _current_input_file='a.ply',
            _autonomous_native_context_path='/private/context.npz',
            sparsepcgc_algorithmic_amount_bin_values=(.001, .002, .003, .01, .03),
            w_com=3., cp_lambda_geom=150., cp_tau_geom=0.)
        exact = dict(input_sha256='a'*64, setting_id='codec-a',
            operation_edit_units={op: [dict(candidate_id=f'{op}{i}') for i in range(32)]
                                  for op in OPERATIONS})
        action = dict(amount_indices={op: 2 for op in OPERATIONS},
            fine_samples={op: .1 for op in OPERATIONS},
            where_prefix_ids={op: [f'{op}{i}' for i in range(4)] for op in OPERATIONS})
        best = dict(compression=-5., geometry=.01, action=action)
        memory = {policy_memory_key(args, policy_identity(args, exact)):
                  dict(best_action=zlib.compress(json.dumps(best).encode()))}
        return args, exact, memory

    def controls(self):
        return {k: torch.tensor(v, requires_grad=True) for k, v in dict(
            native_std=4., where_temperature=1., amount_temperature=2.,
            fine_std=.3, uniform_fraction=.25).items()}

    def test_first_sixty_episodes_leave_outputs_and_rng_identical(self):
        # Directly check the behaviour helpers, not only the schedule flag.
        for step in (0, 99, 2000, 5999, 6000):
            args, exact, memory = self.fixture(step)
            guide = prepare_neighbourhood(args, exact, memory, True)
            controls = self.controls()
            self.assertIs(neighbourhood_controls(controls, guide), controls)
            self.assertFalse(guide['active'])
            selector = torch.tensor([-.2, .1, .3, .1, -.4, -.3])
            means = {op: torch.tensor(.1) for op in OPERATIONS}
            bins = torch.tensor([0., .001, .002, .003, .01, .03])
            torch.manual_seed(11)
            a = sample_joint_amounts(selector, means, torch.zeros(3), bins, .03,
                1., True, uniform_fraction=.25)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(11)
            b = sample_joint_amounts(selector, means, torch.zeros(3), bins, .03,
                1., True, uniform_fraction=.25, neighbourhood=guide)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for op in OPERATIONS:
                for field in ('ratios', 'log_probs', 'samples', 'logits'):
                    self.assertTrue(torch.equal(a[field][op], b[field][op]))

    def test_continuous_boundary_and_nonzero_residual_controls(self):
        args, exact, memory = self.fixture(6000)
        self.assertEqual(search_stage(args)['strength'], 0.)
        args._global_train_step = 6001
        self.assertLess(search_stage(args)['strength'], 1e-6)
        observed = []
        for step in (6001, 8000, 10000, 12000, 16000, 18000, 50000):
            args._global_train_step = step
            guide = prepare_neighbourhood(args, exact, memory, True)
            values = neighbourhood_controls(self.controls(), guide)
            observed.append([float(v) for v in values.values()])
            self.assertTrue(all(float(v) > 0. for v in values.values()))
            sum(values.values()).backward()
        self.assertTrue(all(all(x >= y for x, y in zip(a, b))
                            for a, b in zip(observed, observed[1:])))
        self.assertEqual(observed[-1], observed[-2])

    def test_no_measured_compatible_teacher_keeps_broad_search(self):
        args, exact, memory = self.fixture(20000)
        for change in ('missing', 'different_frame', 'different_gt', 'different_codec', 'missing_id'):
            a, e, m = copy.deepcopy((args, exact, memory))
            if change == 'missing': m = {}
            if change == 'different_frame': a._current_input_file = 'other.ply'
            if change == 'different_gt': e['input_sha256'] = 'b'*64
            if change == 'different_codec': e['setting_id'] = 'other'
            if change == 'missing_id': e['operation_edit_units']['Prune'] = []
            guide = prepare_neighbourhood(a, e, m, True)
            self.assertFalse(guide['active'], change)
            controls = self.controls()
            self.assertIs(neighbourhood_controls(controls, guide), controls)

    def test_nonprofitable_and_corrupt_teachers_are_not_guides(self):
        args, exact, memory = self.fixture()
        key = next(iter(memory))
        for c, g in ((1., .01), (-1., 1.), (float('nan'), .01)):
            best = json.loads(zlib.decompress(memory[key]['best_action']))
            best.update(compression=c, geometry=g)
            memory[key]['best_action'] = zlib.compress(json.dumps(best).encode())
            self.assertFalse(prepare_neighbourhood(args, exact, memory, True)['active'])
        memory[key]['best_action'] = b'corrupt'
        self.assertFalse(prepare_neighbourhood(args, exact, memory, True)['active'])

    def test_eval_and_other_routes_never_use_the_measurement(self):
        args, exact, memory = self.fixture(20000)
        rng = torch.get_rng_state().clone()
        self.assertFalse(prepare_neighbourhood(args, exact, memory, False)['active'])
        for key, value in (('test_route', False), ('compress', 'G-PCC'),
                           ('compress', 'OctAttention'), ('test_route_learned_exploration', False),
                           ('test_route_phased_exploration', False)):
            other = copy.copy(args); setattr(other, key, value)
            self.assertFalse(phased_search_enabled(other))
            self.assertFalse(prepare_neighbourhood(other, exact, memory, True)['active'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_eval_is_unchanged_even_if_a_caller_passes_an_active_guide(self):
        args, exact, memory = self.fixture(20000)
        guide = prepare_neighbourhood(args, exact, memory, True)
        means = {op: torch.tensor(.2) for op in OPERATIONS}
        bins = torch.tensor([0., .001, .002, .003, .01, .03])
        args = (torch.tensor([-.1, .4, .0, .1, .0, .0]), means, torch.zeros(3),
                bins, .03, 0., False)
        before = torch.get_rng_state().clone()
        a = sample_joint_amounts(*args)
        b = sample_joint_amounts(*args, neighbourhood=guide)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for op in OPERATIONS:
            self.assertTrue(torch.equal(a['ratios'][op], b['ratios'][op]))
            self.assertEqual(float(b['log_probs'][op]), 0.)

    def test_disabled_noise_and_masked_candidates_are_preserved(self):
        args, exact, memory = self.fixture(20000)
        guide = prepare_neighbourhood(args, exact, memory, True)
        controls = self.controls()
        controls['native_std'] = torch.tensor(0.)
        controls['uniform_fraction'] = torch.tensor(0.)
        effective = neighbourhood_controls(controls, guide)
        self.assertEqual(float(effective['native_std']), 0.)
        self.assertEqual(float(effective['uniform_fraction']), 0.)
        logits = torch.zeros(32); logits[0] = -float('inf')
        scores = where_neighbourhood(logits, 'Add', guide, effective['where_temperature'])
        self.assertFalse(torch.isfinite(scores[0]))
        order = sample_where_order(scores, 1., True, effective['where_temperature'])
        self.assertNotIn(0, order.tolist())

    def test_late_exploration_scale_still_receives_a_real_optimizer_update(self):
        from models.utils.training.autonomous_exploration import (
            install_learned_exploration, exploration_controls)
        args, exact, memory = self.fixture(25000)
        actor = torch.nn.Linear(2, 1)
        install_learned_exploration(actor, args)
        guide = prepare_neighbourhood(args, exact, memory, True)
        controls = neighbourhood_controls(exploration_controls(actor, args), guide)
        distribution = torch.distributions.Normal(torch.tensor(.5), controls['native_std'])
        loss = -distribution.log_prob(torch.tensor(.56))
        before = actor.exploration_log_scales.detach().clone()
        loss.backward()
        self.assertTrue(torch.isfinite(actor.exploration_log_scales.grad).all())
        self.assertNotEqual(float(actor.exploration_log_scales.grad[0]), 0.)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
        optimizer.step()
        self.assertNotEqual(float(actor.exploration_log_scales[0]), float(before[0]))

    def test_mixture_likelihood_and_parameter_updates_match_sampled_policy(self):
        args, exact, memory = self.fixture(18000)
        guide = prepare_neighbourhood(args, exact, memory, True)
        controls = neighbourhood_controls(self.controls(), guide)
        selector = torch.zeros(6, requires_grad=True)
        means = {op: torch.tensor(.0, requires_grad=True) for op in OPERATIONS}
        gates = torch.zeros(3, requires_grad=True)
        bins = torch.tensor([0., .001, .002, .003, .01, .03])
        torch.manual_seed(7)
        result = sample_joint_amounts(selector, means, gates, bins, .03, 1., True,
            uniform_fraction=controls['uniform_fraction'],
            temperature_override=controls['amount_temperature'],
            fine_std_override=controls['fine_std'], neighbourhood=guide)
        for i, op in enumerate(OPERATIONS):
            logits = selector + means[op]*bins/.03 + gates[i]*(bins > 0.)
            logits, mean = amount_neighbourhood(logits, means[op], bins, op, guide,
                                               controls['amount_temperature'])
            index, sample = result['indices'][op], result['samples'][op].detach()
            normal = torch.distributions.Normal(mean, controls['fine_std']).log_prob(sample)
            uniform_normal = torch.distributions.Normal(mean*0., controls['fine_std']).log_prob(sample)
            policy = torch.log_softmax(logits/controls['amount_temperature'], 0)[index]
            broad = -math.log(len(bins)) + (uniform_normal if int(index) else 0.)
            policy = policy + (normal if int(index) else 0.)
            expected = torch.logaddexp(policy+torch.log1p(-controls['uniform_fraction']),
                                       broad+controls['uniform_fraction'].log())
            self.assertAlmostEqual(float(result['log_probs'][op]), float(expected), places=5)
            self.assertIs(result['means'][op], means[op])  # replay trains the unguided head
        params = [selector, gates] + list(means.values())
        before = [p.detach().clone() for p in params]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        (-sum(result['log_probs'].values())).backward()
        for p in params:
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.norm()), 0.)
        optimizer.step()
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(params, before)))

    def test_where_is_local_but_alternatives_and_score_gradients_remain(self):
        args, exact, memory = self.fixture(18000)
        guide = prepare_neighbourhood(args, exact, memory, True)
        controls = neighbourhood_controls(self.controls(), guide)
        logits = torch.zeros(32, requires_grad=True)
        temperature = controls['where_temperature']
        scores = where_neighbourhood(logits, 'Add', guide, temperature)
        torch.manual_seed(9)
        overlap, changed = 0, 0
        for _ in range(300):
            order = sample_where_order(scores, 1., True, temperature)
            n = sum(int(i) < 4 for i in order[:4])
            overlap += n
            changed += n < 4
        self.assertGreater(overlap / 1200., .65)
        self.assertGreater(changed, 0)  # never hard-replay the measured prefix
        loss = -where_prefix_log_prob(scores, order, 4, 1., True, temperature)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()), 0.)

    def test_guided_where_probability_normalizes_and_has_zero_expected_score(self):
        args, exact, memory = self.fixture(18000)
        guide = prepare_neighbourhood(args, exact, memory, True)
        guide['prefix_indices']['Add'] = [0]
        logits = torch.tensor([.1, -.3, .2], dtype=torch.float64, requires_grad=True)
        temperature = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
        scores = where_neighbourhood(logits, 'Add', guide, temperature)
        total = sum(where_prefix_log_prob(scores, torch.tensor(order), 2, 1., True,
            temperature).exp() for order in itertools.permutations(range(3)))
        self.assertAlmostEqual(float(total), 1., places=12)
        for grad in torch.autograd.grad(total, (logits, temperature)):
            self.assertLess(float(grad.abs().max()), 1e-12)

    def test_reservoir_only_stops_rotating_after_the_wide_phase(self):
        from models.utils.training.sparsepcgc_autonomous_policy import autonomous_pool_phase
        args, _, _ = self.fixture()
        args.test_route_codec_context = True
        args.sparsepcgc_mode = 'dense_lossy'
        args.test_route_context_reservoir_passes = 8
        for step, expected in ((0, 0), (799, 0), (800, 1), (5999, 7), (6000, 7), (10000, 7)):
            args._global_train_step = step
            self.assertEqual(autonomous_pool_phase(args), expected)


if __name__ == '__main__':
    unittest.main()
