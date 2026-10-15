from collections import Counter, OrderedDict
from types import SimpleNamespace
import unittest

import torch

from models.modules.structure_actuator import (
    StructureRepairActuator,
    amount_forward_authority,
    matched_plan_comparison_hash,
    normalized_fine_log_prob,
)


class AmountLearningReachabilityTest(unittest.TestCase):
    def test_where_all_comparison_freezes_counts_but_not_candidate_ids(self):
        ids = {'Add': ('a0',), 'Prune': ('p0',), 'Adjust': ('m0',)}
        changed = {'Add': ('a1',), 'Prune': ('p1',), 'Adjust': ('m1',)}
        counts = {'Add': 1, 'Prune': 1, 'Adjust': 1}
        order = ('Prune', 'Add', 'Adjust')
        def identity(group, selected, requested=counts, pool='pool'):
            return matched_plan_comparison_hash(
                group, selected, counts, requested, order, pool
            )
        self.assertEqual(identity('WhereAll', ids), identity('WhereAll', changed))
        self.assertNotEqual(identity('AddWhere', ids), identity('AddWhere', changed))
        self.assertNotEqual(identity('WhereAll', ids), identity('WhereAll', ids, pool='other'))
        self.assertNotEqual(identity('WhereAll', ids), identity(
            'WhereAll', ids, requested={**counts, 'Prune': 2}
        ))

    def test_every_amount_bin_can_win_with_bounded_logits(self):
        bins = torch.tensor([0.0015, 0.002, 0.0025, 0.00275, 0.003])
        prior = -(bins.log() - torch.tensor(0.0025).log()).abs() / 0.35
        prior = torch.cat((prior.max().view(1) - 1.5, prior))
        old_authority = amount_forward_authority(0.05, 0.0, 0.0)
        new_authority = amount_forward_authority(0.05, 0.0, 0.75)
        old_winners = set()
        new_winners = set()
        for target in range(6):
            logits = torch.full((6,), -100.0)
            logits[target] = 100.0
            centered = logits - logits.mean()
            bounded = 0.35 * torch.nn.functional.softsign(centered / 0.35)
            old_winners.add(int(((1-old_authority)*prior + old_authority*bounded).argmax()))
            new_winners.add(int(((1-new_authority)*prior + new_authority*bounded).argmax()))
        self.assertEqual(old_winners, {3})
        self.assertEqual(new_winners, set(range(6)))

    def test_large_amount_logits_keep_finite_nonzero_gradients(self):
        raw = torch.tensor([100.0, -100.0], requires_grad=True)
        bounded = 0.35 * torch.nn.functional.softsign(raw / 0.35)
        bounded.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad > 0).all())

    def test_gradient_updates_can_leave_initial_bin_for_smaller_amount(self):
        bins = torch.tensor([0.0015, 0.002, 0.0025, 0.00275, 0.003])
        prior = -(bins.log() - torch.tensor(0.0025).log()).abs() / 0.35
        prior = torch.cat((prior.max().view(1) - 1.5, prior))
        winners = []
        for floor in (0.0, 0.75):
            authority = amount_forward_authority(0.05, 0.0, floor)
            raw = torch.tensor([-1., 0., 0., 1., 0., 0.], requires_grad=True)
            optimizer = torch.optim.SGD([raw], lr=1.0)
            for _ in range(300):
                optimizer.zero_grad()
                centered = raw - raw.mean()
                bounded = 0.35 * torch.nn.functional.softsign(centered / 0.35)
                logits = (1-authority)*prior + authority*bounded
                # An observed preference for the smaller positive bin;
                # there is no penalty proportional to operation count here.
                loss = -torch.log_softmax(logits, dim=0)[1]
                loss.backward()
                optimizer.step()
            winners.append(int(logits.argmax()))
        self.assertEqual(winners, [3, 1])

    def test_fine_same_action_has_identical_density_and_score_gradient(self):
        mean = torch.tensor(0.1, requires_grad=True)
        sample = torch.tensor(0.18)
        current = normalized_fine_log_prob(mean, sample, 0.08)
        anchor = normalized_fine_log_prob(mean, sample, 0.08)
        self.assertEqual(float(current), float(anchor))
        gradient = torch.autograd.grad(current - anchor, mean)[0]
        self.assertEqual(float(gradient), 0.0)
        score_gradient = torch.autograd.grad(
            normalized_fine_log_prob(mean, sample, 0.08), mean
        )[0]
        self.assertAlmostEqual(float(score_gradient), 1.0, places=5)

    def test_coverage_floor_prevents_amount_starvation_with_large_where_bonus(self):
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        actuator.args = SimpleNamespace(
            heuristic_guidance_online_simultaneous_operation_exploration=True,
            heuristic_guidance_online_where_coverage_weight=100.0,
            heuristic_guidance_online_factorized_min_share=0.10,
        )
        actuator._den6_pending_matched_groups = OrderedDict()
        actuator._den6_actual_group_stats = {}
        actuator._den6_factorized_group_counts = {}
        actuator._den6_factorized_group_uncertainty = {}
        actuator.train()
        chosen = Counter(actuator._select_den6_factorized_exploration_group({})
                         for _ in range(2200))
        for group in ('Amount', 'Gate', 'Fine', 'WhereAll'):
            self.assertGreaterEqual(chosen[group], 219)


if __name__ == '__main__':
    unittest.main()
