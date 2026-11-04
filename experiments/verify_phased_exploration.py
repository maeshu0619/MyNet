"""CPU sampling diagnostic, NOT training performance or Actual compression.

Compares the broad stage against an immutable source snapshot, then measures
the neighbourhood on an equal-logit fixture. No codec/model is launched and
the fixture's reference cost must never be reported as measured performance.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=400)
    options = parser.parse_args()
    if options.out.exists():
        raise FileExistsError(options.out)
    from experiments.verify_test_route import load
    from tests.test_autonomous_local_search import LocalSearchTest
    from models.utils.training.autonomous_joint_learning import sample_joint_amounts, sample_where_order
    from models.utils.training.autonomous_local_search import (
        prepare_neighbourhood, neighbourhood_controls, where_neighbourhood)
    old = load(options.before, 'models/utils/training/autonomous_joint_learning.py',
               'models.utils.training._broad_before')
    fixture = LocalSearchTest()
    bins = torch.tensor([0., .001, .002, .003, .01, .03])
    means = {op: torch.tensor(0.) for op in ('Add', 'Prune', 'Adjust')}
    identical = []
    for step in (0, 3900, 5999, 6000):
        args, exact, memory = fixture.fixture(step)
        guide = prepare_neighbourhood(args, exact, memory, True)
        for seed in range(24):
            torch.manual_seed(seed)
            expected = old.sample_joint_amounts(torch.zeros(6), means, torch.zeros(3),
                bins, .03, 1., True, uniform_fraction=.25)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            actual = sample_joint_amounts(torch.zeros(6), means, torch.zeros(3),
                bins, .03, 1., True, uniform_fraction=.25, neighbourhood=guide)
            assert torch.equal(expected_rng, torch.get_rng_state())
            for op in means:
                for field in ('ratios', 'indices', 'samples', 'log_probs', 'logits'):
                    assert torch.equal(expected[field][op], actual[field][op]), (step, seed, op, field)
            identical.append(dict(step=step, seed=seed, outputs_and_rng_equal=True))
    rows = []
    # Compare the actual loss implementation and runtime keys on identical
    # analytic test inputs. These inputs are NOT claimed to be codec results.
    from tests.test_autonomous_joint_learning import JointLearningTest
    from models.utils.training.autonomous_joint_learning import joint_actual_policy_loss
    torch.manual_seed(17)
    old_model, old_lp, _ = JointLearningTest().model()
    torch.manual_seed(17)
    new_model, new_lp, _ = JointLearningTest().model()
    for c, g in ((2., .02), (-5., .01), (-4., .008)):
        a = old.joint_actual_policy_loss(old_model, torch.tensor(c), torch.tensor(g))
        b = joint_actual_policy_loss(new_model, torch.tensor(c), torch.tensor(g))
        assert torch.equal(a, b)
        # The tiny fixture reuses its entropy graph across the three inputs.
        a.backward(retain_graph=True); b.backward(retain_graph=True)
        assert torch.equal(old_lp.grad, new_lp.grad)
        assert old_model._den6_online_efficiency_baseline == new_model._den6_online_efficiency_baseline
    for episode in (1, 60, 61, 70, 80, 100, 120, 150, 180, 250):
        args, exact, memory = fixture.fixture((episode - 1)*100)
        guide = prepare_neighbourhood(args, exact, memory, True)
        controls = neighbourhood_controls(fixture.controls(), guide)
        fine_means = {op: torch.tensor(0.) for op in means}
        scores = where_neighbourhood(torch.zeros(32), 'Add', guide, controls['where_temperature'])
        overlap, changed, near, noop, amount_count = 0, 0, 0, 0, 0
        torch.manual_seed(20261104)
        for _ in range(options.draws):
            action = sample_joint_amounts(torch.zeros(6), fine_means, torch.zeros(3),
                bins, .03, 1., True, uniform_fraction=controls['uniform_fraction'],
                temperature_override=controls['amount_temperature'],
                fine_std_override=controls['fine_std'], neighbourhood=guide)
            for op in means:
                index = int(action['indices'][op])
                near += index in (1, 2, 3)  # reference .002, neighbouring .001/.003
                noop += index == 0
                amount_count += 1
            order = sample_where_order(scores, 1., True, controls['where_temperature'])
            count = sum(int(i) < 4 for i in order[:4])
            overlap += count
            changed += count < 4
        rows.append(dict(episode=episode, stage=guide['stage'],
            controls={k: float(v.detach()) for k, v in controls.items()},
            amount_neighbouring_bin_fraction=near/amount_count,
            noop_fraction=noop/amount_count,
            where_reference_overlap=overlap/(4*options.draws),
            where_changed_set_fraction=changed/options.draws))
    result = dict(measurement_kind='synthetic equal-logit sampling diagnostic; NOT Actual codec evaluation',
                  reference='test fixture only, not a measured point cloud',
                  before=str(options.before), broad_snapshot_parity=identical,
                  analytic_loss_backward_and_runtime_key_compatibility=True,
                  draws_per_stage=options.draws, stages=rows)
    with options.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print('PASS: %d snapshot/RNG comparisons; %d non-Actual sampling stages' %
          (len(identical), len(rows)))
    for row in rows:
        print('episode=%d native_std=%.6f uniform=%.6f near_amount=%.3f where_overlap=%.3f changed_sets=%.3f' %
              (row['episode'], row['controls']['native_std'], row['controls']['uniform_fraction'],
               row['amount_neighbouring_bin_fraction'], row['where_reference_overlap'],
               row['where_changed_set_fraction']))


if __name__ == '__main__':
    main()
