"""Compare a restricted backward on the *same* graph; production stays intact."""
import json
from pathlib import Path
import runpy
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.utils.training import train_runtime as runtime


def selective_capture(policy_loss, model):
    if not torch.is_tensor(policy_loss) or not policy_loss.requires_grad:
        return []
    params = runtime._den6_online_decision_parameters(model)
    if not params:
        return []
    decisions = runtime._den6_retained_decision_tensors(model)
    for tensor in decisions:
        tensor.grad = None
    # Unlike autograd.grad, backward also populates retained decision.grad.
    # This is an experimental restriction, NOT an installed training change.
    torch.autograd.backward(policy_loss, retain_graph=True, inputs=params+decisions)
    base = model.module if hasattr(model, 'module') else model
    actuator = getattr(base, 'actuator', None)
    if actuator is not None:
        actuator._den6_policy_decision_grad_snapshot = [
            (value, None if value.grad is None else value.grad.detach().clone()) for value in decisions]
    snapshot = [(param, None if param.grad is None else param.grad.detach().clone()) for param in params]
    model.zero_grad(set_to_none=True)
    return snapshot


def self_test():
    from torch.utils.checkpoint import checkpoint
    for reentrant in (False, True):
        torch.manual_seed(19)
        model = torch.nn.Module()
        model.actuator = torch.nn.Module()
        model.upstream = torch.nn.Linear(8, 8)
        model.actuator.den6_candidate_actor = torch.nn.Linear(8, 1)
        model.actuator.den6_candidate_critic = torch.nn.Linear(8, 1)
        x = torch.randn(32, 8, requires_grad=True)
        features = checkpoint(model.upstream, x, use_reentrant=reentrant)
        decision = model.actuator.den6_candidate_actor(features)
        decision.retain_grad()
        model.actuator._den6_online_operation_decision_tensors = {'Add': {'where': decision}}
        loss = decision.square().mean()
        expected = runtime._capture_den6_policy_only_gradients(loss, model)
        expected_decision = decision.grad.clone()
        actual = selective_capture(loss, model)
        assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                   for (_,a),(_,b) in zip(expected,actual))
        assert torch.equal(decision.grad, expected_decision)
        print('Exact checkpoint fixture', reentrant)
        def trajectory(capture):
            torch.manual_seed(31)
            network = torch.nn.Module()
            network.actuator = torch.nn.Module()
            network.upstream = torch.nn.Linear(8, 8)
            network.actuator.den6_candidate_actor = torch.nn.Linear(8, 1)
            network.actuator.den6_candidate_critic = torch.nn.Linear(8, 1)
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            records = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                x = torch.randn(32, 8, requires_grad=True)
                features = checkpoint(network.upstream, x, use_reentrant=reentrant)
                decision = network.actuator.den6_candidate_actor(features)
                decision.retain_grad()
                network.actuator._den6_online_operation_decision_tensors = {'Add': {'where':decision}}
                policy = decision.square().mean()
                saved = capture(policy, network)
                (policy+features.square().mean()).backward()
                runtime._restore_den6_policy_only_gradients(saved, network)
                gradients = [None if p.grad is None else p.grad.clone() for p in network.parameters()]
                optimizer.step()
                records.append((gradients, [p.detach().clone() for p in network.parameters()], torch.get_rng_state().clone()))
            return records
        baseline = trajectory(runtime._capture_den6_policy_only_gradients)
        trial = trajectory(selective_capture)
        for original, candidate in zip(baseline, trial):
            for a, b in zip(original[0]+original[1]+[original[2]], candidate[0]+candidate[1]+[candidate[2]]):
                assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
        print('Exact gradients, 3 Adam updates and RNG fixture', reentrant)


def main():
    if sys.argv[1:] == ['--self-test']:
        self_test()
        return
    out = Path(sys.argv[sys.argv.index('--out')+1])
    reference = runtime._capture_den6_policy_only_gradients
    calls = [0]
    def compare(loss, model):
        calls[0] += 1
        results, times, retained = {}, {}, {}
        order = [('original', reference), ('restricted', selective_capture)]
        if calls[0] % 2 == 0:
            order.reverse()
        for name, function in order:
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            results[name] = function(loss, model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times[name] = time.perf_counter()-started
            base = model.module if hasattr(model, 'module') else model
            retained[name] = list(base.actuator._den6_policy_decision_grad_snapshot)
        equal = True
        max_difference = 0.
        for lists in (results, retained):
            for (pa, a), (pb, b) in zip(lists['original'], lists['restricted']):
                assert pa is pb
                same = (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                equal &= same
                if a is not None and b is not None:
                    max_difference = max(max_difference, float((a-b).abs().max().cpu()))
        row = dict(step=calls[0], times=times, parameters=len(results['original']),
                   decisions=len(retained['original']), exact=equal, maximum_absolute_difference=max_difference)
        with (out/'capture_comparison.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        assert equal, row
        # Preserve exactly the reference gradient snapshot for the real update.
        base.actuator._den6_policy_decision_grad_snapshot = retained['original']
        return results['original']
    import train
    train._capture_den6_policy_only_gradients = compare
    runpy.run_path(str(ROOT/'experiments/autonomous_speed_probe.py'), run_name='__main__')


if __name__ == '__main__':
    main()
