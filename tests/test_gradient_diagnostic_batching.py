import unittest
import torch

from models.utils.training.train_runtime import (
    _den6_online_grad_norms,
    _capture_den6_policy_only_gradients,
    _restore_den6_policy_only_gradients,
    _probe_den6_policy_source_gradients,
    _summarize_nonfinite_grads,
)


class GradientDiagnosticBatchingTest(unittest.TestCase):
    def test_retained_decision_audit_excludes_source_probe_and_joint_backward(self):
        model = torch.nn.Module()
        model.actuator = torch.nn.Module()
        model.actuator.den6_candidate_actor = torch.nn.Linear(2, 1)
        decision = model.actuator.den6_candidate_actor(torch.ones(1, 2))
        decision.retain_grad()
        model.actuator._den6_online_operation_decision_tensors = {
            'Add': {'where': decision}
        }
        policy = (3 * decision).sum()
        _probe_den6_policy_source_gradients(model, {'a': policy, 'b': 2 * policy})
        self.assertEqual(float(decision.grad), 9.)
        snapshot = _capture_den6_policy_only_gradients(policy, model)
        (policy + 100 * decision.sum()).backward()
        _restore_den6_policy_only_gradients(snapshot, model)
        self.assertEqual(float(decision.grad), 3.)
        for parameter, expected in snapshot:
            torch.testing.assert_close(parameter.grad, expected)
        self.assertEqual(model.actuator._den6_policy_decision_grad_snapshot, [])

    def test_nonfinite_counts_names_and_limit_preserve_parameter_order(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)
        )
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        model[0].weight.grad[0, 0] = float('nan')
        model[0].weight.grad[0, 1] = float('inf')
        model[0].bias.grad[0] = -float('inf')
        model[1].bias.grad = None
        gradients = [
            None if p.grad is None else p.grad.clone()
            for p in model.parameters()
        ]
        result = _summarize_nonfinite_grads(model, limit=1)
        self.assertEqual(result, {
            'has_nonfinite': True,
            'bad_element_count': 3,
            'checked_param_count': 3,
            'checked_element_count': 10,
            'bad_names': ['0.weight:2'],
        })
        for parameter, gradient in zip(model.parameters(), gradients):
            if gradient is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, gradient, equal_nan=True)

    def test_overlapping_group_norms_and_decision_norms_are_unchanged(self):
        model = torch.nn.Module()
        model.actuator = torch.nn.Module()
        actuator = model.actuator
        for name in (
            'drop_head', 'add_head', 'add_voxel_head', 'move_voxel_head',
            'subtree_move_source_head', 'drop_amount_head', 'add_amount_head',
            'move_amount_head', 'den6_candidate_actor', 'operation_gate_head',
            'algorithmic_amount_selector_head',
        ):
            setattr(actuator, name, torch.nn.Linear(3, 2))
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, float(index + 1))
        decisions = {}
        for operation in ('Add', 'Prune', 'Adjust'):
            decisions[operation] = {}
            for decision in ('where', 'amount'):
                value = torch.tensor([1., 2.], requires_grad=True)
                value.grad = torch.tensor([3., 4.])
                decisions[operation][decision] = value
        actuator._den6_online_operation_decision_tensors = decisions

        def old_norm(modules):
            terms = [
                torch.sum(torch.nan_to_num(p.grad.detach().float()) ** 2)
                for module in modules for p in module.parameters()
                if p.grad is not None
            ]
            return float(torch.sqrt(torch.stack(terms).sum()))

        where = old_norm((
            actuator.drop_head, actuator.add_head, actuator.add_voxel_head,
            actuator.move_voxel_head, actuator.den6_candidate_actor,
        ))
        amount = old_norm((
            actuator.drop_amount_head, actuator.add_amount_head,
            actuator.move_amount_head, actuator.algorithmic_amount_selector_head,
        ))
        result = _den6_online_grad_norms(model)
        self.assertEqual(result['den6_online_where_grad_norm'], where)
        self.assertEqual(result['den6_online_candidate_where_grad_norm'], where)
        self.assertEqual(result['den6_online_amount_grad_norm'], amount)
        for operation in ('add', 'prune', 'adjust'):
            for decision in ('where', 'amount'):
                self.assertEqual(result[
                    f'den6_online_{operation}_{decision}_decision_grad_norm'
                ], 5.0)
        self.assertEqual(actuator._den6_online_operation_decision_tensors, {})


if __name__ == '__main__':
    unittest.main()
