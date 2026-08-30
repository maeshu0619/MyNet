from types import SimpleNamespace
import unittest

import torch
from torch import nn

from models.modules.structure_actuator import StructureRepairActuator


def _runner(*, chunk_size, checkpoint_enabled):
    actuator = StructureRepairActuator.__new__(StructureRepairActuator)
    nn.Module.__init__(actuator)
    actuator.args = SimpleNamespace(
        heuristic_guidance_mode="ana_den6_online",
        full_cloud_activation_checkpoint=checkpoint_enabled,
        full_cloud_head_chunk_size=chunk_size,
    )
    actuator.train()
    return actuator


def _head():
    return nn.Sequential(
        nn.Conv1d(7, 13, 1),
        nn.ReLU(inplace=True),
        nn.Conv1d(13, 11, 1),
        nn.ReLU(inplace=True),
        nn.Conv1d(11, 5, 1),
    )


class StructureActuatorChunkingTest(unittest.TestCase):
    def test_large_head_chunking_preserves_forward_and_backward(self):
        torch.manual_seed(7123)
        full_runner = _runner(chunk_size=0, checkpoint_enabled=True)
        chunk_runner = _runner(chunk_size=17, checkpoint_enabled=True)
        full_head = _head()
        chunk_head = _head()
        chunk_head.load_state_dict(full_head.state_dict())

        full_input = torch.randn(2, 7, 73, requires_grad=True)
        chunk_input = full_input.detach().clone().requires_grad_(True)
        weights = torch.randn(2, 5, 73)

        full_output = full_runner._run_large_head(full_head, full_input)
        chunk_output = chunk_runner._run_large_head(chunk_head, chunk_input)
        torch.testing.assert_close(chunk_output, full_output, rtol=0.0, atol=0.0)

        (full_output * weights).sum().backward()
        (chunk_output * weights).sum().backward()
        torch.testing.assert_close(chunk_input.grad, full_input.grad, rtol=1e-6, atol=1e-6)
        for full_parameter, chunk_parameter in zip(full_head.parameters(), chunk_head.parameters()):
            torch.testing.assert_close(
                chunk_parameter.grad,
                full_parameter.grad,
                rtol=2e-6,
                atol=2e-6,
            )

    def test_large_head_chunking_bounds_points_per_head_call(self):
        class RecordingHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Conv1d(7, 5, 1)
                self.seen_point_counts = []

            def forward(self, value):
                self.seen_point_counts.append(int(value.shape[-1]))
                return self.projection(value)

        runner = _runner(chunk_size=17, checkpoint_enabled=False)
        head = RecordingHead()
        value = torch.randn(2, 7, 73, requires_grad=True)
        output = runner._run_large_head(head, value)

        self.assertEqual(output.shape, (2, 5, 73))
        self.assertEqual(head.seen_point_counts, [17, 17, 17, 17, 5])
        output.square().mean().backward()
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(head.projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
