import unittest

import torch

from models.utils.training.saved_tensor_offload import (
    _OFFLOAD_HOLDERS,
    _PackedOffloadTensor,
    release_autograd_transient_references,
    release_saved_tensor_offload_payloads,
    saved_tensor_offload_stats,
    selective_saved_tensor_cpu_offload,
)


class SelectiveSavedTensorOffloadTest(unittest.TestCase):
    def test_cpu_forward_and_gradient_are_unchanged(self):
        reference = torch.arange(16, dtype=torch.float32, requires_grad=True)
        expected = (reference.square() * 0.25).sum()
        expected.backward()
        expected_gradient = reference.grad.detach().clone()

        actual_input = torch.arange(16, dtype=torch.float32, requires_grad=True)
        with selective_saved_tensor_cpu_offload(0.000001, enabled=True):
            actual = (actual_input.square() * 0.25).sum()
        actual.backward()

        self.assertTrue(torch.equal(actual.detach(), expected.detach()))
        self.assertTrue(torch.equal(actual_input.grad, expected_gradient))

    def test_release_clears_only_graph_backed_last_values(self):
        module = torch.nn.Linear(2, 2)
        graph_value = module(torch.ones(1, 2)).sum()
        module.last_graph_debug = {"value": graph_value}
        module.last_scalar_debug = {"value": 3.0}

        released = release_autograd_transient_references(model=module)

        self.assertIn("Linear.last_graph_debug", released)
        self.assertEqual(module.last_graph_debug, {})
        self.assertEqual(module.last_scalar_debug, {"value": 3.0})

    def test_explicit_payload_release_drops_offload_counter(self):
        from models.utils.training import saved_tensor_offload as module

        cpu_tensor = torch.ones(32, dtype=torch.float32)
        tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
        holder = _PackedOffloadTensor(torch.device("cpu"), cpu_tensor, tensor_bytes)
        with module._OFFLOAD_LOCK:
            module._OFFLOAD_OUTSTANDING_BYTES += tensor_bytes
            module._OFFLOAD_OUTSTANDING_COUNT += 1
        _OFFLOAD_HOLDERS.add(holder)

        released = release_saved_tensor_offload_payloads()
        stats = saved_tensor_offload_stats()

        self.assertGreaterEqual(released["released_count"], 1)
        self.assertIsNone(holder.cpu_tensor)
        self.assertEqual(stats["outstanding_bytes"], 0)
        self.assertEqual(stats["outstanding_count"], 0)


if __name__ == "__main__":
    unittest.main()
