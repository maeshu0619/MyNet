import unittest

import torch

from models.utils.training.saved_tensor_offload import (
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


if __name__ == "__main__":
    unittest.main()
