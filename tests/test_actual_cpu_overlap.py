from types import SimpleNamespace
import unittest

import torch

from models.utils.loss.compression import CompressionLossMixin


class _DummyCompression(CompressionLossMixin):
    def __init__(self):
        self.calls = 0

    def _encode_actual_batch(self, args, xyz, final_w=None):
        self.calls += 1
        self.asserted_device = xyz.device.type
        return {
            "bit": float(xyz.sum()),
            "point_count": int(xyz.shape[-1]),
            "encode_time": 1.0,
        }


class ActualCPUOverlapTest(unittest.TestCase):
    def test_cpu_prefetch_runs_the_same_single_guarded_encode(self):
        loss = _DummyCompression()
        args = SimpleNamespace(
            sparsepcgc_device="cpu",
            heuristic_guidance_mode="ana_den6_online",
            _den6_online_training_step_active=True,
            _global_train_step=7,
        )
        xyz = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
        self.assertTrue(loss.start_cpu_actual_encode_prefetch(args, xyz))
        result = loss.consume_cpu_actual_encode_prefetch(args, xyz)
        self.assertEqual(loss.calls, 1)
        self.assertEqual(loss.asserted_device, "cpu")
        self.assertEqual(result["point_count"], 6)
        self.assertEqual(result["bit"], float(xyz.sum()))
        self.assertEqual(
            loss._den6_online_actual_step_guard["edited_encode_count"], 1
        )
        self.assertIsNone(loss.consume_cpu_actual_encode_prefetch(args, xyz))
        loss._cpu_actual_encode_executor.shutdown(wait=True)

    def test_cuda_teacher_is_not_prefetched(self):
        loss = _DummyCompression()
        args = SimpleNamespace(sparsepcgc_device="cuda")
        xyz = torch.zeros((1, 3, 2))
        self.assertFalse(loss.start_cpu_actual_encode_prefetch(args, xyz))
        self.assertEqual(loss.calls, 0)


if __name__ == "__main__":
    unittest.main()
