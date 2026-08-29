from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from models.utils.loss.actual_encoder import _SparsePCGCActualEncoder


class SparsePCGCOOMRecoveryTest(unittest.TestCase):
    def _encoder(self):
        encoder = _SparsePCGCActualEncoder.__new__(_SparsePCGCActualEncoder)
        encoder.args = SimpleNamespace(
            sparsepcgc_device="auto",
            sparsepcgc_gpu_min_free_mb=4096,
            sparsepcgc_gpu_wait_timeout=10.0,
            sparsepcgc_gpu_wait_interval=0.1,
            sparsepcgc_oom_retry_count=2,
            sparsepcgc_auto_cpu_fallback=False,
        )
        encoder.writer = Mock()
        encoder._request_id = 0
        encoder._gpu_admission_wait_count = 0
        encoder._gpu_admission_wait_seconds = 0.0
        encoder._cuda_oom_retry_count = 0
        return encoder

    def test_gpu_admission_waits_until_global_free_memory_is_sufficient(self):
        encoder = self._encoder()
        encoder._cuda_worker_enabled = Mock(return_value=True)
        encoder._cuda_free_mb = Mock(side_effect=[128.0, 5120.0])

        with patch("models.utils.loss.actual_encoder.time.sleep", return_value=None):
            result = encoder._wait_for_cuda_capacity("test")

        self.assertEqual(result["free_before_mb"], 128.0)
        self.assertEqual(result["free_after_mb"], 5120.0)
        self.assertEqual(encoder._gpu_admission_wait_count, 1)
        self.assertGreaterEqual(encoder.writer.write.call_count, 2)

    def test_auto_device_falls_back_to_cpu_before_worker_launch(self):
        encoder = self._encoder()
        encoder.args.sparsepcgc_auto_cpu_fallback = True
        encoder._cuda_worker_enabled = Mock(return_value=True)
        encoder._cuda_free_mb = Mock(return_value=704.7)

        result = encoder._wait_for_cuda_capacity("worker_init")

        self.assertEqual(encoder.args.sparsepcgc_device, "cpu")
        self.assertTrue(result["auto_cpu_fallback"])
        self.assertEqual(result["wait_seconds"], 0.0)
        encoder.writer.write.assert_called_once()

    def test_explicit_cuda_does_not_silently_fall_back_to_cpu(self):
        encoder = self._encoder()
        encoder.args.sparsepcgc_device = "cuda"
        encoder.args.sparsepcgc_auto_cpu_fallback = True
        encoder._cuda_worker_enabled = Mock(return_value=True)
        encoder._cuda_free_mb = Mock(side_effect=[704.7, 5120.0])

        with patch("models.utils.loss.actual_encoder.time.sleep", return_value=None):
            result = encoder._wait_for_cuda_capacity("worker_init")

        self.assertEqual(encoder.args.sparsepcgc_device, "cuda")
        self.assertEqual(result["free_after_mb"], 5120.0)

    def test_cuda_oom_retries_the_identical_request_without_proxy_fallback(self):
        encoder = self._encoder()
        encoder._wait_for_cuda_capacity = Mock(return_value={
            "wait_seconds": 0.0,
            "free_before_mb": 8192.0,
            "free_after_mb": 8192.0,
        })
        encoder._send_worker_request = Mock(side_effect=[
            ({
                "status": "error",
                "error_type": "cuda_oom",
                "message": "CUDA out of memory",
            }, 0.25),
            ({"status": "ok", "result": {"file_size": 100.0}}, 0.50),
        ])

        with tempfile.TemporaryDirectory() as directory:
            output_dir = os.path.join(directory, "encoded")
            os.makedirs(output_dir)
            response, roundtrip, audit = encoder._send_worker_request_with_oom_retry(
                {"input_file": "same.ply", "output_dir": output_dir},
                output_dir,
            )

        self.assertEqual(response["status"], "ok")
        self.assertEqual(encoder._send_worker_request.call_count, 2)
        first_request = encoder._send_worker_request.call_args_list[0].args[0]
        second_request = encoder._send_worker_request.call_args_list[1].args[0]
        self.assertEqual(first_request, second_request)
        self.assertAlmostEqual(roundtrip, 0.75)
        self.assertEqual(audit["oom_retries"], 1)
        self.assertEqual(encoder._cuda_oom_retry_count, 1)


if __name__ == "__main__":
    unittest.main()
