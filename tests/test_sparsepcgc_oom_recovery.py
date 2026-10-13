from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from models.utils.loss.actual_encoder import _SparsePCGCActualEncoder
from models.utils.loss.sparsepcgc_teacher_worker import (
    _configure_persistent_cuda_allocator_cache,
)


class SparsePCGCOOMRecoveryTest(unittest.TestCase):
    def test_persistent_cuda_cache_only_changes_cleanup_lifetime(self):
        coordinate_manager = SimpleNamespace(clear_global_coordinate_manager=Mock())
        full_cleanup = Mock()
        encoder = SimpleNamespace(
            _is_cuda=Mock(return_value=True),
            _cleanup_runtime_cache=full_cleanup,
            ME=coordinate_manager,
        )

        enabled = _configure_persistent_cuda_allocator_cache(encoder, 1)
        encoder._cleanup_runtime_cache()

        self.assertTrue(enabled)
        self.assertIs(encoder._full_cleanup_runtime_cache, full_cleanup)
        self.assertEqual(encoder._cuda_cleanup_interval, 1)
        full_cleanup.assert_not_called()
        coordinate_manager.clear_global_coordinate_manager.assert_called_once_with()

    def _encoder(self):
        encoder = _SparsePCGCActualEncoder.__new__(_SparsePCGCActualEncoder)
        encoder.args = SimpleNamespace(
            sparsepcgc_device="auto",
            sparsepcgc_gpu_min_free_mb=4096,
            sparsepcgc_gpu_wait_timeout=10.0,
            sparsepcgc_gpu_wait_interval=0.1,
            sparsepcgc_oom_retry_count=2,
            sparsepcgc_auto_cpu_fallback=False,
            sparsepcgc_host_min_available_mb=0,
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

    def test_cuda_free_mb_uses_runtime_memory_information(self):
        encoder = self._encoder()
        encoder._cuda_worker_enabled = Mock(return_value=True)
        with patch(
            "models.utils.loss.actual_encoder.torch.cuda.mem_get_info",
            return_value=(3 * 1024**3, 48 * 1024**3),
        ) as memory_info:
            free_mb = encoder._cuda_free_mb()

        memory_info.assert_called_once_with(None)
        self.assertEqual(free_mb, 3072.0)

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

    def test_auto_device_restarts_loaded_worker_on_cpu_before_encode(self):
        encoder = self._encoder()
        encoder.args.sparsepcgc_auto_cpu_fallback = True
        encoder._cuda_worker_enabled = Mock(return_value=True)
        encoder._cuda_free_mb = Mock(return_value=2134.7)

        def restart_on_cpu():
            encoder.args.sparsepcgc_device = "cpu"

        encoder._restart_worker_on_cpu = Mock(side_effect=restart_on_cpu)
        result = encoder._wait_for_cuda_capacity("encode")

        encoder._restart_worker_on_cpu.assert_called_once_with()
        self.assertEqual(encoder.args.sparsepcgc_device, "cpu")
        self.assertTrue(result["auto_cpu_fallback"])
        self.assertEqual(result["wait_seconds"], 0.0)

    def test_explicit_cuda_encode_still_waits_instead_of_switching_device(self):
        encoder = self._encoder()
        encoder.args.sparsepcgc_device = "cuda"
        encoder.args.sparsepcgc_auto_cpu_fallback = True
        encoder._cuda_worker_enabled = Mock(return_value=True)
        encoder._cuda_free_mb = Mock(side_effect=[2134.7, 5120.0])
        encoder._restart_worker_on_cpu = Mock()

        with patch("models.utils.loss.actual_encoder.time.sleep", return_value=None):
            result = encoder._wait_for_cuda_capacity("encode")

        encoder._restart_worker_on_cpu.assert_not_called()
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
