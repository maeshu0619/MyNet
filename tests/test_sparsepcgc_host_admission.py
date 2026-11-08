from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import torch

from models.utils.loss.actual_encoder import _SparsePCGCActualEncoder
from models.utils.loss.host_memory_admission import host_recovery_enabled, reclaim_unused_memory
from models.utils.loss.sparsepcgc_teacher_worker import _release_idle_runtime_memory


class Clock:
    def __init__(self):
        self.seconds = 0.

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


class HostAdmissionTest(unittest.TestCase):
    def encoder(self):
        encoder = _SparsePCGCActualEncoder.__new__(_SparsePCGCActualEncoder)
        encoder.args = SimpleNamespace(test_route=True, compress='SparsePCGC',
            heuristic_guidance_mode='ana_den6_online', sparsepcgc_host_pressure_recovery=True,
            sparsepcgc_host_min_available_mb=8192, sparsepcgc_host_wait_timeout=600.,
            sparsepcgc_gpu_wait_interval=2.)
        encoder.writer = Mock()
        encoder._proc = None
        encoder._loaded = False
        return encoder

    def test_normal_capacity_has_no_cleanup_or_delay(self):
        encoder = self.encoder()
        encoder._host_available_mb = Mock(return_value=10000.)
        with patch('models.utils.loss.host_memory_admission.reclaim_unused_memory') as reclaim:
            with patch('models.utils.loss.host_memory_admission.time.sleep') as sleep:
                encoder._wait_for_host_capacity('encode')
        reclaim.assert_not_called()
        sleep.assert_not_called()

    def test_reproduced_5854mb_pressure_can_survive_600s_without_an_encode(self):
        encoder = self.encoder()
        clock = Clock()
        encoder._host_available_mb = lambda: 5854.9 if clock.seconds < 1202. else 9000.
        with patch('models.utils.loss.host_memory_admission.time', clock):
            with patch('models.utils.loss.host_memory_admission.reclaim_unused_memory') as reclaim:
                encoder._wait_for_host_capacity('encode')
        self.assertEqual(clock.seconds, 1202.)
        self.assertEqual(reclaim.call_count, 3)  # initial + 600/1200 seconds
        messages = [c.args[0] for c in encoder.writer.write.call_args_list]
        self.assertEqual(sum('Deferred:' in s for s in messages), 2)
        self.assertTrue(any('required_mb=8192' in s for s in messages))
        self.assertIn('AdmissionReady:', messages[-1])

    def test_same_pending_request_is_encoded_once_only_after_ram_recovers(self):
        encoder = self.encoder()
        clock = Clock()
        encoder.args.sparsepcgc_device = 'cpu'
        encoder._host_available_mb = lambda: 5854.9 if clock.seconds < 602. else 9000.
        encoder._cuda_worker_enabled = Mock(return_value=False)
        encoder._cuda_free_mb = Mock(return_value=float('inf'))
        encoder._send_worker_request = Mock(return_value=({'status': 'ok', 'result': {'bit': 123}}, .1))
        request = dict(input_file='unchanged.ply', output_dir='same-output')
        with patch('models.utils.loss.host_memory_admission.time', clock):
            with patch('models.utils.loss.host_memory_admission.reclaim_unused_memory'):
                response, _, _ = encoder._send_worker_request_with_oom_retry(request, 'same-output')
        self.assertEqual(response['result']['bit'], 123)
        encoder._send_worker_request.assert_called_once_with(request)
        self.assertGreater(clock.seconds, 600.)

    def test_false_and_explicit_strict_keep_the_original_timeout(self):
        for field in ('test_route', 'sparsepcgc_host_pressure_recovery'):
            encoder = self.encoder()
            setattr(encoder.args, field, False)
            encoder.args.sparsepcgc_host_wait_timeout = 0.
            encoder._host_available_mb = Mock(return_value=5854.9)
            self.assertFalse(host_recovery_enabled(encoder.args))
            with self.assertRaisesRegex(RuntimeError, 'host admission timed out'):
                encoder._wait_for_host_capacity('encode')
        encoder = self.encoder()
        encoder.args.compress = 'G-PCC'
        self.assertFalse(host_recovery_enabled(encoder.args))

    def test_worker_start_keeps_12gib_minimum(self):
        encoder = self.encoder()
        clock = Clock()
        encoder._host_available_mb = lambda: 11000. if clock.seconds == 0. else 13000.
        with patch('models.utils.loss.host_memory_admission.time', clock):
            with patch('models.utils.loss.host_memory_admission.reclaim_unused_memory'):
                encoder._wait_for_host_capacity('worker_init')
        self.assertEqual(clock.seconds, 2.)
        self.assertIn('required_mb=12288', encoder.writer.write.call_args_list[0].args[0])

    def test_user_interrupt_is_not_swallowed_or_replaced_with_a_label(self):
        encoder = self.encoder()
        encoder._host_available_mb = Mock(return_value=5854.9)
        with patch('models.utils.loss.host_memory_admission.reclaim_unused_memory'):
            with patch('models.utils.loss.host_memory_admission.time.sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    encoder._wait_for_host_capacity('encode')

    def test_live_training_graph_parameters_and_rng_survive_reclamation(self):
        encoder = self.encoder()
        encoder._host_available_mb = Mock(side_effect=[5854.9, 9000.])
        parameter = torch.nn.Parameter(torch.tensor(2.))
        optimizer = torch.optim.Adam([parameter], lr=.01)
        pending_loss = parameter.square()
        rng = torch.get_rng_state().clone()
        encoder._wait_for_host_capacity('encode')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(float(parameter), 2.)
        self.assertIsNone(parameter.grad)
        self.assertEqual(len(optimizer.state), 0)
        pending_loss.backward()
        self.assertEqual(float(parameter.grad), 4.)
        optimizer.step()
        self.assertLess(float(parameter), 2.)

    def test_cleanup_rpc_retains_model_and_never_prefetches_or_encodes(self):
        encoder = self.encoder()
        encoder._proc = Mock()
        encoder._proc.poll.return_value = None
        encoder._proc.stdin = StringIO()
        encoder._loaded = True
        encoder._request_id = encoder._worker_request_count = 0
        encoder.timeout = 5.
        encoder._read_worker_json = Mock(return_value=dict(status='ok', result=dict(rss_after_mb=6000.)))
        with patch('models.utils.pointcloud.autonomous_proposals.prefetch_autonomous_pool') as prefetch:
            with patch('models.utils.loss.host_memory_admission.gc.collect'):
                with patch('models.utils.loss.host_memory_admission._MALLOC_TRIM'):
                    result = reclaim_unused_memory(encoder)
        self.assertEqual(result['worker_cleanup'], 'ok')
        self.assertIn('release_runtime_cache', encoder._proc.stdin.getvalue())
        self.assertNotIn('input_file', encoder._proc.stdin.getvalue())
        prefetch.assert_not_called()

    def test_worker_cleanup_preserves_codec_parameters(self):
        model = torch.nn.Linear(3, 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        encoder = SimpleNamespace(model=model, _full_cleanup_runtime_cache=Mock(),
                                  _cleanup_runtime_cache=Mock())
        with patch('models.utils.loss.sparsepcgc_teacher_worker._LIBC_MALLOC_TRIM') as trim:
            result = _release_idle_runtime_memory(encoder)
        encoder._full_cleanup_runtime_cache.assert_called_once_with()
        encoder._cleanup_runtime_cache.assert_not_called()
        trim.assert_called_once_with(0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertNotIn('bit', result)


if __name__ == '__main__':
    unittest.main()
