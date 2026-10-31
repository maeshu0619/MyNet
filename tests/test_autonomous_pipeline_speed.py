"""Numerical and lifetime guards for True-route host-side optimizations."""
import copy
import gc
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import threading
import io

import torch

from models.utils.cache.autonomous_pool_ram import AutonomousPoolRAM, batch_pool_collection
from models.utils.cache.lossless_tensor_ram import LosslessObjectRAM
from models.utils.training.autonomous_joint_learning import clip_joint_gradients, _host_scalars


class PoolRestorationTest(unittest.TestCase):
    def test_identical_fresh_objects_and_capacity(self):
        value = dict(pool=[dict(cid=i, xyz=[i, 2, 3], score=i*.125) for i in range(2000)])
        old, new = LosslessObjectRAM(1024**2), AutonomousPoolRAM(1024**2)
        self.assertTrue(old.put('frame', value))
        self.assertTrue(new.put('frame', value))
        self.assertEqual(old.entries, new.entries)
        self.assertEqual(old.get('frame'), new.get('frame'))
        result = new.get('frame')
        result['pool'][0]['xyz'][0] = -99
        self.assertEqual(new.get('frame'), value)
        self.assertIsNone(new.get('other'))
        self.assertEqual(new.stats()['misses'], 1)
        tiny = AutonomousPoolRAM(1)
        self.assertFalse(tiny.put('frame', value))
        self.assertEqual(tiny.bytes, 0)

    def test_gc_policy_restored_for_success_and_failure(self):
        originally_enabled = gc.isenabled()
        thresholds = gc.get_threshold()
        try:
            for enabled in (True, False):
                (gc.enable if enabled else gc.disable)()
                for fail in (False, True):
                    with patch('models.utils.cache.autonomous_pool_ram.gc.collect', wraps=gc.collect) as collect:
                        try:
                            with batch_pool_collection():
                                self.assertFalse(gc.isenabled())
                                if fail:
                                    raise ValueError('decode failed')
                        except ValueError:
                            pass
                        self.assertEqual(gc.isenabled(), enabled)
                        self.assertEqual(collect.call_count, int(enabled))
            gc.enable()
            with patch('models.utils.cache.autonomous_pool_ram.gc.collect', side_effect=RuntimeError('collection failed')):
                with self.assertRaises(RuntimeError):
                    with batch_pool_collection():
                        pass
            self.assertTrue(gc.isenabled())
            self.assertEqual(gc.get_threshold(), thresholds)
        finally:
            (gc.enable if originally_enabled else gc.disable)()

    def test_prefetch_verifies_full_key_and_retains_fresh_objects(self):
        cache = AutonomousPoolRAM(1024**2)
        try:
            for key in ('a', 'b', 'c'):
                cache.put(key, dict(key=key, coords=[[1, 2, 3]]))
            # First pass learns a -> b -> c -> a; no guessing from filenames.
            for key in ('a', 'b', 'c', 'a'):
                self.assertEqual(cache.get(key)['key'], key)
            state = torch.get_rng_state().clone()
            cache.prefetch_next()
            cache._pending[1].result(timeout=5)
            result = cache.get('b')
            self.assertEqual(result['key'], 'b')
            self.assertEqual(cache.prefetch_hits, 1)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            result['coords'][0][0] = -1
            self.assertEqual(cache.get('b')['coords'], [[1, 2, 3]])
            cache.get('a')
            cache.prefetch_next()  # predicts b; actual order changes to c
            self.assertEqual(cache.get('c')['key'], 'c')
            self.assertEqual(cache.prefetch_hits, 1)
        finally:
            cache.close()
        self.assertIsNone(cache._pending)
        self.assertIsNone(cache._executor)

    def test_rejected_frames_do_not_grow_successor_storage_and_clear_joins(self):
        cache = AutonomousPoolRAM(1024**2, max_entries=2)
        try:
            for key in ('a', 'b', 'a'):
                if cache.get(key) is None:
                    cache.put(key, dict(key=key))
            cache.prefetch_next()
            for i in range(100):
                key = 'rejected-%d' % i
                self.assertIsNone(cache.get(key))
                self.assertFalse(cache.put(key, dict(key=key)))
            self.assertLessEqual(len(cache._successors), 2)
            cache.clear()
            self.assertEqual(cache.bytes, 0)
            self.assertEqual(cache.entries, {})
            self.assertEqual(cache._successors, {})
            self.assertIsNone(cache._pending)
            self.assertIsNone(cache._executor)
        finally:
            cache.close()

    def test_one_background_restore_at_a_time_and_cache_budget(self):
        cache = AutonomousPoolRAM(1024**2)
        entered, release = threading.Event(), threading.Event()
        restore = cache._restore
        def blocked(payload):
            entered.set()
            release.wait(timeout=5)
            return restore(payload)
        try:
            for key in ('a', 'b', 'c'):
                cache.put(key, dict(key=key))
            for key in ('a', 'b', 'a'):
                cache.get(key)
            admitted = cache.bytes
            with patch.object(cache, '_restore', side_effect=blocked):
                cache.prefetch_next()
                self.assertTrue(entered.wait(timeout=5))
                for _ in range(10):
                    cache.prefetch_next()
                self.assertEqual(cache.prefetch_submitted, 1)
                self.assertEqual(cache.bytes, admitted)
                self.assertEqual(cache.get('c'), dict(key='c'))
                cache.prefetch_next()
                self.assertEqual(cache.prefetch_submitted, 1)
                release.set()
        finally:
            release.set()
            cache.close()


class ClipReportingTest(unittest.TestCase):
    def test_scalar_transfer_preserves_order_dtype_and_empty(self):
        values = [torch.tensor(.1, dtype=torch.float32), torch.tensor(1e-30, dtype=torch.float64)]
        self.assertEqual(_host_scalars(values), [float(v) for v in values])
        self.assertEqual(_host_scalars([]), [])
        if torch.cuda.is_available():
            values = [values[0].cuda(), values[1], values[0].cuda()]
            self.assertEqual(_host_scalars(values), [float(v) for v in values])

    def test_clipped_tensors_and_diagnostics_match_sequential_reference(self):
        model = torch.nn.Module()
        model.args = SimpleNamespace(repair_online_decision_grad_max_norm=1.)
        model.actuator = torch.nn.Module()
        names = dict(where='drop_head', actor_proposal='den6_candidate_actor',
                     amount_fine='add_amount_head', amount_coarse='algorithmic_amount_selector_head',
                     gate='operation_gate_head', critic='den6_plan_critic')
        generator = torch.Generator().manual_seed(73)
        for name in list(names.values()) + ['other']:
            layer = torch.nn.Linear(17, 9)
            setattr(model.actuator, name, layer)
            for p in layer.parameters():
                p.grad = torch.randn(p.shape, generator=generator) * (.001 if name == 'add_amount_head' else 3.)
        old = copy.deepcopy(model)
        # deepcopy(Module) omits gradients.
        for a, b in zip(model.parameters(), old.parameters()):
            b.grad = a.grad.clone()
        expected_total = math.sqrt(sum(float(p.grad.double().square().sum()) for p in old.parameters()))
        expected = {}
        for role, name in list(names.items()) + [('backbone', 'other')]:
            params = list(getattr(old.actuator, name).parameters())
            bound = 10. if role == 'backbone' else 1.
            before = float(torch.nn.utils.clip_grad_norm_(params, bound))
            after = math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
            expected[role] = dict(before=before, after=after, bound=bound)
        report = clip_joint_gradients(model, 10.)
        self.assertEqual(report['train_grad_total_norm_before_clip'], expected_total)
        self.assertEqual(report['autonomous_gradient_groups'], expected)
        for a, b in zip(model.parameters(), old.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))


class CodecWaitTest(unittest.TestCase):
    def encoder(self, enabled):
        from models.utils.loss.actual_encoder import _SparsePCGCActualEncoder
        encoder = _SparsePCGCActualEncoder(SimpleNamespace(
            test_route=enabled, compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online'))
        encoder._proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        return encoder

    def test_true_wait_joins_on_success_and_codec_failure(self):
        for fail in (False, True):
            encoder = self.encoder(True)
            future = unittest.mock.Mock()
            with patch('models.utils.training.sparsepcgc_autonomous_policy.autonomous_search_enabled', return_value=True), \
                 patch('models.utils.pointcloud.autonomous_proposals.prefetch_autonomous_pool', return_value=future) as start, \
                 patch.object(encoder, '_read_worker_json', side_effect=RuntimeError('codec timeout') if fail else None,
                              return_value=dict(status='ok', request_id=1)):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'codec timeout'):
                        encoder._send_worker_request_once(dict(command='encode'))
                else:
                    response, elapsed = encoder._send_worker_request_once(dict(command='encode'))
                    self.assertEqual(response['status'], 'ok')
                    self.assertGreaterEqual(elapsed, 0.)
                start.assert_called_once()
                future.result.assert_called_once_with()
                self.assertEqual(encoder._worker_request_count, 1)

    def test_false_does_not_prefetch_and_preserves_request(self):
        encoder = self.encoder(False)
        with patch('models.utils.pointcloud.autonomous_proposals.prefetch_autonomous_pool') as start, \
             patch.object(encoder, '_read_worker_json', return_value=dict(status='ok', request_id=1)):
            encoder._send_worker_request_once(dict(command='encode', input_file='original.ply'))
            start.assert_not_called()
        import json
        self.assertEqual(json.loads(encoder._proc.stdin.getvalue()),
                         dict(command='encode', input_file='original.ply', request_id=1))


if __name__ == '__main__':
    unittest.main()
