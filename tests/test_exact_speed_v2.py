from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


class LosslessRAMTest(unittest.TestCase):
    def test_all_dtype_bits_and_copy_isolation(self):
        from models.utils.cache.lossless_tensor_ram import LosslessTensorRAM
        cache = LosslessTensorRAM(1024**2, 128)
        source = {str(dtype): torch.tensor([0., -0., 1., float('nan')]).to(dtype)
                  for dtype in (torch.float32, torch.float16, torch.bfloat16)}
        source.update(index=torch.arange(40, dtype=torch.int32), scalar=torch.tensor(3.), empty=torch.empty(0))
        self.assertTrue(cache.put('x', source))
        for name, value in cache.get('x').items():
            self.assertEqual(value.dtype, source[name].dtype)
            self.assertEqual(value.shape, source[name].shape)
            self.assertTrue(torch.equal(value.reshape(-1).view(torch.uint8), source[name].reshape(-1).view(torch.uint8)))
        cache.get('x')['index'].zero_()
        self.assertTrue(torch.equal(cache.get('x')['index'], source['index']))
        with self.assertRaises(ValueError):
            cache.put('trainable', {'x': torch.ones(2, requires_grad=True)})

    def test_bounded_admission_without_cyclic_thrash(self):
        from models.utils.cache.lossless_tensor_ram import LosslessTensorRAM
        cache = LosslessTensorRAM(50, 2)
        for i in range(100):
            cache.put(i, {'x': torch.zeros(100, dtype=torch.int32)})
        self.assertLessEqual(cache.bytes, 50)
        self.assertEqual(len(cache.entries), 2)
        self.assertIsNotNone(cache.get(0))
        self.assertIsNone(cache.get(99))
        cache.clear()
        self.assertEqual(cache.bytes, 0)

    def test_disk_hits_promote_only_frozen_true_and_invalidate_weights(self):
        from models.network import Network
        def owner(route):
            model = Network.__new__(Network)
            torch.nn.Module.__init__(model)
            model.args = SimpleNamespace(test_route=route, compress='SparsePCGC',
                heuristic_guidance_mode='ana_den6_online', encoder_0grad=True)
            model.encoder = torch.nn.Linear(3, 2)
            model.point_transformer_feature_cache = OrderedDict()
            model._point_transformer_feature_cache_hits = model._point_transformer_feature_cache_misses = 0
            model._point_transformer_encoder_state_sha256 = None
            return model
        entry = dict(coarse_fused=torch.randn(1, 2, 8), full_to_coarse_idx=torch.arange(8, dtype=torch.int32), coarse_count=8)
        model = owner(True)
        with mock.patch.object(model, '_load_point_transformer_feature_disk_cache', return_value=entry) as disk:
            first = model._get_point_transformer_feature_cache('frame', 'cpu')
            second = model._get_point_transformer_feature_cache('frame', 'cpu')
            self.assertEqual(disk.call_count, 1)
            self.assertTrue(torch.equal(first['coarse_fused'], second['coarse_fused']))
            adapter = torch.nn.Conv1d(2, 1, 1)
            adapter(second['coarse_fused']).sum().backward()
            self.assertGreater(float(adapter.weight.grad.norm()), 0.)
            with torch.no_grad():
                model.encoder.weight.add_(1.)
            model._get_point_transformer_feature_cache('frame', 'cpu')
            self.assertEqual(disk.call_count, 2)
            model.clear_point_transformer_feature_cache()
            self.assertEqual(model._autonomous_frozen_feature_ram.bytes, 0)
        model = owner(False)
        with mock.patch.object(model, '_load_point_transformer_feature_disk_cache', return_value=entry) as disk:
            model._get_point_transformer_feature_cache('frame', 'cpu')
            model._get_point_transformer_feature_cache('frame', 'cpu')
            self.assertEqual(disk.call_count, 2)
            self.assertIsNone(getattr(model, '_autonomous_frozen_feature_ram', None))


class ExactLatticeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA parity')
    def test_integer_full_cloud_distances_indices_gradients_and_far_fallback(self):
        from models.utils.loss.autonomous_exact_geometry import nearest_squared
        from models.utils.loss.utils_loss import chamfer_dist
        torch.manual_seed(19)
        r = torch.randint(-3, 4, (1, 600, 3), device='cuda').float().requires_grad_()
        q = torch.cat((r.detach()[:, :50], torch.tensor([[[100., 10., 20.], [-40., 0., 0.]]], device='cuda')), 1).requires_grad_()
        expected, _, index, _ = chamfer_dist(q, r)
        stats = []
        actual, actual_index = nearest_squared(q, r, chamfer_dist, stats)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(index.long(), actual_index))
        self.assertTrue(stats[-1]['integer_lattice'])
        self.assertEqual(stats[-1]['kdtree_queries'], 2)
        weight = torch.rand_like(expected)
        old = torch.autograd.grad((expected*weight).sum(), (q, r), retain_graph=True)
        new = torch.autograd.grad((actual*weight).sum(), (q, r))
        for a, b in zip(old, new):
            self.assertTrue(torch.allclose(a, b, atol=2e-6, rtol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA duplicate tie')
    def test_duplicate_ties_and_noop_have_exact_first_index(self):
        from models.utils.loss.autonomous_exact_geometry import nearest_squared
        from models.utils.loss.utils_loss import chamfer_dist
        r = torch.zeros(1, 300, 3, device='cuda', requires_grad=True)
        q = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]], device='cuda', requires_grad=True)
        expected, _, index, _ = chamfer_dist(q, r)
        actual, actual_index = nearest_squared(q, r, chamfer_dist)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(index.long(), actual_index))
        old = torch.autograd.grad(expected.sum(), (q, r), retain_graph=True)
        new = torch.autograd.grad(actual.sum(), (q, r))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, new)))


class StaticProxyTest(unittest.TestCase):
    def test_verified_cache_reuses_values_and_refreshes_rd_weight(self):
        from models.modules import heuristic_guidance as module
        args = SimpleNamespace(test_route=True, compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online')
        exact = dict(autonomous_search=True, operation_edit_units={name: [
            dict(optimistic_gain_bits=float(i), geometry_cost=float(3-i)) for i in range(3)]
            for name in ('Add', 'Prune', 'Adjust')})
        def empty():
            return dict(candidate_tensor_map={name: dict(rank_score=torch.zeros(3))
                                               for name in ('Add', 'Prune', 'Adjust')})
        like = torch.zeros(1, 1, 5)
        cached = module._attach_candidate_local_proxy(empty(), exact, like, args)
        args.rd_geometry_weight_multiplier = 2.
        expected = module._attach_candidate_local_proxy(empty(), exact, like, args)
        with mock.patch.object(module, '_candidate_local_proxy_tensors', side_effect=AssertionError('unexpected recompute')):
            actual = module._attach_candidate_local_proxy(cached, exact, like, args, reuse_verified_cache=True)
        for op, mapping in actual['candidate_tensor_map'].items():
            for name in ('local_rate_benefit', 'local_geometry_risk', 'local_utility_target'):
                self.assertTrue(torch.equal(mapping[name], expected['candidate_tensor_map'][op][name]))


class StaticPoolCacheTest(unittest.TestCase):
    def test_exact_pool_copy_rng_and_content_order_phase_invalidation(self):
        from models.utils.pointcloud import autonomous_proposals as module
        coords = torch.cartesian_prod(torch.arange(2, 14), torch.arange(2, 14), torch.arange(2, 14))
        coords = coords[coords.sum(1)%3 == 0].T.unsqueeze(0)
        args = SimpleNamespace(test_route=True, compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online',
            _global_train_step=0, _fixed_episode_steps=100, sparsepcgc_native_resolution=31)
        identity = dict(input_sha256='3'*64, setting_id='unit')
        def stable(payload):
            return {k: v for k, v in payload.items() if k not in (
                'elapsed_sec', 'proposal_timing', 'autonomous_pool_cache_hit', 'autonomous_pool_cache_stats')}
        with mock.patch.object(module, '_POOL_RAM', None), mock.patch.object(module, '_POOL_RAM_CONTEXT', None):
            rng = torch.get_rng_state().clone()
            ctx = dict(global_voxel_coords=coords)
            expected = module._build_autonomous_pool(ctx, args, identity)
            first = module.build_autonomous_pool(ctx, args, identity)
            second = module.build_autonomous_pool(ctx, args, identity)
            self.assertFalse(first['autonomous_pool_cache_hit'])
            self.assertTrue(second['autonomous_pool_cache_hit'])
            self.assertEqual(stable(first), stable(expected))
            self.assertEqual(stable(second), stable(expected))
            second['operation_edit_units']['Add'][0]['candidate_id'] = 'modified return'
            self.assertEqual(stable(module.build_autonomous_pool(ctx, args, identity)), stable(expected))
            reordered = dict(global_voxel_coords=coords.flip(2))
            answer = module.build_autonomous_pool(reordered, args, identity)
            self.assertFalse(answer['autonomous_pool_cache_hit'])
            self.assertEqual(stable(answer), stable(module._build_autonomous_pool(reordered, args, identity)))
            args._global_train_step = 200
            answer = module.build_autonomous_pool(ctx, args, identity)
            self.assertFalse(answer['autonomous_pool_cache_hit'])
            self.assertEqual(stable(answer), stable(module._build_autonomous_pool(ctx, args, identity)))
            args._autonomous_policy_training = False
            answer = module.build_autonomous_pool(ctx, args, identity)
            self.assertEqual(stable(answer), stable(module._build_autonomous_pool(ctx, args, identity)))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))


if __name__ == '__main__':
    unittest.main()
