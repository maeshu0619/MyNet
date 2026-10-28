from types import SimpleNamespace
import unittest

import torch


class SpeedParityTest(unittest.TestCase):
    def test_configurable_bins_and_checkpoint_shape_are_explicit(self):
        from models.modules.structure_actuator import StructureRepairActuator
        from models.utils.training.sparsepcgc_autonomous_policy import configure_autonomous_search
        def make(bins=''):
            args = SimpleNamespace(test_route=True, compress='SparsePCGC',
                heuristic_guidance_mode='ana_den6_online', test_route_amount_bins=bins)
            configure_autonomous_search(args)
            return StructureRepairActuator(2, hidden_dim=4, args=args)
        expanded = make()
        self.assertEqual(expanded.algorithmic_amount_selector_head[-1].out_channels, 16)
        legacy = make('0.001,0.009,0.01,0.015,0.03')
        self.assertEqual(legacy.algorithmic_amount_selector_head[-1].out_channels, 6)
        self.assertEqual(make('0.001,0.009,0.01,0.015,0.03').load_state_dict(legacy.state_dict()).missing_keys, [])
        with self.assertRaisesRegex(RuntimeError, 'test_route Amount'):
            expanded.load_state_dict(legacy.state_dict())
        from models.network import Network
        owner = SimpleNamespace(args=expanded.args, actuator=expanded)
        bins = tuple(expanded._algorithmic_amount_bin_values_from_args(expanded.args))
        state = dict(autonomous_action_space=dict(schema='joint_amount_where_v2',
                     bins=bins, max_operation_ratio=.03))
        self.assertTrue(Network.load_actor_critic_runtime_state_dict(owner, state))
        state['autonomous_action_space']['bins'] = (bins[0]/2.,)+bins[1:]
        with self.assertRaisesRegex(ValueError, 'Amount'):
            Network.load_actor_critic_runtime_state_dict(owner, state)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA Chamfer comparison')
    def test_exact_kdtree_matches_cuda_distances_indices_and_gradients(self):
        from models.utils.loss.autonomous_exact_geometry import nearest_squared
        from models.utils.loss.utils_loss import chamfer_dist
        for lattice in (False, True):
            torch.manual_seed(44)
            q = torch.randn(1, 40, 3, device='cuda')
            r = torch.randn(1, 600, 3, device='cuda')
            if lattice:
                q, r = q.round(), r.round()
            q.requires_grad_()
            r.requires_grad_()
            expected, _, expected_index, _ = chamfer_dist(q, r)
            fast, fast_index = nearest_squared(q, r, chamfer_dist)
            self.assertTrue(torch.equal(expected, fast))
            self.assertTrue(torch.equal(expected_index.long(), fast_index))
            weights = torch.rand_like(expected)
            old_grad = torch.autograd.grad((expected * weights).sum(), (q, r), retain_graph=True)
            new_grad = torch.autograd.grad((fast * weights).sum(), (q, r))
            for a, b in zip(old_grad, new_grad):
                self.assertTrue(torch.allclose(a, b, atol=2e-6, rtol=1e-6), float((a-b).abs().max()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA fallback comparison')
    def test_large_tie_set_falls_back_instead_of_truncating(self):
        from models.utils.loss.autonomous_exact_geometry import nearest_squared
        from models.utils.loss.utils_loss import chamfer_dist
        q = torch.full((1, 1, 3), .5, device='cuda')
        r = torch.zeros(1, 300, 3, device='cuda')
        self.assertIsNone(nearest_squared(q, r, chamfer_dist))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA full CD comparison')
    def test_retained_points_and_weighted_chamfer_preserve_values_and_gradients(self):
        from models.utils.loss.autonomous_exact_geometry import nearest_squared, chamfer_terms
        from models.utils.loss.utils_loss import chamfer_dist, chamfer_l2_loss, chamfer_l2_loss_and_weight_surrogate
        torch.manual_seed(13)
        gt = torch.rand(1, 3, 1000, device='cuda').requires_grad_()
        gen = torch.cat([gt.detach()[:, :, :970], torch.rand(1, 3, 40, device='cuda')], 2).requires_grad_()
        w = (torch.rand(1, 1, gen.shape[-1], device='cuda') * 1.4 - .2).requires_grad_()
        hard, soft, surrogate = chamfer_terms(gen, gt, w, chamfer_dist)
        old_hard, old_surrogate = chamfer_l2_loss_and_weight_surrogate(gen, gt, w)
        old_soft = chamfer_l2_loss(gen, gt, w)
        for a, b in ((hard, old_hard), (soft, old_soft), (surrogate, old_surrogate)):
            self.assertTrue(torch.equal(a, b), (float(a), float(b)))
        for old, new in ((old_hard, hard), (old_soft, soft), (old_surrogate, surrogate)):
            grad_old = torch.autograd.grad(old, (gen, gt, w), retain_graph=True, allow_unused=True)
            grad_new = torch.autograd.grad(new, (gen, gt, w), retain_graph=True, allow_unused=True)
            for a, b in zip(grad_old, grad_new):
                if a is None or b is None:
                    self.assertIs(a, b)
                else:
                    self.assertTrue(torch.allclose(a, b, atol=2e-7, rtol=1e-6))
        stats = []
        zero, _ = nearest_squared(gt.transpose(1, 2).contiguous(), gt.transpose(1, 2).contiguous(), chamfer_dist, stats)
        self.assertEqual(stats[0]['skipped_zero'], 1000)
        self.assertEqual(float(zero.sum()), 0.)
        self.assertEqual(float(torch.autograd.grad(zero.sum(), gt)[0].abs().sum()), 0.)

if __name__ == '__main__':
    unittest.main()
