import argparse
from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from models.utils.pointcloud.geometry_octree_rate import GeometryOctreeRate, remember_rate_model
from models.utils.pointcloud.geometry_joint_policy import select_joint_geometry_plan, GeometryJointContext
from models.utils.pointcloud.geometry_octree_proposals import attach_geometry_octree_guidance
from models.utils.pointcloud.geometry_octree import build_geometry_octree_context
from models.utils.pointcloud.geometry_rd import set_geometry_reference, geometry_penalty
from models.utils.config.args import parse_pugan_args
from models.modules.geometry_octree_structure import GeometryOctreeStructureAnalysis
from models.modules.octree_structure import OctreeStructureAnalysis


def args_for(codec='G-PCC', **values):
    return SimpleNamespace(**dict(dict(compress=codec, gpcc_prequantize=True,
        gpcc_effective_qs=2., qs=2., cp_tau_geom=0., geometry_octree_joint_policy=True), **values))


class GeometryJointPolicyTest(unittest.TestCase):
    def test_frozen_cost_delta_includes_neighbours_and_ancestors(self):
        rows = np.array([[x, y, z] for x in range(6) for y in range(4) for z in range(2)
                         if (x + y + z) % 3 != 0])
        for codec in ('gpcc', 'octattention_categorical'):
            model = GeometryOctreeRate(rows, codec)
            candidate = dict(remove_coords=rows[:6].tolist(), add_coords=[[0, 0, 0]])
            changed, gain = model.patch(candidate)
            after = dict(model.masks, **{})
            after.update(changed)
            keys = set(model.masks) | set(after)
            direct = sum(model.cost(k, model.masks.get(k, 0), model.masks)
                         - model.cost(k, after.get(k, 0), after) for k in keys)
            self.assertAlmostEqual(gain, direct, places=9)
            self.assertEqual(model.masks, GeometryOctreeRate(rows, codec).masks)

    def test_combined_gain_is_not_sum_of_independent_gains(self):
        model = GeometryOctreeRate(np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]]), 'octattention_categorical')
        a = dict(remove_coords=[[0, 0, 0]], add_coords=[])
        b = dict(remove_coords=[[1, 0, 0]], add_coords=[])
        patch_a, ga = model.patch(a)
        _, gb = model.patch(b)
        state = dict(model.masks); state.update(patch_a)
        _, conditional = model.patch(b, state)
        self.assertNotAlmostEqual(gb, conditional)
        _, joint = model.patch(dict(remove_coords=a['remove_coords']+b['remove_coords'], add_coords=[]))
        self.assertAlmostEqual(joint, ga + conditional)

    def fixture(self):
        args = args_for(geometry_octree_initial_total_ratio=.05)
        points = torch.tensor([[[x, y, z] for x in range(12) for y in range(8)
                                 for z in range(4) if (x+y+z) % 3 != 0]], dtype=torch.float32).transpose(1, 2) * 2
        ctx = build_geometry_octree_context(points, args)
        exact = attach_geometry_octree_guidance(ctx, args, device=points.device)['ana_den6_ranked_candidate_guidance']
        pools = exact['operation_edit_units']
        scores = {op: torch.zeros(len(pool), requires_grad=True) for op, pool in pools.items()}
        return args, ctx['global_voxel_coords'], exact, pools, scores

    def test_joint_gradients_budgets_legality_and_actual_encodes(self):
        args, coords, exact, pools, scores = self.fixture()
        context_head = GeometryJointContext()
        result = select_joint_geometry_plan(coords, exact, pools,
            {'Add': 1, 'Prune': 1, 'Adjust': 1}, scores, args, exploration=.7, context_head=context_head)
        selected, removes, adds, _, counts, logp, entropy, audit = result
        occupied = set(map(tuple, coords[0].T.tolist()))
        self.assertTrue(removes <= occupied)
        self.assertFalse(adds & occupied)
        self.assertFalse(removes & adds)
        self.assertTrue(all(counts[op] <= 1 for op in counts))
        self.assertEqual(audit['actual_candidate_encodes'], 0)
        logp.backward()
        self.assertGreater(float(context_head.weight.grad.norm()), 0.)
        for op in scores:
            self.assertTrue(torch.isfinite(scores[op].grad).all())
            self.assertGreater(float(scores[op].grad.norm()), 0.)

    def test_zero_plan_is_valid_and_does_not_credit_where(self):
        args, coords, exact, pools, scores = self.fixture()
        result = select_joint_geometry_plan(coords, exact, pools,
            dict.fromkeys(pools, 0), scores, args)
        self.assertEqual(result[0], [])
        result[5].backward()
        self.assertTrue(all(float(v.grad.norm()) == 0. for v in scores.values()))

    def test_deterministic_when_noise_is_zero_and_respects_proposal_mask(self):
        args, coords, exact, pools, scores = self.fixture()
        masks = {op: torch.ones(len(p), dtype=torch.bool) for op, p in pools.items()}
        masks['Add'][:] = False
        one = select_joint_geometry_plan(coords, exact, pools,
            dict.fromkeys(pools, 1), scores, args, masks=masks)
        torch.manual_seed(992)
        two = select_joint_geometry_plan(coords, exact, pools,
            dict.fromkeys(pools, 1), scores, args, masks=masks)
        self.assertEqual([(op, i) for op, i, _ in one[0]], [(op, i) for op, i, _ in two[0]])
        self.assertEqual(one[4]['Add'], 0)

    def test_noise_changes_executed_voxel_sets_not_just_a_flag(self):
        args, coords, exact, pools, scores = self.fixture()
        edits = set()
        for seed in range(8):
            torch.manual_seed(seed)
            result = select_joint_geometry_plan(coords, exact, pools,
                dict.fromkeys(pools, 2), scores, args, exploration=.8)
            edits.add((tuple(sorted(result[1])), tuple(sorted(result[2]))))
            self.assertEqual(len(result[1]), result[4]['Prune'] + result[4]['Adjust'])
            self.assertEqual(len(result[2]), result[4]['Add'] + result[4]['Adjust'])
        self.assertGreater(len(edits), 1)

    def test_geometry_entropy_uses_parent_count_not_sum_squared_frequencies(self):
        args = args_for()
        coords = torch.tensor([[[2*x, 0, 0] for x in range(300)]
                               + [[2*x+1, 0, 0] for x in range(150)]], dtype=torch.float32).transpose(1, 2)
        context = {'global_voxel_coords': coords.long()}
        old = OctreeStructureAnalysis(args)._prebuilt_octree_context(coords, full_octree_context=context)
        new = GeometryOctreeStructureAnalysis(args)._prebuilt_octree_context(coords, full_octree_context=context)
        self.assertTrue(torch.all(old[:, 4] == 1))
        self.assertTrue(torch.all(new[:, 4] < 1))
        self.assertTrue(torch.allclose(new[:, 3], torch.full_like(new[:, 3], .5)))

    def test_solid_rectangular_cloud_keeps_boundary_add_and_adjust(self):
        args = args_for()
        points = torch.tensor([[[x, y, z] for x in range(12) for y in range(8)
                               for z in range(4)]], dtype=torch.float32).transpose(1, 2) * 2
        context = build_geometry_octree_context(points, args)
        pools = attach_geometry_octree_guidance(context, args, device=points.device)['ana_den6_ranked_candidate_guidance']['operation_edit_units']
        self.assertTrue(all(pools[op] for op in ('Add', 'Prune', 'Adjust')))

    def test_rd_reference_preserves_raw_geometry_and_voxel_gradient(self):
        for codec in ('G-PCC', 'OctAttention'):
            args = args_for(codec)
            points = torch.tensor([[[1., 3., 4., 6.], [1., 0., 2., 3.], [3., 5., 7., 9.]]])
            set_geometry_reference(points, args)
            geometry = torch.tensor(args._geometry_noop_chamfer + .4, requires_grad=True)
            value = geometry_penalty(geometry, args)
            self.assertAlmostEqual(float(value), .1, places=6)
            value.backward()
            self.assertAlmostEqual(float(geometry.grad), .25)
            self.assertAlmostEqual(float(geometry_penalty(torch.tensor(args._geometry_noop_chamfer), args)), 0.)

    def test_sparse_rd_and_configuration_are_unchanged(self):
        args = args_for('SparsePCGC', cp_tau_geom=.06)
        # Explicitly avoid duplicate namespace field in the helper above.
        value = torch.tensor(.1, requires_grad=True)
        self.assertAlmostEqual(float(geometry_penalty(value, args)), .04, places=6)
        set_geometry_reference(torch.zeros(1, 3, 4), args)
        self.assertFalse(hasattr(args, '_geometry_noop_chamfer'))
        parser = argparse.ArgumentParser(); parser.add_argument('--trainORtest', default='train')
        with patch('sys.argv', ['test', '--compression', 'SparsePCGC']):
            parsed = parse_pugan_args(parser, '20261016', 'test')
        self.assertEqual(parsed.cp_geometry_backward_scale, .036)
        self.assertEqual(parsed.cp_tau_geom, 0.)  # pre-existing SparsePCGC normalization
        self.assertEqual(parsed.heuristic_guidance_online_amount_bins, '0.0015,0.002,0.0025,0.00275,0.003')

    def test_geometry_amount_initialization_does_not_inherit_sparse_three_percent(self):
        parser = argparse.ArgumentParser(); parser.add_argument('--trainORtest', default='train')
        with patch('sys.argv', ['test', '--compression', 'G-PCC']):
            parsed = parse_pugan_args(parser, '20261016', 'test')
        self.assertEqual(parsed.sparsepcgc_algorithmic_amount_init_ratio, parsed.geometry_octree_initial_total_ratio)


if __name__ == '__main__':
    unittest.main()
