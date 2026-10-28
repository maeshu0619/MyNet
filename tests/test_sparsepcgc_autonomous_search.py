from types import SimpleNamespace
from unittest.mock import patch
import unittest
import json
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from models.utils.training.sparsepcgc_autonomous_policy import (
    autonomous_amount_bins, autonomous_pool_phase, autonomous_search_enabled,
    validation_promotion, align_autonomous_context,
)


class AutonomousSearchTest(unittest.TestCase):
    def args(self, dataset='8i'):
        return SimpleNamespace(compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online',
                               test_route=True, dataname=dataset,
                               sparsepcgc_scale_m=8, sparsepcgc_native_resolution=31,
                               heuristic_guidance_online_max_total_ratio=.0099,
                               heuristic_guidance_online_fast_pool_max_per_operation=8192,
                               _global_train_step=0, _fixed_episode_steps=100)

    def test_mode_is_opt_in_and_sparsepcgc_only(self):
        args = self.args()
        self.assertTrue(autonomous_search_enabled(args))
        for codec in ('G-PCC', 'OctAttention'):
            args.compress = codec
            self.assertFalse(autonomous_search_enabled(args))
        args = self.args()
        args.test_route = False
        self.assertFalse(autonomous_search_enabled(args))

    def test_false_does_not_reconfigure_or_use_the_historical_switch(self):
        from models.utils.training.sparsepcgc_autonomous_policy import configure_autonomous_search
        args = self.args()
        args.test_route = False
        args.sparsepcgc_autonomous_search = True
        previous = dict(vars(args))
        self.assertIs(configure_autonomous_search(args), args)
        self.assertEqual(vars(args), previous)
        self.assertFalse(autonomous_search_enabled(args))

    def test_true_rejects_other_codec_and_guidance_routes(self):
        from models.utils.training.sparsepcgc_autonomous_policy import configure_autonomous_search
        for key, value in [('compress', 'G-PCC'), ('compress', 'OctAttention'),
                           ('heuristic_guidance_mode', 'single_plan_student')]:
            args = self.args()
            setattr(args, key, value)
            previous = dict(vars(args))
            with self.assertRaises(ValueError):
                configure_autonomous_search(args)
            self.assertEqual(vars(args), previous)

    def test_absolute_bins_span_small_to_maximum_without_dataset_profile(self):
        bins = autonomous_amount_bins(.03)
        self.assertEqual(len(bins), 15)
        self.assertAlmostEqual(bins[-1] / bins[0], 300)
        self.assertEqual(bins[-1], .03)
        self.assertTrue(all(0 < value <= .03 for value in bins))

    def test_reservoir_stays_fixed_for_a_comparison_then_rotates(self):
        args = self.args()
        self.assertEqual(autonomous_pool_phase(args), 0)
        args._global_train_step = 199
        self.assertEqual(autonomous_pool_phase(args), 0)
        args._global_train_step = 200
        self.assertEqual(autonomous_pool_phase(args), 1)
        args._autonomous_policy_training = False
        self.assertEqual(autonomous_pool_phase(args), 0)

    def test_pool_rotation_stops_with_the_noise_window(self):
        args = self.args()
        args._exploration_schedule_steps_estimate = 1000
        args.repair_exploration_fraction = .25
        phases = []
        for step in (249, 250, 500, 999):
            args._global_train_step = step
            phases.append(autonomous_pool_phase(args))
        self.assertEqual(phases, [1, 1, 1, 1])

    def test_promotion_requires_actual_and_geometry_on_every_same_frame(self):
        reference = [dict(frame='a', codec_setting='native1', actual_percent=-3.9, geometry=.03),
                     dict(frame='b', codec_setting='native1', actual_percent=-3., geometry=.02)]
        candidate = [dict(row) for row in reference]
        self.assertTrue(validation_promotion(candidate, reference)['eligible'])
        candidate[0]['actual_percent'] = -2.
        candidate[1]['actual_percent'] = -6.
        self.assertFalse(validation_promotion(candidate, reference)['eligible'])
        candidate = [dict(row) for row in reference]
        candidate[0]['geometry'] = .03001
        self.assertFalse(validation_promotion(candidate, reference)['eligible'])
        self.assertFalse(validation_promotion(candidate[:1], reference)['eligible'])
        self.assertFalse(validation_promotion(candidate*2, reference)['eligible'])
        candidate[0]['geometry'] = float('nan')
        self.assertFalse(validation_promotion(candidate, reference)['eligible'])

    def test_current_pool_is_dataset_independent_and_rotates_without_stale_signature(self):
        from models.utils.pointcloud.ana_den6_online import _build_fast_unordered_teacher
        coords = torch.cartesian_prod(torch.arange(2, 20), torch.arange(2, 20), torch.arange(2, 20))
        coords = coords[(coords.sum(dim=1) % 3) == 0]
        context = {'global_voxel_coords': coords.T.unsqueeze(0)}
        identity = {'input_sha256': '3' * 64, 'setting_id': 'same_codec'}
        outputs = [_build_fast_unordered_teacher(context, self.args(dataset), identity)
                   for dataset in ('8i', 'MVUB', 'UVG')]
        self.assertEqual(outputs[0]['operation_edit_units'], outputs[1]['operation_edit_units'])
        self.assertEqual(outputs[0]['operation_edit_units'], outputs[2]['operation_edit_units'])
        occupied = set(map(tuple, coords.tolist()))
        for op, pool in outputs[0]['operation_edit_units'].items():
            self.assertTrue(pool)
            for candidate in pool:
                self.assertEqual(candidate['rank_score'], 0.)
                for source in candidate['remove_coords']:
                    self.assertIn(tuple(source), occupied)
                    self.assertTrue(all(2 < c < 19 for c in source))
                for target in candidate['add_coords']:
                    self.assertNotIn(tuple(target), occupied)
                    self.assertTrue(all(2 <= c <= 19 for c in target))
        args = self.args()
        args._global_train_step = 200
        rotated = _build_fast_unordered_teacher(context, args, identity)
        self.assertNotEqual(rotated['cache_signature'], outputs[0]['cache_signature'])
        self.assertNotEqual(rotated['operation_edit_units'], outputs[0]['operation_edit_units'])

    def test_autonomous_mapping_does_not_consult_den6_profiles(self):
        from models.utils.pointcloud.ana_den6_online import _build_fast_unordered_teacher
        from models.modules.heuristic_guidance import build_heuristic_guidance
        coords = torch.cartesian_prod(torch.arange(2, 10), torch.arange(2, 10), torch.arange(2, 10))
        coords = coords[(coords.sum(dim=1) % 3) == 0]
        args = self.args()
        payload = _build_fast_unordered_teacher({'global_voxel_coords': coords.T.unsqueeze(0)},
                     args, {'input_sha256': '3' * 64, 'setting_id': 'same_codec'})
        structure = {'global_voxel_coords': coords.T.unsqueeze(0),
                     'occupancy_nll_proxy': torch.zeros(1, 1, len(coords)),
                     'ana_den6_ranked_candidate_guidance': payload}
        with patch('models.modules.heuristic_guidance.resolve_profile', side_effect=AssertionError('den6 profile')):
            result = build_heuristic_guidance(structure, args)
        self.assertTrue(result['enabled'])

    def test_structure_rows_follow_neural_node_coordinates(self):
        from models.modules.octree_structure import OctreeStructureAnalysis
        args = self.args()
        model = OctreeStructureAnalysis(args)
        coords = torch.tensor([[2, 2, 2], [3, 2, 2], [3, 3, 3], [4, 3, 4], [7, 6, 7]])
        permuted = coords[torch.tensor([4, 0, 3, 1, 2])].T.unsqueeze(0)
        original = {'global_voxel_coords': coords.T.unsqueeze(0), 'global_qs': 1.}
        aligned = align_autonomous_context(original, permuted)
        features = model._prebuilt_octree_context(permuted.float(), full_octree_context=aligned)
        slots = ((permuted[:, 0] & 1)*4 + (permuted[:, 1] & 1)*2 + (permuted[:, 2] & 1)).float()/7.
        self.assertTrue(torch.allclose(features[:, 7], slots))
        self.assertTrue(torch.equal(original['global_voxel_coords'], coords.T.unsqueeze(0)))
        self.assertNotEqual(model._fixed_structure_cache_key('frame'),
                            OctreeStructureAnalysis(SimpleNamespace())._fixed_structure_cache_key('frame'))

    def test_unverified_policy_cannot_overwrite_best_checkpoint(self):
        from models.utils.training.checkpointing import save_episode_checkpoint
        args = self.args()
        args.compression_loss_backend = 'sparsepcgc_actual'
        model = torch.nn.Linear(1, 1)
        writer = SimpleNamespace(write=lambda text: None)
        plot = SimpleNamespace(epi_loss_return=lambda: 1.)
        metrics = dict(total_loss=1., checkpoint_actual_delta=-4., checkpoint_actual_count=4,
                       full_cloud_val_actual_count=4, full_cloud_val_fixed_objective=-10.,
                       geometry_ok=True, safety_ok=True, checkpoint_eligible=True)
        with TemporaryDirectory() as directory:
            best = Path(directory)/'best.pth'
            best.write_bytes(b'previous verified model')
            save_episode_checkpoint(model, directory, plot, writer, 0, float('inf'),
                                    args=args, stage='joint', checkpoint_metrics=metrics)
            self.assertEqual(best.read_bytes(), b'previous verified model')
            self.assertTrue((Path(directory)/'0.pth').is_file())
            self.assertTrue((Path(directory)/'candidate_loss_joint.pth').is_file())

    def test_measured_reference_accepts_only_nonregressing_policy(self):
        from models.utils.training.sparsepcgc_autonomous_policy import autonomous_checkpoint_status
        args = self.args()
        row = dict(frame='fixed.ply', input_sha256='a'*64, codec_setting='codec',
                   geometry_setting='geom', actual_percent=-3.9, geometry=.03)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'reference.json'
            path.write_text(json.dumps(dict(measurements=[row])))
            args.sparsepcgc_autonomous_reference_json = str(path)
            args._autonomous_validation_measurements = [dict(row)]
            self.assertTrue(autonomous_checkpoint_status(args)['eligible'])
            args.train_full_cloud_val_frames = 2
            self.assertFalse(autonomous_checkpoint_status(args)['eligible'])
            args.train_full_cloud_val_frames = 1
            args._autonomous_validation_measurements[0]['actual_percent'] = -2.
            self.assertFalse(autonomous_checkpoint_status(args)['eligible'])
            args._autonomous_validation_measurements = []
            self.assertFalse(autonomous_checkpoint_status(args)['eligible'])

    def test_disabled_validation_clears_stale_promotion_measurements(self):
        from models.utils.training.train_runtime import run_episode_full_cloud_validation
        args = self.args()
        args.train_full_cloud_val_frames = 0
        args._autonomous_validation_measurements = [{'stale': True}]
        result = run_episode_full_cloud_validation(model=None, args=args, loss=None,
            writer=None, seq_datasets=[], episode=0, global_step=100,
            use_cuda=False, use_amp=False, amp_dtype=torch.float32)
        self.assertEqual(result['count'], 0)
        self.assertEqual(args._autonomous_validation_measurements, [])

    def test_diagnostic_validation_preserves_network_measurements(self):
        from models.utils.training.train_runtime import run_episode_full_cloud_validation
        args = self.args()
        args.train_full_cloud_val_frames = 0
        measured = [{'network': True}]
        args._autonomous_validation_measurements = measured
        run_episode_full_cloud_validation(model=None, args=args, loss=None,
            writer=None, seq_datasets=[], episode=0, global_step=100,
            use_cuda=False, use_amp=False, amp_dtype=torch.float32,
            diagnostic_policy_override='heuristic')
        self.assertIs(args._autonomous_validation_measurements, measured)

    def test_evaluation_coverage_is_repeatable_without_mutating_training_visits(self):
        from models.utils.training.sparsepcgc_autonomous_policy import proposal_coverage_index
        visits = OrderedDict([('frame', 17)])
        for _ in range(3):
            self.assertEqual(proposal_coverage_index(visits, 'frame', training=False, autonomous=True), 0)
            self.assertEqual(visits['frame'], 17)
        self.assertEqual(proposal_coverage_index(visits, 'frame', training=True, autonomous=True), 17)
        self.assertEqual(visits['frame'], 18)
        self.assertEqual(proposal_coverage_index(visits, 'frame', training=False, autonomous=False), 18)



if __name__ == '__main__':
    unittest.main()
