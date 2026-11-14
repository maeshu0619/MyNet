"""Naming compatibility; these analytic fixtures are not codec measurements."""
import importlib
from types import SimpleNamespace
import unittest

import torch

from models.modules.voxel_edit_controller import VoxelEditController
from models.network import Network


class ReadableVoxelAPITest(unittest.TestCase):
    def test_legacy_imports_share_the_actual_module_and_cache_globals(self):
        pairs = (
            ('models.modules.octree_structure', 'models.modules.voxel_structure_features'),
            ('models.modules.structure_actuator', 'models.modules.voxel_edit_controller'),
            ('models.utils.pointcloud.ana_den6_online',
             'models.utils.pointcloud.voxel_candidate_guidance'),
            ('models.utils.pointcloud.ana_den6_reference',
             'models.utils.pointcloud.reference_voxel_plan'),
            ('models.utils.training.autonomous_joint_learning',
             'models.utils.training.joint_voxel_policy'),
        )
        for old, new in pairs:
            self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_semantic_head_views_do_not_duplicate_checkpoint_or_optimizer_parameters(self):
        controller = VoxelEditController(56, args=SimpleNamespace(
            compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online'))
        pairs = {
            'candidate_actor': 'den6_candidate_actor',
            'candidate_proposal': 'den6_candidate_proposal',
            'candidate_critic': 'den6_candidate_critic',
            'plan_value_estimator': 'den6_plan_critic',
            'prune_location_head': 'drop_head',
            'prune_amount_head': 'drop_amount_head',
            'adjust_direction_head': 'move_voxel_head',
            'amount_bin_head': 'algorithmic_amount_selector_head',
        }
        keys = tuple(controller.state_dict())
        parameters = tuple(controller.parameters())
        for modern, legacy in pairs.items():
            self.assertIs(getattr(controller, modern), getattr(controller, legacy))
            self.assertTrue(any(key.startswith(legacy + '.') for key in keys))
            self.assertFalse(any(key.startswith(modern + '.') for key in keys))
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        restored = VoxelEditController(56, args=controller.args)
        restored.load_state_dict(controller.state_dict(), strict=True)
        self.assertEqual(tuple(restored.state_dict()), keys)
        for name, value in controller.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name],
                                       rtol=0, atol=0, equal_nan=True)

    def test_legacy_candidate_methods_keep_the_same_results(self):
        candidate = dict(remove_coords=[[1, 2, 3]], add_coords=[[2, 2, 3]])
        self.assertEqual(VoxelEditController._candidate_coordinate_changes(candidate),
                         VoxelEditController._den6_candidate_coord_sets(candidate))
        self.assertIs(VoxelEditController.build_joint_voxel_edit_plan,
                      VoxelEditController._build_exact_den6_residual_plan)

    def test_network_module_views_retain_registration_names(self):
        # Avoid the expensive full Network constructor; test the registration contract.
        model = Network.__new__(Network)
        torch.nn.Module.__init__(model)
        model.encoder = torch.nn.Linear(2, 2)
        model.structure_analyzer = torch.nn.Identity()
        model.actuator = torch.nn.Linear(2, 1)
        self.assertIs(model.geometry_encoder, model.encoder)
        self.assertIs(model.voxel_structure_extractor, model.structure_analyzer)
        self.assertIs(model.voxel_edit_controller, model.actuator)
        self.assertEqual(tuple(model._modules), ('encoder', 'structure_analyzer', 'actuator'))


if __name__ == '__main__':
    unittest.main()
