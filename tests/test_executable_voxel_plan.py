import unittest

import torch

from models.modules.executable_voxel_plan import (
    ADD,
    ADJUST,
    PRUNE,
    ExecutableVoxelPlanBuilder,
    apply_selected_executable_plan,
    coordinate_indices,
    executable_plan_hashes,
    scatter_amax_1d_compat_,
    select_executable_plan,
)
from models.modules.structure_actuator import StructureRepairActuator


class ExecutableVoxelPlanTest(unittest.TestCase):
    def test_scatter_amax_compat_matches_scalar_reference_and_keeps_gradient(self):
        index = torch.tensor([3, 1, 3, 4, 1, 3], dtype=torch.long)
        source = torch.tensor(
            [0.2, 0.5, 0.9, 0.4, 0.7, 0.8], requires_grad=True
        )
        output = torch.zeros(6)
        actual = scatter_amax_1d_compat_(output, index, source)
        expected = torch.tensor([0.0, 0.7, 0.0, 0.9, 0.4, 0.0])
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertTrue(torch.equal(
            source.grad,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
        ))

    def test_coordinate_indices_exact_sparse_join(self):
        reference = torch.tensor([[2, 0, 0], [0, 0, 0], [1, 0, 0]])
        query = torch.tensor([[1, 0, 0], [9, 9, 9], [2, 0, 0]])
        self.assertEqual(coordinate_indices(query, reference).tolist(), [2, -1, 0])

    def _inputs(self, order):
        coords = torch.tensor([[[0, 2, 4], [0, 0, 0], [0, 0, 0]]])
        # 同じsourceだけを3操作が要求し、operation orderの差を明確にする。
        scores = torch.full((1, 2, 3, 3), -torch.inf)
        scores[:, :, :, 0] = 3.0
        requested = torch.ones(1, 2, 3, dtype=torch.long)
        orders = torch.tensor([[order, [PRUNE, ADD, ADJUST]]])
        directions = torch.full((1, 2, 2, 26, 3), -100.0)
        builder = ExecutableVoxelPlanBuilder()
        offsets = builder.neighbor_offsets
        plus_x = int(((offsets == torch.tensor((1, 0, 0))).all(1)).nonzero()[0])
        minus_x = int(((offsets == torch.tensor((-1, 0, 0))).all(1)).nonzero()[0])
        directions[:, :, 0, plus_x, 0] = 10.0
        directions[:, :, 0, minus_x, 1] = 10.0  # Add targetは両方とも[1,0,0]
        directions[:, :, 1, plus_x, 0] = 10.0
        return builder, coords, scores, requested, orders, directions

    def test_builds_post_collision_plan_and_hash_is_deterministic(self):
        values = self._inputs([ADD, PRUNE, ADJUST])
        plan = values[0].build(*values[1:], debug_hash=True)
        self.assertEqual(tuple(plan.plan_descriptor.shape), (1, 2, 29))
        self.assertEqual(int(plan.accepted_count[0, 0, ADD]), 1)
        self.assertEqual(int(plan.accepted_count[0, 0, PRUNE]), 1)
        self.assertEqual(int(plan.accepted_count[0, 0, ADJUST]), 0)
        self.assertEqual(plan.plan_hash, executable_plan_hashes(plan))
        self.assertNotEqual(plan.plan_hash[0][0], plan.plan_hash[0][1])

    def test_priority_changes_conflicting_operation(self):
        builder, coords, scores, requested, _, directions = self._inputs([ADD, PRUNE, ADJUST])
        add_first = builder.build(
            coords, scores[:, :1], requested[:, :1],
            torch.tensor([[[ADD, PRUNE, ADJUST]]]), direction_logits=directions[:, :1],
        )
        prune_first = builder.build(
            coords, scores[:, :1], requested[:, :1],
            torch.tensor([[[PRUNE, ADD, ADJUST]]]), direction_logits=directions[:, :1],
        )
        self.assertFalse(torch.equal(add_first.accepted_count, prune_first.accepted_count))

    def test_add_and_adjust_store_source_target_and_direction(self):
        builder, coords, scores, requested, orders, directions = self._inputs([ADD, PRUNE, ADJUST])
        requested[:, :, PRUNE] = 0
        requested[:, :, ADJUST] = 0
        plan = builder.build(coords, scores, requested, orders, direction_logits=directions)
        mask = plan.accepted_mask[0, 0, ADD]
        source = plan.source_coord[0, 0, ADD][mask][0]
        target = plan.target_coord[0, 0, ADD][mask][0]
        offset = builder.neighbor_offsets[plan.direction_index[0, 0, ADD][mask][0]]
        self.assertTrue(torch.equal(source + offset, target))

    def test_selected_apply_count_matches_contract(self):
        values = self._inputs([ADD, PRUNE, ADJUST])
        plan = values[0].build(*values[1:])
        selected = select_executable_plan(plan, torch.tensor((0,)))
        self.assertEqual(selected.operation_order.shape[1], 1)
        edited, valid = apply_selected_executable_plan(values[1], plan, torch.tensor((0,)))
        self.assertEqual(int(valid.sum()), int(plan.final_count[0, 0]))
        self.assertEqual(int(torch.unique(edited[0, :, valid[0]].T, dim=0).shape[0]), int(valid.sum()))

    def test_actuator_external_path_applies_the_critic_plan_without_reselection(self):
        values = self._inputs([ADD, PRUNE, ADJUST])
        plan = values[0].build(*values[1:], debug_hash=True)
        selected = select_executable_plan(plan, torch.tensor((0,)))
        direct_coords, direct_valid = apply_selected_executable_plan(
            values[1], selected, torch.tensor((0,))
        )
        actuator_coords, actuator_valid = StructureRepairActuator._apply_external_executable_plan(
            values[1], selected
        )
        self.assertEqual(selected.plan_hash[0][0], executable_plan_hashes(selected)[0][0])
        self.assertTrue(torch.equal(direct_coords, actuator_coords))
        self.assertTrue(torch.equal(direct_valid, actuator_valid))

    def test_lazy_direction_provider_only_receives_source_window(self):
        builder, coords, scores, requested, orders, _ = self._inputs([ADD, PRUNE, ADJUST])
        requested[:, :, PRUNE] = 0
        requested[:, :, ADJUST] = 0
        calls = []

        def provider(batch, slot, operation, source_index):
            calls.append(int(source_index.numel()))
            logits = torch.zeros(source_index.numel(), 26)
            logits[:, 0] = 1.0
            return logits

        builder.build(
            coords, scores, requested, orders,
            direction_logit_provider=provider,
        )
        self.assertTrue(calls)
        self.assertLessEqual(max(calls), 3)

    def test_target_ste_keeps_hard_forward_and_direction_gradient(self):
        builder, coords, scores, requested, orders, directions = self._inputs(
            [ADD, PRUNE, ADJUST]
        )
        requested[:, :, PRUNE] = 0
        requested[:, :, ADJUST] = 0
        directions = torch.zeros_like(directions, requires_grad=True)
        plan = builder.build(
            coords, scores, requested, orders, direction_logits=directions
        )
        mask = plan.accepted_mask
        self.assertTrue(torch.equal(
            plan.target_coord_ste[mask], plan.target_coord[mask].float()
        ))
        plan.target_coord_ste[mask][:, 0].sum().backward()
        self.assertGreater(float(directions.grad.abs().sum()), 0.0)

    def test_coordinate_bounds_count_target_domain_rejection(self):
        builder, coords, scores, requested, orders, directions = self._inputs([ADD, PRUNE, ADJUST])
        requested[:, :, PRUNE] = 0
        requested[:, :, ADJUST] = 0
        directions.fill_(-torch.inf)
        offsets = builder.neighbor_offsets
        minus_x = int(((offsets == torch.tensor((-1, 0, 0))).all(1)).nonzero()[0])
        directions[:, :, 0, minus_x, 0] = 10.0
        plan = builder.build(
            coords, scores, requested, orders, direction_logits=directions,
            target_coord_min=torch.tensor((0, 0, 0)),
            target_coord_max=torch.tensor((8, 8, 8)),
        )
        self.assertEqual(int(plan.accepted_count[0, 0, ADD]), 0)
        self.assertGreater(int(plan.reject_reason_count[0, 0, ADD, 5]), 0)

    def test_direction_valid_mask_can_restrict_child_slot_mode(self):
        builder, coords, scores, requested, orders, directions = self._inputs([ADD, PRUNE, ADJUST])
        requested[:, :, PRUNE] = 0
        requested[:, :, ADJUST] = 0
        valid = torch.zeros_like(directions, dtype=torch.bool)
        offsets = builder.neighbor_offsets
        plus_z = int(((offsets == torch.tensor((0, 0, 1))).all(1)).nonzero()[0])
        valid[:, :, 0, plus_z, :] = True
        plan = builder.build(
            coords, scores, requested, orders,
            direction_logits=directions, direction_valid_mask=valid,
        )
        mask = plan.accepted_mask[0, 0, ADD]
        self.assertTrue(torch.all(plan.direction_index[0, 0, ADD][mask] == plus_z))

    def test_shortlist_scores_keep_global_source_indices(self):
        builder = ExecutableVoxelPlanBuilder()
        coords = torch.tensor([[[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]]])
        shortlist = torch.tensor([[1, 3]])
        scores = torch.full((1, 1, 3, 2), -torch.inf)
        scores[0, 0, PRUNE] = torch.tensor((1.0, 2.0))
        requested = torch.tensor([[[1, 0, 0]]])
        order = torch.tensor([[[PRUNE, ADD, ADJUST]]])
        plan = builder.build(
            coords, scores, requested, order, source_indices=shortlist
        )
        self.assertEqual(int(plan.source_index[0, 0, PRUNE, 0]), 3)


if __name__ == "__main__":
    unittest.main()
