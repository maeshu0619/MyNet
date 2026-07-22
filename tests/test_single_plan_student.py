from types import SimpleNamespace
import unittest

import torch

from models.modules.executable_voxel_plan import apply_selected_executable_plan
from models.modules.single_plan_student import SinglePlanStudentPolicy


def _args():
    return SimpleNamespace(
        sparsepcgc_psnr_resolution=1023, sparsepcgc_scale_ae=0,
        sparsepcgc_scale_sr=2, sparsepcgc_scale_m=8,
        sparsepcgc_voxel_size=1.0, sparsepcgc_pos_quantscale=1,
        sparsepcgc_native_bit_depth=10, _global_train_step=0,
        network_only_exploration_anneal_steps=200,
        network_only_action_exploration_floor=0.05,
        network_only_where_gumbel_scale=1.0,
        single_plan_debug_hash=True,
        single_plan_local_tile_size=0,
    )


class SinglePlanStudentTest(unittest.TestCase):
    def test_deterministic_one_plan_without_k_or_critic(self):
        torch.manual_seed(1)
        points = 2000
        coords = torch.stack((
            torch.arange(points), torch.zeros(points, dtype=torch.long),
            torch.zeros(points, dtype=torch.long),
        )).view(1, 3, points)
        features = torch.randn(1, 12, points)
        fixed = torch.randn(1, 6, points)
        model = SinglePlanStudentPolicy(12, hidden_dim=16).eval()
        self.assertFalse(any(
            token in name.lower()
            for name, _ in model.named_parameters()
            for token in ("critic", "slot", "actor")
        ))
        with torch.no_grad():
            first = model(features, coords, _args(), training=False, fixed_features=fixed)
            second = model(features, coords, _args(), training=False, fixed_features=fixed)
        self.assertEqual(first["proposal_count"], 1)
        self.assertEqual(first["critic_batch_count"], 0)
        self.assertIsNone(first["selected_slot"])
        self.assertEqual(
            first["executable_plan"].plan_hash,
            second["executable_plan"].plan_hash,
        )
        plan = first["executable_plan"]
        final, valid = apply_selected_executable_plan(
            coords, plan, torch.zeros(1, dtype=torch.long)
        )
        self.assertEqual(int(valid.sum()), int(plan.final_count[0, 0]))

    def test_exact_local_tile_keeps_logits_rank_and_plan_hash(self):
        torch.manual_seed(7)
        points = 512
        coords = torch.stack((
            torch.arange(points) * 2, torch.zeros(points, dtype=torch.long),
            torch.zeros(points, dtype=torch.long),
        )).view(1, 3, points)
        features = torch.randn(1, 12, points)
        fixed = torch.randn(1, 6, points)
        model = SinglePlanStudentPolicy(12, hidden_dim=16).eval()
        full_args = _args()
        tile_args = _args()
        tile_args.single_plan_local_tile_size = 73
        with torch.no_grad():
            full = model(features, coords, full_args, training=False, fixed_features=fixed)
            tiled = model(features, coords, tile_args, training=False, fixed_features=fixed)
        self.assertTrue(torch.allclose(
            full["base_where_logits"], tiled["base_where_logits"], atol=2e-7, rtol=0.0
        ))
        self.assertTrue(torch.equal(
            full["base_where_logits"].argsort(dim=2),
            tiled["base_where_logits"].argsort(dim=2),
        ))
        self.assertEqual(full["executable_plan"].plan_hash, tiled["executable_plan"].plan_hash)


if __name__ == "__main__":
    unittest.main()
