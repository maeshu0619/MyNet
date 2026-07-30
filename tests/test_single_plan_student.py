from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from models.modules.executable_voxel_plan import apply_selected_executable_plan
from models.modules.fast_heuristic_emulator import FastHeuristicEmulator
from models.modules.single_plan_student import SinglePlanStudentPolicy
from models.utils.training.optim_amp import (
    build_emulator_optimizer_and_scheduler,
    build_optimizer_and_scheduler,
)


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
        single_plan_training_stage="representation",
        single_plan_amount_learning_enabled=False,
        single_plan_fixed_total_ratio=0.0025,
        single_plan_fixed_prune_share=0.40,
        single_plan_fixed_add_share=0.40,
        single_plan_fixed_adjust_share=0.20,
    )


class SinglePlanStudentTest(unittest.TestCase):
    def test_fast_emulator_reuses_single_plan_contract(self):
        emulator = FastHeuristicEmulator(in_channels=8, hidden_dim=16, fixed_feature_dim=6)
        self.assertIsInstance(emulator, SinglePlanStudentPolicy)

    def test_ana_den6_emulator_optimizer_is_disjoint(self):
        class DummyWriter:
            def write(self, _message):
                pass

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.main = nn.Linear(2, 2)
                self.single_plan_student = nn.Linear(2, 2)

        args = SimpleNamespace(
            deform=False,
            encoder_0grad=True,
            heuristic_guidance_mode="ana_den6_online",
            single_plan_shadow_distillation=True,
            single_plan_student_lr_scale=1.0,
            optim="adam",
            lr=1e-3,
            weight_decay=0.0,
            lr_decay_step=10,
            gamma=0.5,
        )
        model = DummyModel()
        main_optimizer, _ = build_optimizer_and_scheduler(model, args, DummyWriter())
        emulator_optimizer, _ = build_emulator_optimizer_and_scheduler(
            model, args, DummyWriter()
        )
        main_ids = {
            id(parameter)
            for group in main_optimizer.param_groups
            for parameter in group["params"]
        }
        emulator_ids = {
            id(parameter)
            for group in emulator_optimizer.param_groups
            for parameter in group["params"]
        }
        student_ids = {
            id(parameter) for parameter in model.single_plan_student.parameters()
        }
        self.assertFalse(main_ids & emulator_ids)
        self.assertEqual(emulator_ids, student_ids)

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

    def test_fast_inference_builder_keeps_executable_plan(self):
        torch.manual_seed(9)
        points = 1024
        coords = torch.stack((
            torch.arange(points) * 2, torch.zeros(points, dtype=torch.long),
            torch.zeros(points, dtype=torch.long),
        )).view(1, 3, points)
        features = torch.randn(1, 12, points)
        fixed = torch.randn(1, 6, points)
        model = SinglePlanStudentPolicy(12, hidden_dim=16).eval()
        diagnostic_args = _args()
        diagnostic_args.single_plan_collect_reject_reasons = True
        fast_args = _args()
        fast_args.single_plan_debug_hash = True
        fast_args.single_plan_collect_reject_reasons = False
        with torch.no_grad():
            diagnostic = model(
                features, coords, diagnostic_args, training=False,
                fixed_features=fixed,
            )
            fast = model(
                features, coords, fast_args, training=False,
                fixed_features=fixed,
            )
        self.assertEqual(
            diagnostic["executable_plan"].plan_hash,
            fast["executable_plan"].plan_hash,
        )
        self.assertTrue(torch.equal(
            diagnostic["executable_plan"].accepted_count,
            fast["executable_plan"].accepted_count,
        ))

    def test_representation_train_and_test_use_identical_plan(self):
        torch.manual_seed(11)
        points = 4000
        coords = torch.stack((
            torch.arange(points) * 2,
            torch.zeros(points, dtype=torch.long),
            torch.zeros(points, dtype=torch.long),
        )).view(1, 3, points)
        features = torch.randn(1, 12, points)
        fixed = torch.randn(1, 6, points)
        model = SinglePlanStudentPolicy(12, hidden_dim=16)
        args = _args()
        model.train()
        train_terms = model.generate_plan(
            features, coords, args, training=True, fixed_features=fixed
        )
        model.eval()
        with torch.no_grad():
            test_terms = model.generate_plan(
                features, coords, args, training=False, fixed_features=fixed
            )
        self.assertEqual(
            train_terms["executable_plan"].plan_hash,
            test_terms["executable_plan"].plan_hash,
        )
        self.assertTrue(torch.equal(
            train_terms["executed_requested_count"],
            torch.tensor([[[4, 4, 2]]]),
        ))
        self.assertTrue(torch.equal(
            train_terms["executed_operation_order"],
            torch.tensor([[[1, 0, 2]]]),
        ))


if __name__ == "__main__":
    unittest.main()
