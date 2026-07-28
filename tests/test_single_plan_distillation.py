from types import SimpleNamespace
import unittest

import torch

from models.modules.single_plan_student import SinglePlanStudentPolicy
from models.network import Network
from models.utils.loss.single_plan_distillation import SinglePlanDistillationLoss
from models.utils.training.compression_primary_loss import (
    _compression_primary_support_balance,
    monotonic_support_scale,
)


class SinglePlanDistillationTest(unittest.TestCase):
    def test_shadow_distillation_is_bounded_by_primary_objective(self):
        """Shadow蒸留を残しつつ、圧縮主目的より大きい勾配支配を防ぐ。"""
        args = SimpleNamespace(
            single_plan_shadow_target_ratio=0.10,
            single_plan_shadow_balance_min_scale=0.01,
            single_plan_shadow_balance_max_scale=1.0,
        )
        primary = torch.tensor(-10.0)
        shadow = torch.tensor(100.0, requires_grad=True)
        balance = _compression_primary_support_balance(
            args,
            primary,
            shadow,
            target_ratio_name="single_plan_shadow_target_ratio",
            min_scale_name="single_plan_shadow_balance_min_scale",
            max_scale_name="single_plan_shadow_balance_max_scale",
        )
        self.assertAlmostEqual(balance["scale"], 0.01, places=7)
        weighted = balance["scale"] * shadow
        self.assertAlmostEqual(float(weighted.detach()), 1.0, places=7)
        weighted.backward()
        self.assertGreater(float(shadow.grad.abs()), 0.0)

    def test_shadow_scale_does_not_hide_distillation_convergence(self):
        """raw蒸留loss低下時にscaleを逆増幅せず、weighted lossも低下させる。"""
        args = SimpleNamespace(
            single_plan_shadow_target_ratio=0.10,
            single_plan_shadow_balance_min_scale=0.01,
            single_plan_shadow_balance_max_scale=1.0,
        )
        primary = torch.tensor(-10.0)
        first = _compression_primary_support_balance(
            args,
            primary,
            torch.tensor(100.0),
            target_ratio_name="single_plan_shadow_target_ratio",
            min_scale_name="single_plan_shadow_balance_min_scale",
            max_scale_name="single_plan_shadow_balance_max_scale",
        )
        later = _compression_primary_support_balance(
            args,
            primary,
            torch.tensor(10.0),
            target_ratio_name="single_plan_shadow_target_ratio",
            min_scale_name="single_plan_shadow_balance_min_scale",
            max_scale_name="single_plan_shadow_balance_max_scale",
        )
        effective_later = monotonic_support_scale(
            first["scale"], later["scale"]
        )
        self.assertAlmostEqual(first["scale"], 0.01, places=7)
        self.assertAlmostEqual(later["scale"], 0.10, places=7)
        self.assertAlmostEqual(effective_later, 0.01, places=7)
        self.assertLess(effective_later * 10.0, first["scale"] * 100.0)

    def test_operation_specific_teacher_has_gradient_without_add_source(self):
        torch.manual_seed(2)
        points = 128
        coords = torch.stack((
            torch.arange(points) * 3, torch.zeros(points, dtype=torch.long),
            torch.zeros(points, dtype=torch.long),
        )).view(1, 3, points)
        features = torch.randn(1, 8, points)
        fixed = torch.randn(1, 6, points)
        args = SimpleNamespace(
            sparsepcgc_psnr_resolution=1023, sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2, sparsepcgc_scale_m=8,
            sparsepcgc_voxel_size=1.0, sparsepcgc_pos_quantscale=1,
            sparsepcgc_native_bit_depth=10, _global_train_step=0,
            network_only_exploration_anneal_steps=200,
            network_only_action_exploration_floor=0.0,
            network_only_where_gumbel_scale=0.0,
            single_plan_debug_hash=False,
        )
        model = SinglePlanStudentPolicy(8, hidden_dim=16).train()
        terms = model(features, coords, args, training=False, fixed_features=fixed)
        teacher = {
            "total_ratio_percent": 0.25,
            "shares": {"Prune": 0.4, "Add": 0.4, "Adjust": 0.2},
            "operation_order": "Prune>Add>Adjust",
            "actual_gain_percent": 3.0,
            "geometry": {"D1_loss_db": 0.1, "D2_loss_db": 0.2},
            "candidates": [
                {"operation": "Prune", "remove_coords": [[0, 0, 0]], "add_coords": []},
                # Add source/directionは旧schemaどおり欠損のままにする。
                {"operation": "Add", "remove_coords": [], "add_coords": [[1, 0, 0]]},
                {"operation": "Adjust", "remove_coords": [[3, 0, 0]], "add_coords": [[4, 0, 0]]},
            ],
        }
        value, metrics = SinglePlanDistillationLoss()(terms, coords, teacher)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(metrics["add_target_reachable"], 0)
        self.assertIsNotNone(model.policy.local_cost_head.weight.grad)
        self.assertGreater(float(model.policy.local_cost_head.weight.grad.abs().sum()), 0.0)

    def test_den6_shadow_distillation_updates_persistent_counter(self):
        """Exact訓練中だけshadow蒸留を許可し、checkpoint契約値を増やす。"""
        class _Loss(torch.nn.Module):
            def forward(self, terms, voxel_coords, teacher):
                del voxel_coords, teacher
                value = terms["probe"].pow(2).mean()
                return value, {"probe": float(value.detach())}

        network = Network.__new__(Network)
        torch.nn.Module.__init__(network)
        network.args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            single_plan_shadow_distillation=True,
            single_plan_distillation_weight=1.0,
        )
        network.last_single_plan_student_terms = {
            "probe": torch.tensor([2.0], requires_grad=True)
        }
        network.last_actuator_voxel_state = {
            "initial_voxel_coords": torch.zeros((1, 3, 1), dtype=torch.long)
        }
        network.single_plan_distillation_loss_module = _Loss()
        network.last_single_plan_distillation_debug = {}
        network.register_buffer(
            "single_plan_distillation_updates",
            torch.zeros((), dtype=torch.long),
        )
        network.training = True
        loss = network.single_plan_teacher_distillation_loss({"plan_key": "one"})
        loss.backward()
        self.assertEqual(int(network.single_plan_distillation_updates), 1)
        self.assertGreater(
            float(network.last_single_plan_student_terms["probe"].grad.abs().sum()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
