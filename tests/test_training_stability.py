import os
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory

import torch

from models.utils.training.actual_compression_guard import (
    apply_actual_compression_guard,
    update_network_autonomy_from_guard,
)
from models.utils.training.compression_primary_loss import (
    build_compression_primary_loss,
)
from models.utils.training.lr_control import step_scheduler_with_floor


class _Writer:
    def __init__(self):
        self.lines = []

    def write(self, value):
        self.lines.append(str(value))


class _Loss:
    compression_surrogate = None
    surrogate_optimizer = None


class TrainingStabilityTest(unittest.TestCase):
    def _guard_args(self):
        return SimpleNamespace(
            compression_loss_backend="sparsepcgc_surrogate",
            actual_compression_guard=True,
            actual_guard_require_fixed_validation=True,
            actual_guard_require_full_state_restore=True,
            checkpoint_full_cloud_min_count=1,
            actual_guard_min_fresh=1,
            actual_guard_improvement_epsilon=1e-6,
            actual_guard_tolerance=0.01,
            actual_guard_patience=1,
            actual_guard_restore_best=True,
            actual_guard_decay_lr=False,
            actual_guard_lr_decay=0.5,
            min_main_lr=1e-6,
            min_surrogate_lr=1e-6,
            _global_train_step=3,
            _sparsepcgc_full_cloud_sequence_baseline_memory={"seq": {"baseline": -3.0}},
        )

    def test_guard_uses_fixed_validation_only(self):
        args = self._guard_args()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        event = apply_actual_compression_guard(
            args=args,
            model=model,
            loss=_Loss(),
            optimizer=optimizer,
            writer=_Writer(),
            guard_state={},
            checkpoint_metrics={
                "checkpoint_eligible": True,
                "checkpoint_actual_source": "fresh",
                "checkpoint_actual_delta": -3.0,
                "checkpoint_actual_count": 1,
            },
            ckpt_dir=".",
            episode=0,
        )
        self.assertEqual(event["action"], "skipped")
        self.assertEqual(event["reason"], "fixed_full_cloud_validation_required")

    def test_guard_restores_optimizer_scheduler_scaler_and_runtime_memory(self):
        torch.manual_seed(4)
        args = self._guard_args()
        writer = _Writer()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        mapping = {"state": 3}
        runtime_state = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "mutable_mappings": {"network_k_state_visit_counts": mapping},
        }

        # Adam momentを作った同じ時点をbestとして保存する。
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        saved_weight = model.weight.detach().clone()
        saved_lr = optimizer.param_groups[0]["lr"]

        with TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "0.pth")
            torch.save(model.state_dict(), model_path)
            guard_state = {}
            best_event = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=writer,
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -3.0,
                    "checkpoint_actual_count": 1,
                },
                ckpt_dir=directory,
                episode=0,
                runtime_state=runtime_state,
            )
            self.assertTrue(best_event["training_state_saved"])

            with torch.no_grad():
                model.weight.add_(10.0)
            optimizer.param_groups[0]["lr"] = 0.123
            scheduler.step()
            mapping["state"] = 99
            args._sparsepcgc_full_cloud_sequence_baseline_memory = {
                "seq": {"baseline": 99.0}
            }

            rollback = apply_actual_compression_guard(
                args=args,
                model=model,
                loss=_Loss(),
                optimizer=optimizer,
                writer=writer,
                guard_state=guard_state,
                checkpoint_metrics={
                    "checkpoint_eligible": True,
                    "checkpoint_actual_source": "full_cloud",
                    "checkpoint_actual_delta": -2.0,
                    "checkpoint_actual_count": 1,
                },
                ckpt_dir=directory,
                episode=1,
                runtime_state=runtime_state,
            )
            self.assertEqual(rollback["action"], "rollback")
            self.assertTrue(rollback["training_state_restored"])
            self.assertTrue(torch.equal(model.weight.detach(), saved_weight))
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], saved_lr)
            self.assertEqual(mapping["state"], 3)
            self.assertEqual(
                args._sparsepcgc_full_cloud_sequence_baseline_memory["seq"]["baseline"],
                -3.0,
            )

    def test_disabled_scheduler_does_not_decay_emulator_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        args = SimpleNamespace(lr_scheduler_enabled=False, min_main_lr=1e-6)
        event = step_scheduler_with_floor(scheduler, optimizer, args)
        self.assertFalse(event["scheduler_stepped"])
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_network_autonomy_grows_only_on_fixed_validation_new_best(self):
        args = SimpleNamespace(
            heuristic_guidance_network_residual_weight=0.05,
            heuristic_guidance_network_residual_weight_max=0.10,
            heuristic_guidance_network_residual_weight_increment=0.025,
        )
        first = update_network_autonomy_from_guard(args, {"action": "new_best"})
        self.assertAlmostEqual(first["current"], 0.075)
        unchanged = update_network_autonomy_from_guard(
            args, {"action": "within_tolerance"}
        )
        self.assertAlmostEqual(unchanged["current"], 0.075)
        second = update_network_autonomy_from_guard(args, {"action": "new_best"})
        self.assertAlmostEqual(second["current"], 0.10)
        capped = update_network_autonomy_from_guard(args, {"action": "new_best"})
        self.assertAlmostEqual(capped["current"], 0.10)

    def test_sparsepcgc_geometry_penalty_is_continuous(self):
        args = SimpleNamespace(
            compression_loss_backend="sparsepcgc_surrogate",
            w_com=1.0,
            cp_tau_geom=0.0,
            cp_lambda_geom=100.0,
            compression_primary_aux_target_ratio=0.25,
            compression_primary_aux_balance_min_scale=0.0,
            compression_primary_aux_balance_max_scale=1.0,
        )
        main = torch.tensor(-4.0, requires_grad=True)
        geom = torch.tensor(0.003, requires_grad=True)
        total, _, debug = build_compression_primary_loss(
            args,
            terms={"main": main},
            L_com=main,
            L_geom=geom,
            L_actuator=torch.tensor(0.0),
            global_train_step=0,
            stage_factors={},
        )
        self.assertAlmostEqual(debug["cp_P_geom"], 0.003, places=7)
        self.assertAlmostEqual(debug["cp_geom_block_raw"], 0.3, places=6)
        total.backward()
        self.assertGreater(float(geom.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
