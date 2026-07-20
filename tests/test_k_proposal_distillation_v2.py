import gzip
import json
import tempfile
import unittest
from types import SimpleNamespace

import torch

from models.utils.loss.k_proposal_distillation import (
    KProposalSetLoss,
    OfflineKProposalTeacherStore,
)
from tools.build_k_proposal_offline_dataset import _candidate_row
from tools.prepare_k_proposal_training_dataset import (
    _explicit_theta,
    _mode_voxel_targets,
)


def _loss_inputs():
    descriptor = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    slot_logits = torch.tensor(
        [[[[2.0, -2.0]], [[-2.0, 2.0]]]]
    ).expand(-1, -1, 3, -1).clone().requires_grad_(True)
    predicted_gain = torch.tensor([[[1.0], [2.0]]], requires_grad=True)
    predicted_geometry = torch.tensor([[[0.2], [0.3]]], requires_grad=True)
    predicted_interaction = torch.tensor([[[0.1], [0.2]]], requires_grad=True)
    uncertainty = torch.tensor([[[0.5], [0.5]]], requires_grad=True)
    target_logits = torch.zeros(1, 2, 3, 2, requires_grad=True)
    direction_logits = torch.zeros(1, 2, 2, 26, 2, requires_grad=True)
    output = {
        "compact_plans": {"descriptor": descriptor},
        "slot_logits": slot_logits,
        "predicted_gain": predicted_gain,
        "predicted_plan_gain": predicted_gain + predicted_interaction,
        "predicted_geometry": predicted_geometry,
        "predicted_interaction": predicted_interaction,
        "uncertainty": uncertainty,
        "slot_target_logits": target_logits,
        "slot_direction_logits": direction_logits,
        "critic_score": predicted_gain,
        "total_ratio": torch.tensor([[[0.001], [0.005]]]),
        "shares": torch.tensor([[[0.5, 0.3, 0.2], [0.2, 0.4, 0.4]]]),
    }
    source_value = torch.zeros(1, 2, 3, 2)
    source_value[0, 0, 0, 0] = 1.0
    source_value[0, 1, 0, 1] = 1.0
    source_mask = torch.zeros_like(source_value, dtype=torch.bool)
    source_mask[:, :, 0] = True
    target_value = torch.zeros(1, 2, 3, 2)
    target_value[0, 0, 1, 0] = 1.0
    target_value[0, 1, 1, 1] = 1.0
    target_mask = torch.zeros_like(target_value, dtype=torch.bool)
    target_mask[:, :, 1] = True
    direction_index = torch.full((1, 2, 2, 2), -1, dtype=torch.long)
    direction_index[0, 0, 1, 0] = 17
    direction_index[0, 1, 1, 1] = 5
    direction_mask = direction_index >= 0
    teacher = {
        "mode_descriptor": descriptor.detach().clone(),
        "actual_gain": torch.tensor([[1.5, 2.5]]),
        "geometry": torch.tensor([[0.1, 0.2]]),
        "interaction": torch.tensor([[0.05, 0.15]]),
        "mode_mask": torch.ones(1, 2, dtype=torch.bool),
        "high_value_mask": torch.tensor([[False, True]]),
        "mode_source_value": source_value,
        "mode_source_mask": source_mask,
        "mode_target_value": target_value,
        "mode_target_value_mask": target_mask,
        "mode_direction_index": direction_index,
        "mode_direction_mask": direction_mask,
        "actual_replay_gain": torch.zeros(1, 2),
        "actual_replay_mask": torch.zeros(1, 2, dtype=torch.bool),
    }
    return output, teacher


class KProposalDistillationV2Test(unittest.TestCase):
    def test_matched_theta_uses_differentiable_classification_heads(self):
        output, teacher = _loss_inputs()
        output["ratio_logits"] = torch.zeros(1, 2, 5, requires_grad=True)
        output["order_logits"] = torch.zeros(1, 2, 6, requires_grad=True)
        output["variant_logits"] = torch.zeros(1, 2, 6, requires_grad=True)
        teacher["theta"] = {
            "ratio_class": torch.tensor([[0, 4]]),
            "total_ratio": torch.tensor([[0.0005, 0.01]]),
            "share": output["shares"].detach().clone(),
            "order_class": torch.tensor([[1, 5]]),
            "variant_class": torch.tensor([[2, 4]]),
            "mask": torch.ones(1, 2, dtype=torch.bool),
        }
        result = KProposalSetLoss()(output, teacher)
        self.assertGreater(float(result["raw"]["theta_supervision"]), 0.0)
        result["total"].backward()
        for key in ("ratio_logits", "order_logits", "variant_logits"):
            self.assertGreater(float(output[key].grad.abs().sum()), 0.0)

    def test_post_valid_target_jaccard_uses_executable_sets(self):
        target_coord = torch.zeros(1, 2, 3, 1, 3, dtype=torch.long)
        accepted = torch.zeros(1, 2, 3, 1, dtype=torch.bool)
        target_coord[0, 0, 1, 0] = torch.tensor([1, 2, 3])
        target_coord[0, 1, 1, 0] = torch.tensor([4, 5, 6])
        accepted[:, :, 1] = True
        teacher_coord = target_coord.clone()
        teacher_mask = accepted.clone()
        cost = KProposalSetLoss._target_pair_cost(
            {"executable_plans": {
                "target_coord": target_coord,
                "accepted_mask": accepted,
            }},
            {
                "mode_target_coord": teacher_coord,
                "mode_target_mask": teacher_mask,
            },
            torch.zeros(1, 2, 1),
        )
        self.assertEqual(float(cost[0, 0, 0]), 0.0)
        self.assertEqual(float(cost[0, 1, 1]), 0.0)
        self.assertEqual(float(cost[0, 0, 1]), 1.0)

    def test_hungarian_supervises_each_slot_without_slot_amax(self):
        output, teacher = _loss_inputs()
        result = KProposalSetLoss()(output, teacher)
        result["total"].backward()
        gradient = output["slot_logits"].grad
        self.assertIsNotNone(gradient)
        self.assertLess(float(gradient[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(gradient[0, 0, 0, 1]), 0.0)
        self.assertGreater(float(gradient[0, 1, 0, 0]), 0.0)
        self.assertLess(float(gradient[0, 1, 0, 1]), 0.0)
        self.assertGreater(float(output["slot_target_logits"].grad.abs().sum()), 0.0)
        self.assertGreater(float(output["slot_direction_logits"].grad.abs().sum()), 0.0)

    def test_actual_k_oracle_is_not_teacher_soft_best(self):
        output, teacher = _loss_inputs()
        without_replay = KProposalSetLoss()(output, teacher)
        self.assertNotIn("oracle_best", without_replay["raw"])
        self.assertIsNone(without_replay["metrics"]["actual_k_oracle"])
        teacher["actual_replay_gain"] = torch.tensor([[0.2, 0.7]])
        teacher["actual_replay_mask"] = torch.tensor([[True, True]])
        with_replay = KProposalSetLoss()(output, teacher)
        self.assertAlmostEqual(
            float(with_replay["metrics"]["actual_k_oracle"]), 0.7, places=6
        )
        self.assertIn("teacher_soft_best", with_replay["raw"])
        self.assertIn("actual_replay_value", with_replay["raw"])

    def test_teacher_critic_value_is_suppressed_for_low_overlap_plan(self):
        close_output, teacher = _loss_inputs()
        close = KProposalSetLoss()(close_output, teacher)
        far_output, far_teacher = _loss_inputs()
        far_output["compact_plans"]["descriptor"] = (
            far_output["compact_plans"]["descriptor"] + 100.0
        )
        far = KProposalSetLoss()(far_output, far_teacher)
        self.assertLess(
            float(far["raw"]["candidate_value"]),
            float(close["raw"]["candidate_value"]),
        )

    def test_ranking_is_computed_inside_each_state(self):
        output, teacher = _loss_inputs()
        for key in (
            "slot_logits", "predicted_geometry", "predicted_interaction",
            "uncertainty", "critic_score", "total_ratio", "shares",
            "slot_target_logits", "slot_direction_logits",
        ):
            output[key] = output[key].detach().expand(2, *output[key].shape[1:]).clone()
        output["compact_plans"]["descriptor"] = output["compact_plans"]["descriptor"].detach().expand(2, -1, -1).clone()
        output["predicted_gain"] = torch.tensor([[[1.0], [2.0]], [[-100.0], [-99.0]]])
        output["predicted_plan_gain"] = output["predicted_gain"]
        output["critic_score"] = output["predicted_gain"]
        for key in (
            "mode_descriptor", "geometry", "interaction", "mode_mask",
            "high_value_mask", "mode_source_value", "mode_source_mask",
            "mode_target_value", "mode_target_value_mask", "mode_direction_index",
            "mode_direction_mask", "actual_replay_gain", "actual_replay_mask",
        ):
            teacher[key] = teacher[key].expand(2, *teacher[key].shape[1:]).clone()
        teacher["actual_gain"] = torch.tensor([[1.0, 2.0], [100.0, 101.0]])
        teacher["state_ids"] = ["state-a", "state-b"]
        result = KProposalSetLoss()(output, teacher)
        expected = torch.nn.functional.softplus(torch.tensor(-1.0))
        self.assertAlmostEqual(float(result["raw"]["ranking"]), float(expected), places=5)

    def test_v2_store_exposes_theta_mode_sources_and_direction(self):
        mode = {
            "descriptor": [0.0, 1.0],
            "actual_gain_percent": 2.0,
            "geometry": {"D1_loss_db": 0.2},
            "interaction_gain_percent": -0.1,
            "actual_rank": 1,
            "high_value": True,
            "explicit_theta": {
                "ratio_class": 2,
                "total_ratio_fraction": 0.0025,
                "share": [0.4, 0.4, 0.2],
                "order_class": 2,
                "variant": 1,
            },
            "voxel_targets": {
                "Prune": {
                    "source_coords": [[1, 2, 3]],
                    "target_coords": [None],
                    "direction_index": [-1],
                    "source_available": [True],
                    "target_available": [False],
                    "direction_available": [False],
                },
                "Add": {
                    "source_coords": [None],
                    "target_coords": [[2, 2, 3]],
                    "direction_index": [-1],
                    "source_available": [False],
                    "target_available": [True],
                    "direction_available": [False],
                },
                "Adjust": {
                    "source_coords": [[4, 5, 6]],
                    "target_coords": [[5, 5, 6]],
                    "direction_index": [17],
                    "source_available": [True],
                    "target_available": [True],
                    "direction_available": [True],
                },
            },
        }
        payload = {
            "schema_version": "mynet_kproposal_mode_dataset_v2",
            "offline_only": True,
            "voxel_target_semantics": "rank_weighted_relative_value_not_causal_gain",
            "split": {"train": "sha|setting"},
            "states": {"sha|setting": {"mode_medoids": [mode]}},
        }
        output = {
            "proposal_count": 1,
            "slot_logits": torch.zeros(1, 1, 3, 3),
            "shortlist_indices": torch.tensor([[0, 1, 2]]),
            "compact_plans": {"descriptor": torch.zeros(1, 1, 2)},
            "target_candidate_coords": torch.tensor(
                [[[2, 5, 8], [2, 5, 8], [3, 6, 8]]]
            ),
        }
        coords = torch.tensor([[[1, 4, 9], [2, 5, 9], [3, 6, 9]]])
        with tempfile.NamedTemporaryFile(suffix=".json.gz") as stream:
            with gzip.open(stream.name, "wt", encoding="utf-8") as compressed:
                json.dump(payload, compressed)
            teacher = OfflineKProposalTeacherStore(stream.name).teacher_for_output(
                "sha|setting", output, coords, split="train"
            )
        self.assertEqual(int(teacher["theta"]["ratio_class"][0, 0]), 2)
        self.assertTrue(bool(teacher["mode_source_mask"][0, 0, 0].any()))
        self.assertFalse(bool(teacher["mode_source_mask"][0, 0, 1].any()))
        self.assertTrue(bool(teacher["mode_direction_mask"][0, 0, 1, 1]))
        self.assertTrue(bool(teacher["mode_target_value_mask"][0, 0, 1].any()))
        self.assertEqual(float(teacher["target_reachable_recall"][0, 0, 1]), 1.0)
        self.assertFalse(bool(teacher["actual_replay_mask"].any()))

    def test_v1_conversion_never_fabricates_add_source(self):
        add = SimpleNamespace(operation="Add", remove_coords=(), add_coords=((2, 3, 4),))
        converted = _candidate_row(add)
        self.assertFalse(converted["source_available"])
        self.assertEqual(converted["source_coords"], [])
        row = {
            "total_ratio_percent": 0.25,
            "shares": {"Prune": 0.4, "Add": 0.4, "Adjust": 0.2},
            "operation_order": "Add>Prune>Adjust",
            "plan_variant": 0,
            "candidates": [{
                "operation": "Add", "remove_coords": (), "add_coords": ((2, 3, 4),)
            }],
        }
        theta = _explicit_theta(row)
        targets = _mode_voxel_targets(row)
        self.assertEqual(theta["ratio_class"], 2)
        self.assertEqual(targets["Add"]["source_coords"], [None])
        self.assertFalse(targets["Add"]["source_available"][0])


if __name__ == "__main__":
    unittest.main()
