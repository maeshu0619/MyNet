import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch

from models.modules.exact_teacher_theta_adapter import (
    CatalogThetaSelector,
    ExactTeacherThetaCatalog,
    MISSING_COORDINATE,
)
from models.modules.executable_voxel_plan import apply_selected_executable_plan


class ExactTeacherThetaAdapterTest(unittest.TestCase):
    def _dataset(self, root):
        source = root / "run_rows.json"
        source_row = {
            "input_file": str(root / "input.ply"),
            "setting_id": "setting",
            "pattern_key": "pattern",
            "plan_key": "plan",
            "actual_saved_percent": 4.0,
            "operation_order": "Add>Prune>Adjust",
            "variant_index": 2,
            "screening_score": 3.0,
            "screening_rank_within_total": 1,
            "selection_reason": "screening_top1",
        }
        source.write_text(json.dumps({"actual_rows": [source_row]}), encoding="utf-8")
        record = {
            "state_key": {
                "input_file": str(root / "input.ply"),
                "input_sha256": "a" * 64,
                "setting_id": "setting",
                "scale_m": 8,
                "scale_ae": 0,
                "scale_sr": 2,
                "voxel_size": 1.0,
                "pos_quantscale": 1,
                "native_resolution": 1023,
            },
            "pattern_key": "pattern",
            "plan_key": "plan",
            "requested_counts": {"Add": 1, "Prune": 1, "Adjust": 1},
            "operation_counts": {"Add": 1, "Prune": 1, "Adjust": 1},
            "total_ratio_percent": 0.25,
            "shares": {"Add": 0.4, "Prune": 0.4, "Adjust": 0.2},
            "actual_gain_percent": 4.0,
            "geometry": {"D1_loss_db": 0.1, "D2_loss_db": 0.1},
            "estimated_gain_percent": 3.0,
            "final_voxel_hash": "hash",
            "candidates": [
                {"operation": "Add", "remove_coords": [], "add_coords": [[3, 0, 0]], "heuristic_score": 3.0},
                {"operation": "Prune", "remove_coords": [[0, 0, 0]], "add_coords": [], "heuristic_score": 2.0},
                {"operation": "Adjust", "remove_coords": [[1, 0, 0]], "add_coords": [[1, 1, 0]], "heuristic_score": 1.0},
            ],
        }
        dataset = root / "dataset.json.gz"
        with gzip.open(str(dataset), "wt", encoding="utf-8") as stream:
            json.dump({
                "schema_version": "mynet_kproposal_actual_plan_dataset_v1",
                "contains_virtual_actual_labels": False,
                "source_run_rows": str(source),
                "records": [record],
            }, stream)
        return dataset

    def test_exact_members_preserve_missing_add_source_and_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = ExactTeacherThetaCatalog.from_actual_plan_datasets(
                [str(self._dataset(Path(directory)))]
            )
            state_id = catalog.state_ids[0]
            theta_id = catalog.records(state_id)[0].theta_id
            coords = torch.tensor(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
            generated = catalog.generate(state_id, [theta_id], coords, debug_hash=True)
            plan = generated.executable
            add_mask = plan.accepted_mask[0, 0, 1]
            self.assertEqual(generated.missing_add_source_count, [1])
            self.assertTrue(torch.all(plan.source_coord[0, 0, 1][add_mask] == MISSING_COORDINATE))
            self.assertEqual(plan.target_coord[0, 0, 1][add_mask].tolist(), [[3, 0, 0]])
            edited, valid = apply_selected_executable_plan(
                coords.transpose(0, 1).unsqueeze(0), plan, torch.tensor((0,))
            )
            result = {tuple(value) for value in edited[0, :, valid[0]].transpose(0, 1).tolist()}
            self.assertEqual(result, {(1, 1, 0), (2, 0, 0), (3, 0, 0)})

    def test_selector_is_deterministic(self):
        torch.manual_seed(1)
        selector = CatalogThetaSelector(4, 6, hidden_dim=8).eval()
        state = torch.randn(1, 4)
        theta = torch.randn(5, 6)
        self.assertTrue(torch.equal(selector(state, theta), selector(state, theta)))


if __name__ == "__main__":
    unittest.main()
