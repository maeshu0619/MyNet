from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from models.utils.loss.geometry import GeometryLossMixin


def _torch_chamfer(a, b):
    squared = torch.cdist(a, b).square()
    dist_a, index_a = squared.min(dim=2)
    dist_b, index_b = squared.min(dim=1)
    return dist_a, dist_b, index_a, index_b


class _Geometry(GeometryLossMixin):
    pass


class ExactSparseGeometryTest(unittest.TestCase):
    def test_sparse_edit_chamfer_equals_full_chamfer(self):
        initial_rows = torch.tensor([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [3, 0, 0], [4, 0, 0], [5, 0, 0],
        ], dtype=torch.long)
        # 2を削除し、4を隣へAdjustし、新しい点を1つAddする。
        final_rows = torch.tensor([
            [0, 0, 0], [1, 0, 0], [3, 0, 0],
            [4, 1, 0], [5, 0, 0], [2, 1, 0],
        ], dtype=torch.long)
        gt = initial_rows.float().transpose(0, 1).unsqueeze(0)
        gen = final_rows.float().transpose(0, 1).unsqueeze(0)
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            _last_actuator_voxel_state={
                "voxel_edit_state_enabled": True,
                "initial_voxel_coords": initial_rows.transpose(0, 1).unsqueeze(0),
                "final_voxel_coords": final_rows.transpose(0, 1).unsqueeze(0),
                "final_voxel_valid_mask": torch.ones((1, len(final_rows)), dtype=torch.bool),
                "voxel_restore_meta": {
                    "effective_qs_tensor": torch.ones((1, 1, 1)),
                    "global_offset_tensor": torch.zeros((1, 3, 1)),
                },
            },
        )
        full_dist = torch.cdist(
            gen.transpose(1, 2), gt.transpose(1, 2)
        ).square()
        expected = full_dist.min(dim=2).values.mean() + full_dist.min(dim=1).values.mean()
        with mock.patch("models.utils.loss.geometry.chamfer_dist", _torch_chamfer):
            sparse = _Geometry()._exact_sparse_edit_chamfer(
                args, gen, gt, final_w_f=None
            )
        self.assertIsNotNone(sparse)
        self.assertTrue(torch.allclose(sparse["hard"], expected, atol=1e-7, rtol=0.0))
        self.assertEqual(sparse["removed_count"], 2)
        self.assertEqual(sparse["added_count"], 2)


if __name__ == "__main__":
    unittest.main()
