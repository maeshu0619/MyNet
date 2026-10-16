"""Tests limited to the new standalone experiment; never load training state."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from geometry_micro_edit_probe import (ActualCodec, unique, difference,
                                       phase_add, phase_prune, phase_adjust, local_variants,
                                       combo_variants, quality_guard_variants)


class MicroEditTests(unittest.TestCase):
    def setUp(self):
        self.q = unique(np.random.RandomState(3).randint(0, 9, (120, 3)))

    def test_anchor_does_not_move_original_geometry(self):
        out = phase_add(self.q, (0, 1, 1))
        self.assertEqual(len(difference(self.q, out)), 0)
        added = difference(out, self.q)
        self.assertEqual(len(added), 2)
        np.testing.assert_array_equal(out.min(0), [0, -1, -1])
        np.testing.assert_array_equal(cKDTree(self.q).query(added)[0], [1, 1])

    def test_phase_prune_is_only_pruning(self):
        out = phase_prune(self.q, (0, 1, 1))
        self.assertEqual(len(difference(out, self.q)), 0)
        self.assertTrue(np.all(out[:, 1:] >= 1))

    def test_phase_adjust_is_not_hidden_pruning(self):
        out, count = phase_adjust(self.q, (0, 1, 1))
        self.assertEqual(count, 2)
        self.assertEqual(len(out), len(self.q))
        self.assertEqual(len(difference(self.q, out)), 2)
        self.assertEqual(len(difference(out, self.q)), 2)
        out, count = phase_adjust(np.array([[0, 0, 0]]), (1, 1, 1))
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(out, [[-1,-1,-1]])

    def test_local_accounting_and_budget(self):
        for name, out, count in local_variants(self.q, [.1], np.random.RandomState(21)):
            self.assertEqual(len(difference(self.q, out)), count.get('prune', 0)+count.get('adjust', 0), name)
            self.assertEqual(len(difference(out, self.q)), count.get('add', 0)+count.get('adjust', 0), name)
            self.assertLessEqual(sum(count.values()), round(len(self.q)*.1), name)

    def test_combo_counting(self):
        for name, out, count in combo_variants(self.q, [.1]):
            self.assertEqual(len(difference(self.q, out)), count['prune'], name)
            self.assertEqual(len(difference(out, self.q)), count['add'], name)

    def test_quality_guard_is_jointly_safe(self):
        raw = (self.q*2).astype(np.float64)
        for _, out, count in quality_guard_variants(self.q, raw, np.zeros(3), 2., [.1]):
            self.assertEqual(count['prune'], 0)
            np.testing.assert_array_equal(out, self.q)

    def test_bbox_guard_preserves_extrema(self):
        raw = np.repeat(self.q*2,2,axis=0).astype(np.float64) + np.tile([[1,0,0],[-1,0,0]],(len(self.q),1))
        for _,out,_ in quality_guard_variants(self.q,raw,np.zeros(3),2.,[.1],preserve_bbox=True):
            np.testing.assert_array_equal(out.min(0),self.q.min(0))
            np.testing.assert_array_equal(out.max(0),self.q.max(0))

    def test_actual_teacher_parity_and_decode(self):
        torch.set_num_threads(2)
        codec = ActualCodec(2., 2)
        for out in (self.q, phase_add(self.q, (0, 1, 1))):
            ours = codec.encode(out, verify=True)
            teacher = codec.encoder.encode_tensor(torch.from_numpy((out*2).T.copy()).float())
            self.assertEqual(ours['bits'], teacher['bit'])
            self.assertTrue(ours['decoded_exact'])


if __name__ == '__main__':
    unittest.main()
