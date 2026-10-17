"""Small executable checks for the isolated joint-mask experiment."""
import unittest
import numpy as np
from octree_joint_mask_research import coverage_valid, parts, apply_plan, quality_ok
from octree_joint_topology_research import operation_counts, generate, compose
from scipy.spatial import cKDTree


class JointTests(unittest.TestCase):
    def test_no_exclusive_nn_assignment_is_vacuously_covered(self):
        self.assertTrue(coverage_valid([]).all())

    def test_add_can_rescue_two_prunes(self):
        # Existing slots 0 and 1 separately cover one assigned raw point each.
        # New slot 2 covers both. Prune-only is illegal, joint rewrite is legal.
        covers = np.array([0b0101, 0b0110])
        self.assertEqual(coverage_valid(covers, np.array([0b0011, 0, 0b0100])).tolist(),
                         [True, False, True])
        p = parts(3, 4, np.zeros(3, dtype=np.int64))
        self.assertEqual({k:len(v) for k,v in p.items()}, dict(Adjust=1, Prune=1, Add=0))

    def test_mixed_operation_accounting(self):
        q = np.array([[0,0,0], [0,0,1], [4,0,0], [8,0,0], [8,0,1], [2,1,1]])
        seq = np.zeros((3,4,6), dtype=np.int64)
        seq[:,-1,0] = [3,1,3]
        seq[:,-1,3:6] = [[0,0,0], [4,0,0], [8,0,0]]
        # Two boundary anchors remain; all changes occur in their bbox.
        after, counts = apply_plan(q, seq, [(0,1), (1,3), (2,5)])
        self.assertEqual(counts, dict(Add=1, Prune=1, Adjust=1))
        self.assertEqual(len(after), len(q))

    def test_every_mask_rewrite_cost(self):
        for before in range(1,256):
            for after in range(1,256):
                p = parts(before, after, np.zeros(3, dtype=np.int64))
                count = sum(map(len,p.values()))
                nr,na = bin(before & ~after).count('1'),bin(after & ~before).count('1')
                self.assertEqual(count,max(nr,na))

    def test_duplicate_mask_rejected(self):
        q = np.array([[0,0,0], [1,1,1]])
        seq = np.zeros((1,4,6),dtype=np.int64)
        seq[0,-1,0] = 129
        with self.assertRaisesRegex(AssertionError,'Duplicate'):
            apply_plan(q,seq,[(0,129),(0,129)])

    def test_geometry_guard_checks_all_metrics(self):
        self.assertFalse(quality_ok(dict(chamfer_sq=2., d1_mse=1.),
                                    dict(chamfer_sq=1.9,d1_mse=1.01)))
        self.assertTrue(quality_ok(dict(chamfer_sq=2.,d1_mse=1.),
                                   dict(chamfer_sq=1.9,d1_mse=1.)))

    def test_nonlocal_pair_is_not_one_adjust(self):
        self.assertEqual(operation_counts([[0,0,0]],[[2,0,0]]),
                         dict(Add=1,Prune=1,Adjust=0))
        self.assertEqual(operation_counts([[0,0,0]],[[1,1,1]]),
                         dict(Add=0,Prune=0,Adjust=1))

    def test_cross_parent_joint_rescue_changes_topology(self):
        q=np.array([[0,0,0],[0,3,0],[2,1,1],[4,0,0],[6,3,3],
                    [8,1,1],[10,0,0],[12,2,2],[14,1,1],[16,3,3]])
        raw=q.astype(float)*2.
        raw[3]=[7,0,0]  # banker's rounding maps 7/2 to the original x=4.
        candidates,_=generate(q,raw,np.zeros(3),100)
        rescue=[x for x in candidates if x['rescued_raw_points']]
        self.assertTrue(rescue)
        out,audit=compose(q,rescue,1,'node')
        self.assertEqual(audit['joint_rescue_patches'],1)
        self.assertEqual(audit['counts'],dict(Add=0,Prune=0,Adjust=1))
        self.assertLess(len(np.unique(out//2,axis=0)),len(np.unique(q//2,axis=0)))
        before=cKDTree(q*2.).query(raw)[0]
        after=cKDTree(out*2.).query(raw)[0]
        np.testing.assert_array_less(after,before+1e-9)


if __name__=='__main__':
    unittest.main()
