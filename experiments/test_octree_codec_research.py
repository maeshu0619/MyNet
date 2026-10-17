import itertools
import unittest
import numpy as np
import torch
from scipy.spatial import cKDTree
from octree_codec_research import OAActual, safe_prune_order, quality, unique


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.oa=OAActual('cpu')
        cls.q=unique(np.random.RandomState(14).randint(0,16,(900,3)))

    def test_actual_teacher_arithmetic_parity(self):
        ours=self.oa.encode(self.q,roundtrip=True)
        native=self.oa.teacher.encode_bits(torch.from_numpy(self.q.T.copy()).float())
        self.assertEqual(ours['bits'],native['bit'])

    def test_causal_target_probability_no_own_symbol_leak(self):
        seq=self.oa.sequence(self.q)
        before=self.oa.probabilities(seq)
        changed=seq.copy()
        row=len(seq)//2
        changed[row,-1,0]=1 if seq[row,-1,0]!=1 else 2
        after=self.oa.probabilities(changed)
        np.testing.assert_array_equal(before[:row+1],after[:row+1])

    def test_strict_pruning_keeps_every_input_nearest_distance(self):
        q=np.array(list(itertools.product(range(2),range(8),range(8))),dtype=np.int64)
        raw=np.array([[1,y*2,z*2] for y,z in itertools.product(range(8),repeat=2)])
        order,audit=safe_prune_order(q,raw,np.zeros(3),2.)
        self.assertGreater(audit['safe_deletion_capacity'],0)
        mask=np.ones(len(q),dtype=bool)
        mask[order]=False
        before=cKDTree(q*2).query(raw)[0]
        after=cKDTree(q[mask]*2).query(raw)[0]
        np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(q[mask].min(0),q.min(0))
        np.testing.assert_array_equal(q[mask].max(0),q.max(0))

    def test_all_48_transforms_have_exact_inverse(self):
        q=self.q-self.q.min(0)
        for perm in itertools.permutations(range(3)):
            for sign in itertools.product((1,-1),repeat=3):
                signed=q[:,perm]*np.array(sign)
                coded=unique(signed-signed.min(0))
                restored=np.where(np.array(sign)>0,coded,coded.max(0)-coded)
                restored=restored[:,np.argsort(perm)]
                np.testing.assert_array_equal(unique(restored),q)

    def test_integer_cdf_cost_matches_arithmetic_length(self):
        from octree_window_research import integer_cdf_cost
        import numpyAc
        rng=np.random.RandomState(42)
        probs=rng.uniform(.1,1.,(128,255)).astype(np.float32)
        probs/=probs.sum(1,keepdims=True)
        symbols=rng.randint(0,255,128).astype(np.int16)
        _,bits=numpyAc.arithmeticCoding().encode(probs,symbols)
        estimate=integer_cdf_cost(probs,symbols)
        self.assertLess(abs(bits-estimate),16.)

    def test_fixed_window_probability_independence(self):
        q=unique(np.random.RandomState(21).randint(0,64,(5000,3)))
        seq=self.oa.sequence(q)
        self.assertGreater(len(seq),2048)
        before=self.oa.probabilities(seq)
        row=1124
        changed=seq.copy()
        changed[row,-1,0]=1 if seq[row,-1,0]!=1 else 2
        after=self.oa.probabilities(changed)
        outside=np.ones(len(seq),dtype=bool)
        outside[1025:2049]=False
        np.testing.assert_array_equal(before[outside],after[outside])


if __name__=='__main__':
    unittest.main()
