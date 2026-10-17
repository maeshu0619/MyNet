import unittest
import numpy as np
from context_oracle_study import knapsack,pick
from context_oracle_core import OAActual,WindowCodec,unique,costs,stream
import torch


class ContextTests(unittest.TestCase):
    def test_dp_leaves_budget_unused_and_selects_noop(self):
        menus=[[dict(cost=0,total=0.,selected=[]),dict(cost=1,total=-2.,selected=[1])],
               [dict(cost=0,total=0.,selected=[]),dict(cost=2,total=3.,selected=[2])]]
        chosen,audit=knapsack(menus,4,'total')
        self.assertEqual(chosen,[2]);self.assertEqual(audit['used_cost'],2)

    def test_dp_beats_greedy_density(self):
        menus=[[dict(cost=0,total=0.,selected=[]),dict(cost=3,total=4.,selected=[1]),
                dict(cost=2,total=3.,selected=[2])],
               [dict(cost=0,total=0.,selected=[]),dict(cost=2,total=3.,selected=[3])]]
        chosen,audit=knapsack(menus,4,'total')
        self.assertEqual(set(chosen),{2,3});self.assertEqual(audit['dp_value'],6.)

    def test_one_mask_per_row_and_actual_cost(self):
        c=np.array([(1,1),(1,2),(2,1)],dtype=[('row','i4'),('cost','i4')])
        chosen=pick(np.arange(3),c,np.array([3.,5.,1.]),2)
        self.assertEqual(chosen,[0,2])


class ExactContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.oa=OAActual('cpu')
        cls.q=unique(np.random.RandomState(21).randint(0,64,(5000,3)))
        cls.w=WindowCodec(cls.oa,cls.q)

    def test_partial_context_matches_fresh_full_model_and_coder(self):
        w=self.w
        leaves=np.flatnonzero(w.seq[:,-1,1]==w.seq[:,-1,1].max())
        row=next(int(r) for r in leaves if r%w.bptt and (w.seq[r,-1,3:6]>1).all())
        before=int(w.seq[row,-1,0]);after=before^1
        if not after:
            after=before|2
        result=w.evaluate([(row,after)],actual=True)
        changed=w.seq.copy();changed[row,-1,0]=after
        prob=self.oa.probabilities(changed)
        target=(changed[:,-1,0]-1).astype(np.int16)
        actual=stream(prob,target)
        self.assertEqual(actual['stream_sha256'],result['actual_oa']['stream_sha256'])
        expected=float(w.base_cost.sum()-costs(prob,target).sum())
        self.assertAlmostEqual(expected,result['total_cdf_gain'],places=7)
        self.assertAlmostEqual(result['edited_node_cdf_gain']+result['unedited_context_cdf_gain'],expected,places=7)

    def test_noop_exact_stream(self):
        result=self.w.evaluate([],actual=True)
        self.assertEqual(result['actual_oa']['stream_sha256'],self.w.baseline['stream_sha256'])
        self.assertEqual(result['total_cdf_gain'],0.)

    def test_integer_cdf_not_raw_softmax_floor(self):
        prob=np.full((1,255),1./254,dtype=np.float32);prob[0,0]=1e-25
        self.assertAlmostEqual(float(costs(prob,np.array([0]))[0]),16.)


if __name__=='__main__':
    unittest.main()
