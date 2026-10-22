import unittest
import numpy as np

from .analyze_native import counterfactual_nll,decompose


def example(coords,probability,symbols):
    p=np.asarray(probability,dtype=np.float32)
    symbols=np.asarray(symbols,dtype=np.int8)
    nll=counterfactual_nll(p,symbols)
    return dict(coords=np.asarray(coords,dtype=np.int64),probability=p,occupancy=symbols,
                nll=nll,cdf_probability=2.**(-nll))


class AccountingTests(unittest.TestCase):
    def test_equal_cloud_has_zero_decomposition(self):
        a=example([[0,0,0],[1,0,0]],[.2,.9],[0,1])
        result=decompose(a,a)
        self.assertEqual(result['total_NLL_change'],0.)
        self.assertEqual(result['context_change_unchanged_symbols_NLL'],0.)

    def test_symbol_context_and_topology_accounting(self):
        a=example([[0,0,0],[1,0,0],[2,0,0]],[.01,.8,.3],[1,1,0])
        b=example([[0,0,0],[1,0,0],[3,0,0]],[.02,.9,.7],[0,1,1])
        r=decompose(a,b)
        self.assertAlmostEqual(r['decomposition_roundoff'],0.,places=12)
        self.assertEqual(r['removed_candidates'],1);self.assertEqual(r['added_candidates'],1)
        self.assertEqual(r['changed_common_symbols'],1)
        self.assertAlmostEqual(r['symmetric_count_term']+r['symmetric_mean_cost_term'],r['total_NLL_change'])

    def test_finite_probabilities_at_float_endpoints(self):
        result=counterfactual_nll(np.array([0.,1.,.5],np.float32),np.array([1,0,1]))
        self.assertTrue(np.array_equal(result,np.array([16.,16.,1.])))


if __name__=='__main__':unittest.main()
