import unittest
import numpy as np
from .common import materialize,correlations,geometry,geometry_with_fit


class CommonTest(unittest.TestCase):
    def test_same_plan_and_conflicts(self):
        source=np.array([[0,0,0],[1,0,0]])
        pools={'Add':[dict(remove_coords=[],add_coords=[[2,0,0]])],
               'Prune':[dict(remove_coords=[[1,0,0]],add_coords=[])],
               'Adjust':[dict(remove_coords=[[1,0,0]],add_coords=[[3,0,0]])]}
        points,selected=materialize(source,pools,list(pools),dict.fromkeys(pools,1),dict.fromkeys(pools,[0]))
        self.assertEqual(points.tolist(),[[0,0,0],[2,0,0]])
        self.assertEqual(selected['Adjust'],[])

    def test_rank_ties_are_not_fabricated_correlations(self):
        r=correlations([0,0,0],[1,2,3])
        self.assertIsNone(r['pearson']);self.assertIsNone(r['spearman'])
        self.assertEqual(r['pair_accuracy'],.5);self.assertEqual(r['predicted_tie_fraction'],1.)
        r=correlations([3,2,1],[1,2,3])
        self.assertAlmostEqual(r['pearson'],-1.);self.assertEqual(r['pair_accuracy'],0.)

    def test_quality_identity(self):
        self.assertEqual(geometry([[0,0,0]],[[0,0,0]])['chamfer_sum'],0.)

    def test_real_added_point_fit_on_plane(self):
        from types import SimpleNamespace
        source=np.array([[x,y,0] for x in range(3) for y in range(3)])
        plan=dict(source=source,voxel_step=1.,voxel_offset=np.zeros(3),
                  pools={'Add':[dict(add_coords=[[1,1,1]],remove_coords=[])]})
        edited=np.concatenate([source,[[1,1,1]]])
        measured=geometry_with_fit(plan,edited,{'Add':[0]},SimpleNamespace(geometry_fit_weight=.05))
        self.assertAlmostEqual(measured['fit'],1.,places=6)
        self.assertAlmostEqual(measured['chamfer_sum'],.1,places=6)
        self.assertAlmostEqual(measured['training_G_reconstructed'],.15,places=6)


if __name__=='__main__':unittest.main()
