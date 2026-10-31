from collections import OrderedDict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


class AutonomousArrayParityTest(unittest.TestCase):
    def tree_equal(self,a,b):
        if torch.is_tensor(a):
            self.assertEqual(a.dtype,b.dtype)
            self.assertTrue(torch.equal(a,b),(a.shape,b.shape))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.tree_equal(a[key],b[key])
        else:self.assertEqual(a,b)

    def test_guidance_matches_reference_including_all_add_neighbours_and_invalid_candidates(self):
        from models.modules import heuristic_guidance as hg
        offsets=hg._NEIGHBOR_OFFSETS_26
        coords=torch.tensor(list(offsets)+[(4,4,4),(4,4,5)],dtype=torch.long)
        coords=coords[torch.randperm(len(coords),generator=torch.Generator().manual_seed(7))].T.unsqueeze(0)
        pools={op:[] for op in ('Add','Prune','Adjust')}
        entries={'Add':[(None,(0,0,0)),(None,(4,4,4)),(None,(90,90,90))],
                 'Prune':[((4,4,4),None),((99,99,99),None)],
                 'Adjust':[((4,4,4),(4,4,5)),((4,4,4),(4,5,4)),
                           ((4,4,4),(6,4,4)),((99,99,99),(99,100,99))]}
        for op,actions in entries.items():
            for i,(r,a) in enumerate(actions):
                pools[op].append(dict(candidate_id=op+str(i),remove_coords=[r] if r else [],
                    add_coords=[a] if a else [],pool_rank=i,rank_score=0.,optimistic_gain_bits=i-2.,geometry_cost=.2))
        exact=dict(source='mynet_fast_unordered_safe_pool_v1',autonomous_search=True,
            operation_edit_units=pools,cache_signature='unit',total_ratio=.045,
            operation_shares=dict(Add=1/3,Prune=1/3,Adjust=1/3))
        args=SimpleNamespace(test_route=True,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online')
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            local=coords.to(device);like=torch.zeros(1,1,local.shape[-1],device=device)
            structure=dict(global_voxel_coords=local)
            def mapped(legacy):
                with mock.patch.object(hg,'_EXACT_GUIDANCE_CACHE',OrderedDict()),\
                     mock.patch.object(hg,'_EXACT_GUIDANCE_CPU_CACHE',OrderedDict()),\
                     mock.patch.object(hg,'_EXACT_GUIDANCE_CPU_CACHE_BYTES',0),\
                     mock.patch.object(hg,'_load_exact_guidance_disk',return_value=None),\
                     mock.patch.object(hg,'_store_exact_guidance_disk'):
                    if legacy:
                        with mock.patch('models.utils.pointcloud.autonomous_guidance_mapping.map_single_voxel_candidates',return_value=None):
                            return hg._exact_den6_guidance(structure,args,like,exact)
                    return hg._exact_den6_guidance(structure,args,like,exact)
            expected=mapped(True);actual=mapped(False)
            self.tree_equal(expected,actual)
            add=actual['candidate_tensor_map']['Add']
            self.assertEqual(int((add['pair_candidate_index']==0).sum()),26)

    def test_non_autonomous_never_calls_fast_mapper(self):
        from models.modules import heuristic_guidance as hg
        args=SimpleNamespace(test_route=False,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online')
        exact=dict(source='mynet_fast_unordered_safe_pool_v1',autonomous_search=False,
            operation_edit_units={op:[] for op in ('Add','Prune','Adjust')},
            operation_shares=dict(Add=1,Prune=1,Adjust=1),cache_signature='false-unit')
        like=torch.zeros(1,1,1)
        with mock.patch.object(hg,'_EXACT_GUIDANCE_CACHE',OrderedDict()),\
             mock.patch.object(hg,'_EXACT_GUIDANCE_CPU_CACHE',OrderedDict()),\
             mock.patch.object(hg,'_EXACT_GUIDANCE_CPU_CACHE_BYTES',0),\
             mock.patch.object(hg,'_load_exact_guidance_disk',return_value=None),\
             mock.patch.object(hg,'_store_exact_guidance_disk'),\
             mock.patch('models.utils.pointcloud.autonomous_guidance_mapping.map_single_voxel_candidates',side_effect=AssertionError('False entered fast mapper')):
            hg._exact_den6_guidance(dict(global_voxel_coords=torch.zeros(1,3,1,dtype=torch.long)),args,like,exact)

    def test_context_candidates_and_core_cache_are_exact_for_cold_and_warm(self):
        from models.utils.pointcloud.autonomous_context_proposals import canonical_voxel_sha,_context_proposals_reference
        from models.utils.pointcloud.autonomous_context_arrays import context_proposals_arrays
        source=np.array([[x,y,z] for x in range(9) for y in range(9) for z in range(9) if (x+y+z)%3==0],dtype=np.int64)
        cells=np.array([[x,y,z] for x in range(5) for y in range(5) for z in range(5)],dtype=np.int64)
        occupied=set(map(tuple,source//2))
        probability=np.linspace(1/65536,65535/65536,len(cells))
        for budget,seed in ((1,0),(17,21),(60,7),(17,-2)):
            with tempfile.TemporaryDirectory() as directory:
                a,b=Path(directory)/'old.npz',Path(directory)/'new.npz'
                for path in (a,b):
                    np.savez(path,coords=cells,probability=probability,
                        occupied=np.array([tuple(v) in occupied for v in cells]),
                        metadata=json.dumps(dict(scale=2,input_voxel_sha256=canonical_voxel_sha(source))))
                random={op:[] for op in ('Add','Prune','Adjust')}
                random['Prune']=[dict(remove_coords=[source[0].tolist()],add_coords=[],geometry_cost=.7,extra='retained')]*3
                random['Add']=[dict(remove_coords=[],add_coords=[[0,0,1]],geometry_cost=.3)]*3
                random['Adjust']=[dict(remove_coords=[source[0].tolist()],add_coords=[[0,0,1]])]*3
                for warm in (False,True):
                    expected=_context_proposals_reference(source,a,budget,seed,random)
                    actual=context_proposals_arrays(source,b,budget,seed,random)
                    self.assertEqual(expected,actual,(budget,seed,warm))
                with np.load(str(a)+'.proposals_v1_%d_%d.npz'%(budget,seed&0x7fffffff)) as x,\
                     np.load(str(b)+'.proposals_v1_%d_%d.npz'%(budget,seed&0x7fffffff)) as y:
                    for op in random:self.assertTrue(np.array_equal(x[op],y[op]))

    def test_integer_row_index_and_dedup_handle_overflow_and_outside_keys(self):
        from models.utils.pointcloud.autonomous_context_arrays import RowIndex,first_rows
        for rows in (np.array([[0,0,0],[2,3,4]]),np.array([[-2**62,0,0],[2**62,1,2]])):
            query=np.concatenate((rows,[[99,-99,0]]))
            self.assertTrue(np.array_equal(RowIndex(rows).find(query),[0,1,-1]))
            repeated=np.concatenate((rows,rows))
            self.assertTrue(np.array_equal(first_rows(repeated),[0,1]))
        self.assertEqual(RowIndex(np.empty((0,3),dtype=np.int64)).find([[0,0,0]]).tolist(),[-1])

    def test_typed_spatial_input_preserves_list_candidate_values(self):
        from models.utils.pointcloud.autonomous_context_proposals import canonical_voxel_sha
        from models.utils.pointcloud.autonomous_context_arrays import SpatialCandidates,context_proposals_arrays
        source=np.array([[x,y,z] for x in range(7) for y in range(7) for z in range(7) if (x+y+z)%3==0])
        cells=np.array([[x,y,z] for x in range(4) for y in range(4) for z in range(4)])
        occupied=set(map(tuple,source//2))
        spatial={op:[] for op in ('Add','Prune','Adjust')}
        spatial['Add']=[dict(remove_coords=[],add_coords=[[0,0,1]],geometry_cost=7/26)]
        spatial['Prune']=[dict(remove_coords=[source[0].tolist()],add_coords=[],geometry_cost=7/26)]
        spatial['Adjust']=[dict(remove_coords=[source[0].tolist()],add_coords=[[0,0,1]],geometry_cost=7/26)]
        typed={op:SpatialCandidates(np.array([(c['remove_coords'] or [[0,0,0]])[0]+
                    (c['add_coords'] or [[0,0,0]])[0] for c in pool]),[c['geometry_cost'] for c in pool])
               for op,pool in spatial.items()}
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/n for n in ('list.npz','arrays.npz')]
            for path in paths:
                np.savez(path,coords=cells,probability=np.linspace(.01,.99,len(cells)),
                    occupied=np.array([tuple(v) in occupied for v in cells]),
                    metadata=json.dumps(dict(scale=2,input_voxel_sha256=canonical_voxel_sha(source))))
            self.assertEqual(context_proposals_arrays(source,paths[0],50,11,spatial),
                             context_proposals_arrays(source,paths[1],50,11,typed))
        self.assertTrue(np.array_equal(np.arange(27,dtype=np.float64)/26.,[float(n/26.) for n in range(27)]))


if __name__=='__main__':unittest.main()
