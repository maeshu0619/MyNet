import copy
import hashlib
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from models.utils.loss.autonomous_shadow_metrics import adjust_pair_recall
from models.utils.loss.autonomous_native_loader import make_exact_loader
from models.utils.pointcloud.autonomous_context_proposals import (
    canonical_voxel_sha, _CANONICAL_SHA_CACHE, _CANONICAL_SHA_CACHE_LIMIT)


class ExactMetricsTest(unittest.TestCase):
    def test_pair_sets_match_reference_with_duplicates_negative_and_int64_extremes(self):
        rng = np.random.RandomState(41)
        for n in (0, 1, 17, 1200):
            a = rng.randint(-1024, 1024, (n, 6)).astype(np.int64)
            a = np.concatenate((a, a[:7]))
            if n:
                a[0] = [np.iinfo(np.int64).min, 0, 1, np.iinfo(np.int64).max, 1, 0]
            b = np.concatenate((a[:n//2], a[:3], rng.randint(-9, 9, (30, 6))))
            candidates = [dict(remove_coords=[row[:3].tolist()], add_coords=[row[3:].tolist()]) for row in a]
            candidates += [dict(remove_coords=[], add_coords=[[1,2,3]])]
            expected = set(map(tuple, a.tolist()))
            predicted = set(map(tuple, b.tolist()))
            reference = len(expected & predicted)/len(expected) if expected else None
            self.assertEqual(adjust_pair_recall(candidates, b[:,:3], b[:,3:]), reference)
            self.assertEqual(adjust_pair_recall(candidates, np.empty((0,3)), np.empty((0,3))),
                             0. if expected else None)

    def test_distillation_loss_metrics_and_all_gradients_are_unchanged(self):
        from models.modules.single_plan_student import SinglePlanStudentPolicy
        from models.utils.loss.single_plan_distillation import SinglePlanDistillationLoss
        torch.manual_seed(2)
        coords = torch.stack((torch.arange(128)*3, torch.zeros(128,dtype=torch.long),
                              torch.zeros(128,dtype=torch.long))).view(1,3,128)
        features, fixed = torch.randn(1,8,128), torch.randn(1,6,128)
        args = SimpleNamespace(sparsepcgc_psnr_resolution=1023, sparsepcgc_scale_ae=0,
            sparsepcgc_scale_sr=2, sparsepcgc_scale_m=8, sparsepcgc_voxel_size=1., sparsepcgc_pos_quantscale=1,
            sparsepcgc_native_bit_depth=10, _global_train_step=0, network_only_exploration_anneal_steps=200,
            network_only_action_exploration_floor=0., network_only_where_gumbel_scale=0.,single_plan_debug_hash=False)
        model = SinglePlanStudentPolicy(8,hidden_dim=16).train()
        other = copy.deepcopy(model)
        teacher = dict(total_ratio_percent=.25,shares=dict(Prune=.4,Add=.4,Adjust=.2),
            operation_order='Prune>Add>Adjust',actual_gain_percent=3.,geometry=dict(D1_loss_db=.1,D2_loss_db=.2),
            candidates=[dict(operation='Prune',remove_coords=[[0,0,0]],add_coords=[]),
                        dict(operation='Add',remove_coords=[],add_coords=[[1,0,0]]),
                        dict(operation='Adjust',remove_coords=[[3,0,0]],add_coords=[[4,0,0]])])
        outputs=[]
        for network, fast in ((model,False),(other,True)):
            current_teacher=copy.deepcopy(teacher)
            if fast:
                from models.utils.cache.autonomous_candidate_layout import immutable_candidate_coordinates
                immutable_candidate_coordinates({'selected':current_teacher['candidates']})
            terms = network(features,coords,args,training=False,fixed_features=fixed)
            value, metrics = SinglePlanDistillationLoss()(terms,coords,current_teacher,compact_metrics=fast)
            value.backward()
            outputs.append((value.detach(),metrics))
        self.assertTrue(torch.equal(outputs[0][0],outputs[1][0]))
        for key,value in outputs[0][1].items():
            other_value=outputs[1][1][key]
            if isinstance(value,float) and math.isnan(value):self.assertTrue(math.isnan(other_value))
            else:self.assertEqual(value,other_value,key)
        for a,b in zip(model.parameters(),other.parameters()):
            self.assertEqual(a.grad is None,b.grad is None)
            if a.grad is not None:self.assertTrue(torch.equal(a.grad,b.grad))

    def test_coordinate_layout_preserves_json_and_untracks_candidate_dicts(self):
        import gc,json,pickle
        from models.utils.cache.autonomous_candidate_layout import immutable_candidate_coordinates
        pools={'Add':[dict(candidate_id='a',remove_coords=[],add_coords=[[1,2,3]],score=.125)],
               'Adjust':[dict(candidate_id='b',remove_coords=[[1,2,3]],add_coords=[[2,2,3]],score=-.25)]}
        original=json.dumps(pools,sort_keys=True,separators=(',',':'))
        immutable_candidate_coordinates(pools)
        self.assertEqual(json.dumps(pools,sort_keys=True,separators=(',',':')),original)
        restored=pickle.loads(pickle.dumps(pools,protocol=4))
        # Match restoration: young collection first untracks coordinate
        # tuples; a later full collection can then untrack their dictionaries.
        gc.collect(1)
        gc.collect(2)
        for pool in restored.values():
            self.assertIsInstance(pool,list)
            for candidate in pool:
                self.assertFalse(gc.is_tracked(candidate))
                self.assertIsInstance(candidate['add_coords'],tuple)
        self.assertEqual(json.dumps(restored,sort_keys=True,separators=(',',':')),original)


class FingerprintTest(unittest.TestCase):
    def test_canonical_hash_tracks_mutation_order_views_and_bounds_cache_size(self):
        _CANONICAL_SHA_CACHE.clear()
        rng=np.random.RandomState(7)
        rows=rng.randint(-500,500,(1000,3)).astype(np.int64)
        for data in (rows,rows[::-1],rows[::2],np.r_[rows,rows[:20]],np.empty((0,3),dtype=np.int64)):
            expected=hashlib.sha256(np.unique(data,axis=0).tobytes()).hexdigest()
            self.assertEqual(canonical_voxel_sha(data),expected)
            self.assertEqual(canonical_voxel_sha(data.copy()),expected)
        rows[0]=[999,999,999]
        self.assertEqual(canonical_voxel_sha(rows),hashlib.sha256(np.unique(rows,axis=0).tobytes()).hexdigest())
        for i in range(300):canonical_voxel_sha(np.array([[i,0,0]],dtype=np.int64))
        self.assertEqual(len(_CANONICAL_SHA_CACHE),_CANONICAL_SHA_CACHE_LIMIT)
        self.assertTrue(all(isinstance(k,bytes) and len(k)==32 and isinstance(v,str)
                            for k,v in _CANONICAL_SHA_CACHE.items()))


class IdentityQuantizationTest(unittest.TestCase):
    def test_only_exact_identity_case_bypasses_reference(self):
        def reference(points,precision=.001,quant_mode='round',return_offset=False):
            value=points.astype('float')/float(precision)
            quant=(np.round(value) if quant_mode=='round' else np.floor(value)).astype('int')
            return (quant,value-quant) if return_offset else quant
        spy=Mock(side_effect=reference)
        env=dict(np=np,quantize_precision=spy)
        exec('def load(points, precision, mode="round", offset=False):\n'
             ' return quantize_precision(points, precision=precision, quant_mode=mode, return_offset=offset)',env)
        wrapped=make_exact_loader(env['load'])
        for dtype in (np.int32,np.uint32,np.int16,np.float32,np.int64):
            rows=np.array([[0,1,2],[100,2**14,10]],dtype=dtype)
            for precision,mode,offset in ((1.,'round',False),(1.,'floor',False),
                                          (2.,'round',False),(1.,'round',True)):
                spy.reset_mock()
                actual=wrapped(rows,precision,mode,offset)
                expected=reference(rows,precision,mode,offset)
                if offset:
                    self.assertTrue(all(np.array_equal(a,b) for a,b in zip(actual,expected)))
                else:self.assertTrue(np.array_equal(actual,expected))
                identity=(np.dtype(dtype).kind in 'iu' and np.dtype(dtype).itemsize<=4
                          and precision==1 and not offset)
                self.assertEqual(spy.call_count,int(not identity))


class StaticAdmissionTest(unittest.TestCase):
    def test_size_matches_actual_clone_alias_and_view_behavior(self):
        from models.network import Network
        from models.utils.cache.autonomous_static_admission import static_clone_bytes
        a=torch.arange(120,dtype=torch.float32).reshape(3,40)
        state=dict(original=a,alias=a,views=[a[:,:20],a[:,20:]],nested=(a,torch.ones(7,dtype=torch.int64)))
        cloned=Network._clone_static_cache_value(state,torch.device('cpu'))
        self.assertEqual(static_clone_bytes(state),Network._input_cache_value_bytes(cloned))
        self.assertIs(cloned['original'],cloned['alias'])

    def test_rejected_true_working_set_never_clones_and_false_keeps_legacy_path(self):
        from collections import OrderedDict
        from unittest.mock import patch
        from models.network import Network
        for route in (False,True):
            net=Network.__new__(Network);torch.nn.Module.__init__(net)
            net.args=SimpleNamespace(test_route=route,compress='SparsePCGC',
                heuristic_guidance_mode='ana_den6_online',static_node_cache_cpu=True)
            net.cache_enabled=True;net.input_cache=OrderedDict();net.expected_input_cache_entries=100
            net._static_node_cache_key=lambda *a:'key'
            net._input_cache_limits=lambda:(10,1024)
            state=dict(value=torch.zeros(128))
            with patch.object(net,'_clone_static_cache_value',wraps=Network._clone_static_cache_value) as clone:
                net._put_static_node_cache('frame','full_octree_context',state)
                self.assertEqual(clone.call_count,int(not route))
            self.assertEqual(net._input_cache_working_set_bypassed,1)
            self.assertEqual(net.input_cache,{})

    def test_background_restore_preserves_young_generation_collection(self):
        import gc,pickle,zlib
        from unittest.mock import patch
        from models.utils.cache.autonomous_pool_ram import AutonomousPoolRAM
        enabled=gc.isenabled()
        try:
            gc.enable()
            encoded=zlib.compress(pickle.dumps(dict(value=[1,2,3]),protocol=4))
            with patch('models.utils.cache.autonomous_pool_ram.gc.collect',wraps=gc.collect) as collect:
                self.assertEqual(AutonomousPoolRAM._restore(encoded),dict(value=[1,2,3]))
                collect.assert_called_once_with(1)
            self.assertTrue(gc.isenabled())
        finally:
            (gc.enable if enabled else gc.disable)()


if __name__=='__main__':unittest.main()
