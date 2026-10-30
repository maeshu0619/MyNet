import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zlib
from unittest import mock

import numpy as np
import torch

from models.utils.loss.autonomous_native_rate import exact_dense_logical_rate, autonomous_rate_schema
from models.utils.pointcloud.autonomous_context_proposals import canonical_voxel_sha, cell_counts, context_proposals


class AutonomousRecoveryTest(unittest.TestCase):
    def test_context_reservoir_survives_eight_passes_and_stops_rotating_after_exploration(self):
        from models.utils.training.sparsepcgc_autonomous_policy import autonomous_pool_phase
        args=SimpleNamespace(test_route=True,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online',
            test_route_codec_context=True,sparsepcgc_mode='dense_lossy',_fixed_episode_steps=100,
            _exploration_schedule_steps_estimate=38200,_global_train_step=799,repair_exploration_fraction=.25)
        self.assertEqual(autonomous_pool_phase(args),0)
        args._global_train_step=800;self.assertEqual(autonomous_pool_phase(args),1)
        args._global_train_step=10000;a=autonomous_pool_phase(args)
        args._global_train_step=38000;self.assertEqual(autonomous_pool_phase(args),a)
        args.test_route_codec_context=False;args._global_train_step=200
        self.assertEqual(autonomous_pool_phase(args),1)

    def test_cpu_threads_are_true_only_and_honor_explicit_omp(self):
        import os
        from models.utils.training.autonomous_cpu import configure_autonomous_cpu
        args=SimpleNamespace(test_route=False,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online')
        with mock.patch.dict(os.environ,{},clear=True),mock.patch('torch.set_num_threads') as setter:
            self.assertIsNone(configure_autonomous_cpu(args));setter.assert_not_called()
            self.assertNotIn('OMP_NUM_THREADS',os.environ)
            args.test_route=True;self.assertEqual(configure_autonomous_cpu(args),2)
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'2')
            os.environ['OMP_NUM_THREADS']='7'
            self.assertEqual(configure_autonomous_cpu(args),7)

    def test_uniform_component_explores_absolute_amount_after_fine_collapse(self):
        from models.utils.training.autonomous_joint_learning import sample_joint_amounts
        means={op:torch.tensor(-3.,requires_grad=True) for op in ('Add','Prune','Adjust')}
        bins=torch.tensor([0.,.001,.01,.03])
        selector=torch.tensor([5.,0.,0.,0.],requires_grad=True)
        gate=torch.zeros(3,requires_grad=True)
        observed=[]
        for seed in range(60):
            torch.manual_seed(seed)
            action=sample_joint_amounts(selector,means,gate,bins,.03,1.,True,uniform_fraction=.5)
            observed.extend(float(v.detach()) for v in action['ratios'].values())
            # Compare the evaluated density with the exact joint mixture.
            for op in means:
                index=action['indices'][op];x=action['samples'][op].detach()
                q=torch.log_softmax(action['logits'][op]/2.,0)[index]
                if bins[index]>0:
                    q=q+torch.distributions.Normal(means[op],.3).log_prob(x)
                    u=-np.log(4)+torch.distributions.Normal(torch.tensor(0.),.3).log_prob(x)
                else:u=q*0.-np.log(4)
                expected=torch.logaddexp(q+np.log(.5),u+np.log(.5))
                self.assertTrue(torch.allclose(action['log_probs'][op],expected))
        self.assertGreater(max(observed),.02)
        self.assertIn(0.,observed)

    def fixture(self):
        ops=('Add','Prune','Adjust')
        amount={op:torch.tensor([2.,0.],requires_grad=True) for op in ops}
        means={op:torch.tensor(-2.,requires_grad=True) for op in ops}
        where={op:torch.tensor([0.,1.],requires_grad=True) for op in ops}
        action=dict(amount_indices={op:1 for op in ops},fine_samples={op:0. for op in ops},
                    where_prefix_ids={op:['good'] for op in ops})
        debug=dict(_joint_policy_log_prob=torch.tensor(0.,requires_grad=True),
            _joint_action=action,_joint_amount_logits=amount,_joint_amount_means=means,
            _joint_candidate_ids={op:('good','bad') for op in ops},_joint_replay_logits=where)
        best=dict(schema='measured_joint_action_v1',compression=-4.,geometry=.02,action=action)
        previous={'best_action':zlib.compress(json.dumps(best).encode())}
        args=SimpleNamespace(test_route_actual_plan_replay=True,w_com=3.,cp_lambda_geom=150.)
        return debug,previous,args

    def test_measured_good_plan_trains_all_heads_even_when_greedy_policy_is_noop(self):
        from models.utils.training.autonomous_actual_replay import measured_plan_replay
        d,previous,args=self.fixture()
        loss,replacement,report=measured_plan_replay(d,previous,0.,0.,args)
        self.assertTrue(report['used']);self.assertIsNone(replacement)
        loss.backward()
        for op in ('Add','Prune','Adjust'):
            self.assertLess(float(d['_joint_amount_logits'][op].grad[1]),0.)
            self.assertLess(float(d['_joint_amount_means'][op].grad),0.)
            self.assertLess(float(d['_joint_replay_logits'][op].grad[0]),0.)
        self.assertIsNone(d['_joint_policy_log_prob'].grad)

    def test_missing_action_never_becomes_partial_actual_teacher(self):
        from models.utils.training.autonomous_actual_replay import measured_plan_replay
        d,previous,args=self.fixture();d['_joint_candidate_ids']['Prune']=('other','bad')
        loss,_,report=measured_plan_replay(d,previous,0.,0.,args)
        self.assertFalse(report['used']);self.assertEqual(float(loss),0.)

    def test_unreachable_old_best_does_not_block_a_new_measured_profitable_plan(self):
        from models.utils.training.autonomous_actual_replay import measured_plan_replay
        d,previous,args=self.fixture();d['_joint_candidate_ids']['Prune']=('other','bad')
        d['_joint_action']['where_prefix_ids']['Prune']=['other']
        _,replacement,report=measured_plan_replay(d,previous,-1.,.001,args)
        self.assertFalse(report['used'])
        self.assertIsNotNone(replacement)
        self.assertEqual(json.loads(zlib.decompress(replacement))['compression'],-1.)

    def test_only_profitable_measured_joint_plans_are_stored_and_memory_is_bounded(self):
        from models.utils.training.autonomous_actual_replay import measured_plan_replay,bound_replay_memory
        d,previous,args=self.fixture()
        for c,g in [(2.,.0),(-1.,.1),(0.,0.)]:
            _,replacement,_=measured_plan_replay(d,None,c,g,args)
            self.assertIsNone(replacement)
        _,replacement,_=measured_plan_replay(d,None,-4.,.01,args)
        self.assertIsNotNone(replacement)
        memory={'a':dict(best_action=replacement,ema_c=-4.),'b':dict(ema_c=0.)}
        self.assertEqual(bound_replay_memory(memory,0),0)
        self.assertEqual(memory['a']['ema_c'],-4.)


class NativeRateTest(unittest.TestCase):
    def test_gt_preparation_accepts_parsed_lists_and_encodes_only_once(self):
        from models.utils.loss.autonomous_codec_context import prepare_codec_context
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(test_route=True,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online',
                test_route_codec_context=True,sparsepcgc_mode='dense_lossy',sparsepcgc_scale_ae=0,
                sparsepcgc_scale_sr=2,sparsepcgc_dense_scale_ae_list=[0],sparsepcgc_dense_scale_sr_list=[2],
                heuristic_guidance_online_cache_dir=directory)
            class FakeGTOnlyLoss:
                stats=None
                encodes=0
                def _get_actual_encoder(self,args):return SimpleNamespace(_codec_cache_fingerprint=lambda:'fixture')
                def _get_cached_actual_gt(self,key):return self.stats
                def _store_cached_actual_gt(self,key,stats):self.stats=stats
                def _encode_actual_batch(self,args,gt):
                    self.encodes+=1
                    path=Path(args._native_codec_context_output);path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez(path,metadata=json.dumps(args._native_codec_context_metadata))
                    return dict(bit=100.,fixture_not_codec_measurement=True)
            loss=FakeGTOnlyLoss();gt=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]])
            first=prepare_codec_context(args,loss,gt,'frame')
            second=prepare_codec_context(args,loss,gt,'frame')
            self.assertEqual(first,second);self.assertEqual(loss.encodes,1)
            self.assertFalse(hasattr(args,'_native_codec_context_output'))
            with self.assertRaisesRegex(ValueError,'integer GT'):
                prepare_codec_context(args,loss,gt+.1,'frame')

    def test_logical_native_result_is_not_billed_twice(self):
        result=exact_dense_logical_rate({'file_size':38280},
            {'file_size':38280,'bitstream_file_bits':38272},2,0)
        self.assertEqual(result['file_size'],38280)
        self.assertEqual(result['main_bin_bits'],38272)
        self.assertEqual(result['side_information_bits'],8)

    def test_old_normalizer_and_ae_information_are_preserved(self):
        self.assertEqual(exact_dense_logical_rate({'file_size':108},
            {'file_size':100,'bitstream_file_bits':100},2,0)['file_size'],108)
        self.assertEqual(exact_dense_logical_rate({'file_size':132},
            {'bitstream_file_bits':100},2,1)['file_size'],132)
        with self.assertRaises(ValueError):
            exact_dense_logical_rate({'file_size':100},{},2,0)

    def test_schema_is_sparse_true_dense_only(self):
        args=SimpleNamespace(test_route=False,compress='SparsePCGC',
            heuristic_guidance_mode='ana_den6_online',sparsepcgc_mode='dense_lossy')
        self.assertIsNone(autonomous_rate_schema(args))
        args.test_route=True
        self.assertIsNotNone(autonomous_rate_schema(args))
        args.compress='gpcc'
        self.assertIsNone(autonomous_rate_schema(args))


class CodecContextProposalTest(unittest.TestCase):
    def fixture(self,directory):
        source=np.array([[2,2,2],[3,2,2],[4,4,4],[6,6,6],[8,8,8],[5,4,4],[4,5,4]],dtype=np.int64)
        cells=np.stack(np.meshgrid(*[np.arange(1,5)]*3,indexing='ij'),axis=-1).reshape(-1,3)
        occupied=np.array([tuple(c) in set(map(tuple,source//2)) for c in cells],dtype=np.uint8)
        path=Path(directory)/'context.npz'
        np.savez(path,coords=cells,probability=np.where(occupied,.1,.9),occupied=occupied,
            metadata=json.dumps(dict(scale=2,input_voxel_sha256=canonical_voxel_sha(source))))
        random={op:[] for op in ('Add','Prune','Adjust')}
        random['Prune']=[dict(remove_coords=[[4,4,4]],add_coords=[])]
        return source,path,random

    def test_cell_counts_match_numpy_including_negatives_and_duplicates(self):
        source=np.random.RandomState(1).randint(-10,11,(1000,3))
        a=cell_counts(source);b=np.unique(source,axis=0,return_inverse=True,return_counts=True)
        for x,y in zip(a,b):self.assertTrue(np.array_equal(x,y))

    def test_hash_is_order_and_duplicate_independent(self):
        a=np.array([[2,3,4],[-2,1,3],[2,3,4]])
        self.assertEqual(canonical_voxel_sha(a),canonical_voxel_sha(a[::-1]))
        self.assertEqual(canonical_voxel_sha(a),canonical_voxel_sha(np.unique(a,axis=0)))

    def test_exact_cache_parity_legal_edits_and_signed_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            source,path,random=self.fixture(directory)
            a,stats=context_proposals(source,path,20,31,random)
            b,again=context_proposals(source,path,20,31,random)
            self.assertEqual(a,b)
            self.assertTrue(all(v['core_cache_hit'] for v in again.values()))
            occupied=set(map(tuple,source));all_candidates=[]
            for op,pool in a.items():
                self.assertLessEqual(len(pool),20)
                self.assertEqual(len({c['candidate_id'] for c in pool}),len(pool))
                for c in pool:
                    self.assertEqual(c['rank_score'],0.)
                    self.assertEqual(c['heuristic_score'],0.)
                    r=set(map(tuple,c['remove_coords']));p=set(map(tuple,c['add_coords']))
                    self.assertTrue(r.issubset(occupied));self.assertFalse(p & occupied)
                    if op=='Adjust':
                        self.assertLessEqual(np.max(np.abs(np.array(c['remove_coords'])-np.array(c['add_coords']))),1)
                    all_candidates.append(c)
            self.assertTrue(any(c['optimistic_gain_bits']>0 for c in all_candidates))

    def test_different_gt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source,path,random=self.fixture(directory)
            with self.assertRaisesRegex(ValueError,'different GT'):
                context_proposals(source+1,path,20,31,random)


if __name__=='__main__':unittest.main()
