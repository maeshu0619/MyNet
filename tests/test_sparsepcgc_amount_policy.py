from types import SimpleNamespace
import unittest
import torch

from models.utils.training.sparsepcgc_amount_policy import (
    amount_learning_enabled, calibrated_log_residual, migrate_amount_reference,
    reference_raw, register_amount_reference, sampled_bin_log_probs,
)


class _Owner(torch.nn.Module):
    def __init__(self, codec='SparsePCGC'):
        super().__init__()
        self.args=SimpleNamespace(compress=codec, heuristic_guidance_mode='ana_den6_online',
                                  sparsepcgc_amount_exploration=True)
        for name in ['add_amount_head','drop_amount_head','move_amount_head']:
            setattr(self,name,torch.nn.Conv1d(2,1,1))
        register_amount_reference(self)

    def _load_from_state_dict(self,state,prefix,*args):
        migrate_amount_reference(self,state,prefix)
        super()._load_from_state_dict(state,prefix,*args)


class SparseAmountPolicyTest(unittest.TestCase):
    def actuator_and_guidance(self, downstream_scale=1.):
        from models.modules.structure_actuator import StructureRepairActuator
        args=SimpleNamespace(compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online',
            sparsepcgc_amount_exploration=True,sparsepcgc_amount_joint_exploration=True,
            heuristic_guidance_anchor_steps=0,heuristic_guidance_exact_anchor_steps=0,
            _global_train_step=2,_total_train_steps_estimate=1000,
            repair_policy_exploration_mode='constant',repair_policy_exploration_constant_multiplier=1.,
            repair_exploration_fraction=.25,_current_input_file='unit_frame')
        args.unit_mix_start=0.;args.unit_mix_end=0.
        args.sparsepcgc_algorithmic_amount_bins='0.0015,0.002,0.0025,0.00275,0.003'
        model=StructureRepairActuator(2,hidden_dim=4,args=args)
        model._select_den6_factorized_exploration_group=lambda audit:'Amount'
        n=5000;coords=torch.stack((torch.arange(n),torch.zeros(n,dtype=torch.long),torch.zeros(n,dtype=torch.long))).unsqueeze(0)
        pools={op:[] for op in ['Add','Prune','Adjust']};mapping={}
        for op,offset in [('Add',0),('Prune',100),('Adjust',200)]:
            for i in range(32):
                pools[op].append(dict(candidate_id=op+str(i),operation=op,
                    remove_coords=[] if op=='Add' else [[offset+i,0,0]],
                    add_coords=[] if op=='Prune' else [[offset+i,1,0]]))
            mapping[op]=dict(rank_score=torch.linspace(1.,0.,32),pool_rank=torch.arange(32),
                static_compatible=torch.ones(32,dtype=torch.bool),source_index=torch.arange(32)+offset,
                pair_candidate_index=torch.arange(32),pair_source_index=torch.arange(32)+offset,
                pair_direction_index=torch.zeros(32,dtype=torch.long),direction_index=torch.zeros(32,dtype=torch.long))
        guidance=dict(exact_candidate_guidance=dict(operation_edit_units=pools,total_ratio=.0025,
            operation_shares={'Add':.4,'Prune':.4,'Adjust':.2},operation_priority=['Add','Prune','Adjust']),candidate_tensor_map=mapping)
        features=torch.randn(1,2,11)
        def run():
            ratios=[model._learned_operation_ratio(features,getattr(model,name),maximum,'unit_mix_start','unit_mix_end')
                for name,maximum in [('add_amount_head',.002),('drop_amount_head',.002),('move_amount_head',.001)]]
            ratios=[ratio*downstream_scale for ratio in ratios]
            return model._build_exact_den6_residual_plan(guidance,coords,*ratios,
                torch.zeros(1,1,n),torch.zeros(1,1,n),torch.zeros(1,26,n),torch.zeros(1,n,26),
                model.algorithmic_amount_selector_head(features).mean(dim=2))
        return model,run

    def test_reference_keeps_downstream_calibration_in_the_legacy_forward(self):
        model,run=self.actuator_and_guidance(downstream_scale=.8);model.eval()
        new=run()[1]
        model.args.sparsepcgc_amount_exploration=False
        old=run()[1]
        for operation in ['Add','Prune','Adjust']:
            self.assertAlmostEqual(new['operation_amount_policy_means'][operation],
                                   old['operation_amount_policy_means'][operation],places=7)
        self.assertEqual(new['plan_hash'],old['plan_hash'])

    def test_actual_amount_group_reaches_all_three_heads_and_coarse(self):
        model,run=self.actuator_and_guidance();model.train()
        torch.manual_seed(7);first=run();torch.manual_seed(8);second=run()
        debug=second[1]
        self.assertTrue(debug['matched_actual_anchor_reused'])
        self.assertTrue(debug['sparse_amount_joint_exploration'])
        # A real score likelihood, without inventing an Actual codec label.
        self.assertFalse(debug['no_op_selected'])
        tensors=[model.add_amount_head.weight,model.drop_amount_head.weight,model.move_amount_head.weight,
                 model.algorithmic_amount_selector_head[-1].bias]
        grads=torch.autograd.grad(debug['_actual_credit_group_log_prob'],tensors,allow_unused=True)
        for grad in grads:
            self.assertIsNotNone(grad);self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0.)
        counts=debug['selected_counts']
        self.assertLessEqual(sum(counts.values()),49)
        self.assertLessEqual(counts['Add']+counts['Prune']+2*counts['Adjust'],49)

    def test_noise_is_zero_in_eval_and_after_the_existing_window(self):
        model,run=self.actuator_and_guidance();model.eval()
        results=[]
        for seed in [2,3]:
            torch.manual_seed(seed);result=run();results.append(result[1]['plan_hash'])
            self.assertTrue(all(s==0. for s in result[1]['operation_amount_sigmas'].values()))
        self.assertEqual(results[0],results[1])
        model.train();model.args.repair_policy_exploration_mode='annealed';model.args._global_train_step=300
        result=run()
        self.assertFalse(result[1]['exploration_active'])
        self.assertTrue(all(s==0. for s in result[1]['operation_amount_sigmas'].values()))

    def test_learning_changes_executed_voxels_not_only_the_gradient(self):
        model,run=self.actuator_and_guidance();model.eval()
        first=run()[1]
        with torch.no_grad():
            model.add_amount_head.bias.add_(.5)
        second=run()[1]
        self.assertGreater(second['selected_counts']['Add'], first['selected_counts']['Add'])
        self.assertNotEqual(second['plan_hash'],first['plan_hash'])
        self.assertAlmostEqual(sum(second['amount_behavior_probabilities']),1.,places=6)

    def test_exploration_decreases_before_the_existing_quarter_window_ends(self):
        model,run=self.actuator_and_guidance();model.train()
        model.args.repair_policy_exploration_mode='annealed'
        sigmas=[]
        for step in [0,125,250]:
            model.args._global_train_step=step
            model._den6_matched_actual_anchor_records={}
            result=run()[1]
            sigmas.append(result['operation_amount_sigmas']['Add'])
        self.assertGreater(sigmas[0],sigmas[1]);self.assertGreater(sigmas[1],0.)
        self.assertEqual(sigmas[2],0.)

    def test_other_codecs_disabled_and_no_reference_buffers(self):
        for codec in ['G-PCC','OctAttention','Draco']:
            model=_Owner(codec)
            self.assertFalse(amount_learning_enabled(model.args))
            self.assertFalse(list(model.buffers()))

    def test_legacy_loading_preserves_forward_for_every_input(self):
        old=_Owner();state={k:v for k,v in old.state_dict().items() if not k.startswith('sparse_amount_')}
        state['drop_amount_head.bias']=torch.tensor([-55.])
        new=_Owner();result=new.load_state_dict(state,strict=True)
        self.assertEqual(result.missing_keys,[]);self.assertEqual(result.unexpected_keys,[])
        x=torch.randn(1,2,7)
        for op,name in [('Add','add_amount_head'),('Prune','drop_amount_head'),('Adjust','move_amount_head')]:
            raw=getattr(new,name)(x);ref=reference_raw(new,op,x)
            self.assertTrue(torch.equal(raw,ref))
            self.assertTrue(torch.equal(calibrated_log_residual(raw,ref,.693),torch.zeros_like(raw)))

    def test_saturated_head_has_real_forward_sensitivity_not_gradient_only(self):
        ref=torch.tensor([-55.]);raw=ref.clone().requires_grad_()
        y=calibrated_log_residual(raw,ref,.693)
        self.assertEqual(float(y),0.)
        self.assertEqual(float(torch.autograd.grad(y,raw)[0]),1.)
        for delta in [-.01,.01]:
            changed=calibrated_log_residual(ref+delta,ref,.693)
            self.assertGreater(float(changed)*delta,0.)
            self.assertLess(abs(float(changed)),.693)

    def test_reference_is_fixed_and_roundtrip_keeps_learned_delta(self):
        model=_Owner();initial=model.sparse_amount_reference_prune_bias.clone()
        with torch.no_grad():model.drop_amount_head.bias.add_(.1)
        self.assertTrue(torch.equal(model.sparse_amount_reference_prune_bias,initial))
        restored=_Owner();restored.load_state_dict(model.state_dict(),strict=True)
        x=torch.randn(1,2,1)
        for m in [model,restored]:
            self.assertAlmostEqual(float((m.drop_amount_head(x)-reference_raw(m,'Prune',x)).detach()),.1,places=5)

    def test_sampling_temperature_matches_the_executed_distribution(self):
        logits=torch.tensor([0.,1.,-1.],requires_grad=True)
        logp=sampled_bin_log_probs(logits,2.)
        self.assertTrue(torch.allclose(logp.exp(),torch.softmax(logits/2.,dim=0)))
        (-logp[1]).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.allclose(logits.grad,(torch.softmax(logits.detach()/2.,dim=0)-torch.tensor([0.,1.,0.]))/2.))


if __name__=='__main__':unittest.main()
