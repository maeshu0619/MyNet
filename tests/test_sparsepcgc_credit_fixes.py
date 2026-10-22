from types import SimpleNamespace
from collections import OrderedDict
import unittest
import torch

from models.utils.training.sparsepcgc_actual_credit import (
    matched_frame_repeats, matched_sparsepcgc_dataset,
)


class _Dataset:
    files = ['frame0', 'frame1', 'frame2']
    all_files = ['frame0', 'frame1', 'frame2', 'heldout']
    def __getitem__(self, index):
        return self.files[index]


class SparseActualCreditTest(unittest.TestCase):
    def args(self, codec='SparsePCGC', mode='train'):
        return SimpleNamespace(compress=codec, heuristic_guidance_mode='ana_den6_online',
                               trainORtest=mode, sparsepcgc_matched_frame_repeats=2)

    def test_consecutive_matching_without_manifest_mutation(self):
        base = _Dataset()
        paired = matched_sparsepcgc_dataset(base, self.args())
        self.assertEqual([paired[i] for i in range(len(paired))],
                         ['frame0','frame0','frame1','frame1','frame2','frame2'])
        self.assertEqual(base.files, ['frame0','frame1','frame2'])
        self.assertEqual(paired.all_files, base.all_files)
        self.assertNotIn('heldout', paired.files)

    def test_other_codecs_and_inference_unchanged(self):
        base = _Dataset()
        for args in [self.args('G-PCC'), self.args('OctAttention'), self.args(mode='test')]:
            self.assertEqual(matched_frame_repeats(args), 1)
            self.assertIs(matched_sparsepcgc_dataset(base, args), base)

    def test_actual_policy_not_subject_to_legacy_ste_attenuation(self):
        from models.modules.structure_actuator import StructureRepairActuator
        actuator = StructureRepairActuator.__new__(StructureRepairActuator)
        torch.nn.Module.__init__(actuator)
        for codec, expected in [('SparsePCGC',1.),('G-PCC',0.005),('OctAttention',0.005)]:
            actuator.args = self.args(codec)
            actuator.args.repair_drop_where_downstream_grad_scale = 0.005
            actuator.args.repair_where_downstream_grad_min_scale = 0.005
            value=torch.tensor([1.,2.],requires_grad=True)
            result=actuator._scale_where_downstream_grad(value,'drop')
            self.assertTrue(torch.equal(result.detach(),value.detach()))
            result.sum().backward()
            self.assertTrue(torch.allclose(value.grad,torch.full_like(value,expected)))

    def test_coarse_clip_cannot_shrink_fine_amount(self):
        from models.utils.training.train_runtime import _balance_actual_operation_head_gradients
        model=torch.nn.Module(); model.actuator=torch.nn.Module()
        for name in ['drop_amount_head','add_amount_head','move_amount_head',
                     'algorithmic_amount_selector_head']:
            setattr(model.actuator,name,torch.nn.Linear(1,1,bias=False))
        for codec in ['SparsePCGC','G-PCC','OctAttention']:
            args=self.args(codec)
            args.repair_balance_operation_head_grads=True
            args.repair_online_decision_grad_max_norm=1.
            for name,p in model.actuator.named_parameters():
                p.grad=torch.full_like(p,100. if name.startswith('algorithmic') else .1)
            _balance_actual_operation_head_gradients(args,model,{})
            fine=model.actuator.drop_amount_head.weight.grad.item()
            if codec=='SparsePCGC':
                self.assertAlmostEqual(fine,.1,places=6)
            else:
                self.assertLess(fine,.002)

    def test_native_adapter_auto_warmup_has_representable_gradient(self):
        from models.network import Network
        model=Network.__new__(Network); torch.nn.Module.__init__(model)
        model.point_transformer_feature_gate=torch.nn.Parameter(torch.full((1,8,1),.1))
        model.args=self.args()
        model.args.point_transformer_feature_warmup_steps=0
        model.args._total_train_steps_estimate=400000
        model.args.repair_exploration_fraction=.25
        model.args._global_train_step=0
        gate,warmup=model._point_transformer_fusion_gate(torch.ones(1,8,2))
        self.assertGreater(warmup,1e-4)
        self.assertLess(float(gate.abs().max()),.1)
        gate.sum().backward()
        self.assertGreater(float(model.point_transformer_feature_gate.grad.norm()),1e-5)
        model.args.compress='G-PCC'
        _,old_warmup=model._point_transformer_fusion_gate(torch.ones(1,8,2))
        self.assertLess(old_warmup,1e-8)

    def test_geometry_credit_requires_same_frame_anchor_and_rate_guard(self):
        from models.utils.training.sparsepcgc_actual_credit import matched_geometry_advantage
        memory=OrderedDict()
        def observe(frame,anchor,reused,rate,geometry):
            return matched_geometry_advantage(memory,frame_key=frame,anchor_hash=anchor,
                group='Fine',reused=reused,rate=rate,geometry=geometry,tolerance=.05)
        self.assertIsNone(observe('frame0','anchor0',False,-2.,.03))
        self.assertIsNone(observe('frame1','anchor0',True,-2.,.02))
        self.assertIsNone(observe('frame0','other',True,-2.,.02))
        self.assertAlmostEqual(observe('frame0','anchor0',True,-2.,.027),.1)
        observe('frame0','anchor1',False,-2.,.03)
        self.assertIsNone(observe('frame0','anchor1',True,-1.,.02))

    def test_runtime_and_parameter_checkpoint_roundtrip(self):
        import io
        from models.network import Network
        def make():
            model=Network.__new__(Network);torch.nn.Module.__init__(model)
            model.args=self.args();model.actuator=torch.nn.Linear(2,2)
            return model
        before=make(); after=make()
        before._sparsepcgc_geometry_pair_memory=OrderedDict({('frame','anchor','Fine'):(-2.,.03)})
        before.actuator._sparsepcgc_group_last_opened_step={'Fine':10,'WhereAll':12}
        stream=io.BytesIO()
        torch.save(dict(model=before.state_dict(),runtime=before.actor_critic_runtime_state_dict()),stream)
        stream.seek(0);saved=torch.load(stream)
        after.load_state_dict(saved['model'],strict=True)
        self.assertTrue(after.load_actor_critic_runtime_state_dict(saved['runtime']))
        self.assertEqual(before._sparsepcgc_geometry_pair_memory,after._sparsepcgc_geometry_pair_memory)
        self.assertEqual(before.actuator._sparsepcgc_group_last_opened_step,
                         after.actuator._sparsepcgc_group_last_opened_step)
        x=torch.ones(1,2)
        self.assertTrue(torch.equal(before.actuator(x),after.actuator(x)))

    def test_amount_tanh_keeps_true_tail_derivative_and_same_forward(self):
        from models.utils.training.sparsepcgc_actual_credit import stable_amount_tanh
        values=torch.tensor([-55.37666/6.,-52.06878/6.,-45.63126/6.,0.,2.,10.],requires_grad=True)
        actual=stable_amount_tanh(values)
        self.assertTrue(torch.allclose(actual,torch.tanh(values),atol=1.2e-7,rtol=0.))
        actual.sum().backward()
        reference=1.-torch.tanh(values.detach().double()).square()
        self.assertTrue(torch.allclose(values.grad.double(),reference,atol=1e-8,rtol=1e-5))
        self.assertGreater(values.grad[0].item(),0.)
        legacy=values.detach().clone().requires_grad_()
        torch.tanh(legacy).sum().backward()
        self.assertEqual(legacy.grad[0].item(),0.)

    def test_recent_teacher_coverage_does_not_inherit_lifetime_starvation(self):
        from models.utils.training.sparsepcgc_actual_credit import refresh_actual_credit_group
        groups=('WhereAll','Amount','Gate','Fine')
        scores={'WhereAll':.64,'Amount':.62,'Gate':.60,'Fine':.69}
        recent={}
        chosen=[refresh_actual_credit_group('Fine',groups,scores,recent,step,.1)
                for step in range(0,8,2)]
        self.assertEqual(chosen,['Fine','WhereAll','Amount','Gate'])
        self.assertEqual(refresh_actual_credit_group('Fine',groups,scores,recent,8,.1),'Fine')
        self.assertEqual(refresh_actual_credit_group('Fine',groups,scores,recent,24,.1),'WhereAll')
        # A clock rewind is a warm-start, not an exact optimizer resume.
        self.assertEqual(refresh_actual_credit_group('Fine',groups,scores,recent,0,.1),'Fine')
        self.assertEqual(set(recent),{'Fine'})

    def test_native_group_refresh_keeps_other_codecs_selector_unchanged(self):
        from models.modules.structure_actuator import StructureRepairActuator
        for codec in ('SparsePCGC','G-PCC','OctAttention'):
            module=StructureRepairActuator.__new__(StructureRepairActuator)
            torch.nn.Module.__init__(module)
            module.args=self.args(codec)
            module.args.heuristic_guidance_online_simultaneous_operation_exploration=True
            module._den6_factorized_group_counts={'WhereAll':4435,'Amount':1816,'Gate':990,'Fine':2309}
            module._den6_group_actual_authority=lambda name: 0. if name=='Fine' else 1.
            selected=[]
            for step in range(0,8,2):
                module.args._global_train_step=step
                selected.append(module._select_den6_factorized_exploration_group({}))
            if codec=='SparsePCGC':
                self.assertEqual(set(selected),{'WhereAll','Amount','Gate','Fine'})
            else:
                self.assertEqual(selected,['Fine']*4)

    def test_real_amount_head_tail_keeps_bounds_and_inference_consistency(self):
        from models.modules.structure_actuator import StructureRepairActuator
        outputs={}
        for codec,mode in [('SparsePCGC','train'),('SparsePCGC','test'),('G-PCC','train')]:
            module=StructureRepairActuator.__new__(StructureRepairActuator)
            torch.nn.Module.__init__(module)
            module.args=self.args(codec,mode)
            module._annealed_value=lambda *a:0.
            head=torch.nn.Conv1d(2,1,1)
            torch.nn.init.zeros_(head.weight);torch.nn.init.constant_(head.bias,-55.37666)
            ratio=module._learned_operation_ratio(torch.zeros(1,2,3),head,.3,'unused','unused')
            outputs[(codec,mode)]=ratio.detach()
            ratio.sum().backward()
            self.assertGreaterEqual(ratio.item(),0.)
            self.assertLessEqual(ratio.item(),.3)
            if codec=='SparsePCGC':
                self.assertGreater(head.bias.grad.item(),0.)
            else:
                self.assertEqual(head.bias.grad.item(),0.)
        self.assertTrue(torch.equal(outputs[('SparsePCGC','train')],outputs[('SparsePCGC','test')]))
        self.assertTrue(torch.allclose(outputs[('SparsePCGC','train')],outputs[('G-PCC','train')],atol=1e-9))


if __name__ == '__main__':
    unittest.main()
