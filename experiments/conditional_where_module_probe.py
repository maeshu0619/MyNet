"""CPU real-actuator graph probe; analytic costs, NOT Actual compression.

Checks executed sampling, gradients and optimizer updates around the old
9550-step cutoff. No codec is run and no GPU memory is allocated.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-snapshot', type=Path)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    if options.source_snapshot:
        from experiments.sparsepcgc_direction.common import FrozenSources
        sys.meta_path.insert(0, FrozenSources(options.source_snapshot))
    import torch
    from models.modules.structure_actuator import StructureRepairActuator
    from models.utils.training.sparsepcgc_autonomous_policy import configure_autonomous_search
    from models.utils.training.autonomous_joint_learning import joint_actual_policy_loss, clip_joint_gradients
    from tests.test_sparsepcgc_amount_policy import SparseAmountPolicyTest
    torch.set_num_threads(2)
    legacy, builder = SparseAmountPolicyTest().actuator_and_guidance()
    captured = inspect.getclosurevars(builder).nonlocals
    coords, guidance = captured['coords'], copy.deepcopy(captured['guidance'])
    n = coords.shape[-1]
    for mapping in guidance['candidate_tensor_map'].values():
        mapping['local_rate_benefit'] = torch.linspace(-2.,2.,32)
        mapping['local_geometry_risk'] = torch.zeros(32)
    records = []
    for learned in (False, True):
        for global_step in (0,9549,9550,38200):
            for seed in (7,8):
                torch.manual_seed(seed)
                args = copy.deepcopy(legacy.args)
                args.test_route = True
                args.test_route_codec_context = True
                args.test_route_learned_exploration = learned
                args.test_route_where_conditional_baseline = True
                args.sparsepcgc_mode = 'dense_lossy'
                args.repair_policy_exploration_mode = 'annealed'
                args._exploration_schedule_steps_estimate = 38200
                args._total_train_steps_estimate = 38200
                args._global_train_step = global_step
                args.test_route_amount_uniform_fraction = .25
                args.test_route_structured_where_noise = 4.
                args._current_input_file = 'analytic_module_fixture'
                configure_autonomous_search(args)
                actuator = StructureRepairActuator(2,hidden_dim=4,args=args)
                actuator.train()
                features = torch.randn(1,2,n)
                heads = [('add_amount_head',.03),('drop_amount_head',.03),('move_amount_head',.03)]
                ratios = [actuator._learned_operation_ratio(features,getattr(actuator,name),maximum,
                          'unit_mix_start','unit_mix_end') for name,maximum in heads]
                drop = actuator._run_large_head(actuator.drop_head,features)
                source = actuator._run_large_head(actuator.subtree_move_source_head,features)
                move = actuator._run_large_head(actuator.move_voxel_head,features)
                add = actuator._run_large_head(actuator.add_voxel_head,features).transpose(1,2)
                result = actuator._build_exact_den6_residual_plan(guidance,coords,*ratios,
                    drop,source,move,add,actuator.algorithmic_amount_selector_head(features).mean(2),
                    actuator.operation_gate_head(features).mean(2))
                final,debug = result
                rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
                counts = debug['selected_counts']
                # Explicitly synthetic cost: no invented Actual labels.
                cost = .01*sum(counts.values())
                geom = .001*sum(counts.values())/n
                model = SimpleNamespace(args=args,actuator=actuator,
                    last_actuator_voxel_state={'ana_den6_exact_residual_plan_debug':debug})
                loss = joint_actual_policy_loss(model,torch.tensor(cost),torch.tensor(geom))
                before = {name:p.detach().clone() for name,p in actuator.named_parameters()}
                loss.backward()
                groups = dict(actor=('den6_candidate_actor',),proposal=('den6_candidate_proposal',),
                    where=('drop_head','subtree_move_source_head','move_voxel_head','add_voxel_head'),
                    amount=('add_amount_head','drop_amount_head','move_amount_head','algorithmic_amount_selector_head'),
                    gate=('operation_gate_head',))
                norms = {}
                for role,names in groups.items():
                    ps=[p for name,module in actuator.named_children() if name in names for p in module.parameters()]
                    square=sum(float(p.grad.detach().double().square().sum()) for p in ps if p.grad is not None)
                    norms[role]=dict(l2=math.sqrt(square),rms=math.sqrt(square/max(sum(p.numel() for p in ps),1)))
                owner = torch.nn.Module();owner.actuator=actuator;owner.args=args
                clip_joint_gradients(owner,1.)
                optimizer = torch.optim.Adam(actuator.parameters(),lr=1e-4)
                optimizer.step()
                delta = math.sqrt(sum(float((p.detach()-before[name]).double().square().sum())
                                      for name,p in actuator.named_parameters()))
                row=dict(learned_exploration=learned,global_step=global_step,seed=seed,
                    cost_scope='analytic_module_fixture_not_Actual',counts=counts,
                    edit_hash=hashlib.sha256(final.detach().numpy().tobytes()).hexdigest(),
                    rng_hash=rng,executed_phase=debug['exploration_multiplier'],
                    credit_family=debug.get('joint_credit_family'),
                    likelihood=float(debug['_joint_policy_log_prob'].detach()),
                    gradients=norms,parameter_delta_l2=delta)
                records.append(row)
                with (options.out/'records.jsonl').open('a') as output:
                    output.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    (options.out/'completed.json').write_text(json.dumps(dict(cases=len(records))))


if __name__=='__main__':
    main()
