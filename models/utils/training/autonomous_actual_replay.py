"""Self-imitation of measured COMPLETE profitable plans on identical GT.

No individual candidate is labelled with the whole plan's improvement. This
is supervised likelihood of a stored joint action, not an on-policy gradient
of the current greedy action and not a new codec measurement.
"""
import json
import math
import zlib

import torch
import torch.nn.functional as F

OPERATIONS = ('Add', 'Prune', 'Adjust')


def measured_plan_replay(debug, previous, compression, geometry, args):
    zero = debug['_joint_policy_log_prob']*0.
    if not bool(getattr(args, 'test_route_actual_plan_replay', False)) or '_joint_action' not in debug:
        return zero, None, dict(used=False, reason='disabled_or_no_joint_action')
    wc, wg = float(getattr(args, 'w_com', 1.)), float(getattr(args, 'cp_lambda_geom', 1.))
    tau = float(getattr(args, 'cp_tau_geom', 0.))
    best = json.loads(zlib.decompress(previous['best_action'])) if previous and previous.get('best_action') else None
    best_rd = wc*best['compression']+wg*max(best['geometry']-tau, 0.) if best else float('inf')
    loss, used, reason = zero, False, 'no_profitable_measurement'
    teacher_available = True
    if best is not None and best_rd < 0. and best['compression'] < 0.:
        amount_terms, where_terms = [], []
        valid = True
        for op in OPERATIONS:
            logits = debug['_joint_amount_logits'][op]
            index = int(best['action']['amount_indices'][op])
            if index >= logits.numel() or index < 0:
                valid = False
                break
            amount_lp = F.log_softmax(logits, dim=0)[index]
            if index > 0:
                sigma = max(float(getattr(args, 'sparsepcgc_amount_exploration_log_sigma', .3)), .02)
                amount_lp = amount_lp+torch.distributions.Normal(
                    debug['_joint_amount_means'][op], sigma).log_prob(logits.new_tensor(
                        best['action']['fine_samples'][op]))
            amount_terms.append(-amount_lp)
            ids = best['action']['where_prefix_ids'][op]
            if not ids:
                continue
            mapping = {name:i for i,name in enumerate(debug['_joint_candidate_ids'][op])}
            if any(name not in mapping for name in ids):
                valid = False
                break
            prefix = torch.tensor([mapping[name] for name in ids], device=logits.device, dtype=torch.long)
            scores = debug['_joint_replay_logits'][op]
            if not bool(torch.isfinite(scores[prefix]).all()):
                valid = False
                break
            selected = scores.new_zeros(scores.shape, dtype=torch.bool)
            selected[prefix] = True
            suffix = torch.nonzero(~selected & torch.isfinite(scores), as_tuple=False).reshape(-1)
            order = torch.cat((prefix, suffix))
            from .autonomous_joint_learning import where_prefix_log_prob
            where_terms.append(-where_prefix_log_prob(scores,order,len(ids),1.,True)/len(ids))
        if valid:
            amount = torch.stack(amount_terms).mean()
            where = torch.stack(where_terms).mean() if where_terms else zero
            # A bounded measured RD advantage over no-op is an importance
            # weight for self-imitation, not a multiplier for zero gradients.
            loss = min(-best_rd, 1.)*(amount+where)
            used, reason = True, 'same_gt_codec_measured_profitable_joint_plan'
        else:
            teacher_available = False
            reason = 'stored_action_not_in_current_candidate_space'
    c,g=float(compression),float(geometry)
    rd=wc*c+wg*max(g-tau,0.)
    replacement=None
    feasible_best_rd = best_rd if teacher_available else float('inf')
    if c < 0. and rd < min(feasible_best_rd,0.) and any(debug['_joint_action']['where_prefix_ids'].values()):
        value=dict(schema='measured_joint_action_v1',compression=c,geometry=g,action=debug['_joint_action'])
        replacement=zlib.compress(json.dumps(value,separators=(',',':')).encode(),1)
    return loss, replacement, dict(used=used,reason=reason,
        teacher_compression=best['compression'] if best else None,
        teacher_geometry=best['geometry'] if best else None,
        stored_new_profitable_plan=replacement is not None)


def bound_replay_memory(memory, maximum_bytes):
    size=sum(len(row.get('best_action',b'')) for row in memory.values())
    for row in memory.values():
        if size <= maximum_bytes:
            break
        size-=len(row.pop('best_action',b''))
    return size
