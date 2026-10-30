"""Small truthful audit for the independent joint policy's active fields."""
import json


def compact_autonomous_audit(args, plan, compression, elapsed):
    fields = ('current_compression_value','current_geometry_value','advantage',
              'actual_plan_replay','actual_plan_replay_bytes','actual_plan_replay_loss')
    policy = {key:compression.get('den6_online_policy_'+key) for key in fields}
    row = dict(schema='autonomous_joint_context_v3',
        frame=str(getattr(args,'_current_input_file','')),
        global_step=int(getattr(args,'_global_train_step',0)),
        policy=policy,counts=plan.get('selected_counts'),ratios=plan.get('selected_amount_ratios'),
        amount_bins=plan.get('joint_amount_bins_by_operation'),
        no_op=plan.get('no_op_selected'),exploration=plan.get('exploration_multiplier'),
        recovery_probe=plan.get('autonomous_noop_recovery'),
        uniform_fraction=plan.get('joint_amount_uniform_fraction'),
        structured_where=plan.get('structured_where_coefficients'),
        amount_log_prob=plan.get('joint_amount_log_prob'),where_log_prob=plan.get('joint_where_log_prob'),
        decision_capture_l2=compression.get('den6_policy_only_capture_grad_norm'),
        context=getattr(args,'_autonomous_native_context_debug',None),
        step_before_audit=float(elapsed),actual_seconds=compression.get('actual_encode_time_total'))
    return 'AutonomousLearningAudit: '+json.dumps(row,separators=(',',':'),default=str)
