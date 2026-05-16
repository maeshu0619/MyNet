from models.utils.training.utils import (_adapt_encoder_state_dict_for_sparse_input,
                                         _resolve_amp_dtype,
                                         _warmup_whole_cloud_caches,
                                         _use_memory_safe_loader_workers,
                                         _resolve_training_stage_for_episode,
                                         _stage_loss_factors,
                                         _make_step_cache_key,
                                         _should_log_step,
                                         _downsample_input_batch,
                                         _get_patch_info,
                                         _effective_patch_batch_size,
                                         select_patch_subset_ids,
                                         make_patch_subset_cache_key,
                                         _accumulate_grouped_patch_geometry,
                                         _run_geometry_audit,
                                         _new_metric_sums,
                                         _add_metric_sums,
                                         _surrogate_plot_metrics,
                                         _sync_for_timing,
                                         _prepare_whole_cloud_inputs,
                                         _metric_avgs_to_floats,
                                         _format_metric_summary,
                                         _new_point_edit_sums,
                                         _add_point_edit_sums,
                                         _finalize_point_edit_sums,
                                         _summarize_point_edits,
                                         _cuda_bf16_ops_safe,
                                         _log_grad_flow, 
                                         _format_named_float_map,
                                         _uses_actual_total_bit_objective,
                                         _write_structure_decision_debug,
                                         _compression_stat_qs,
                                         _format_triplet,
                                         _summarize_subtree_octree_stats,)
from models.utils.training.utils import *

def subtree_pcd_setup(pts, args, cache_key, use_cuda):
    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
    input_pcd = _downsample_input_batch(input_pcd, args, cache_key)
    if use_cuda:
        input_pcd = input_pcd.cuda(non_blocking=True)
    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()
    input_xyz = input_pcd[:, :3, :]
    return input_xyz, input_pcd

def split2patch_pcd_setup(pts, args, cache_key, use_cuda):
    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
    input_pcd = _downsample_input_batch(input_pcd, args, cache_key)
    if use_cuda:
        input_pcd = input_pcd.cuda(non_blocking=True)
    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()
    input_xyz = input_pcd[:, :3, :]
    return input_xyz, input_pcd

def ifNot_is_anchor_step(eligible_groups, all_subtree_keys, subtree_index_lists, selected_subtree_keys):
    selected_key_set = set(selected_subtree_keys.detach().cpu().tolist())
    group_source = eligible_groups
    if not group_source:
        group_source = [(int(subtree_key.detach().cpu()), point_idx) for subtree_key, point_idx in zip(all_subtree_keys, subtree_index_lists)]
    selected_groups = [(subtree_key, point_idx) for subtree_key, point_idx in group_source if subtree_key in selected_key_set]
    if not selected_groups and group_source:
        selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]
    if not selected_groups:
        raise RuntimeError("Subtree mode did not select any subtree group.")
    return selected_key_set, group_source, selected_groups

def If_log_this_step(point_idx, selected_groups, is_anchor_step, eligible_groups, all_subtree_keys, subtree_index_lists, selected_subtree_keys, input_xyz, subtree_depth_meta, writer, args):
    if is_anchor_step:
        point_counts = [int(point_idx.numel()) for _, point_idx in (eligible_groups or [])]
        if not point_counts:
            point_counts = [int(input_xyz.shape[-1])]
        stat_groups = eligible_groups or [(0, torch.arange(input_xyz.shape[-1], device=input_xyz.device))]
        loss_scope = "full_cloud_output_vs_full_cloud_input"
    else:
        point_counts = [int(point_idx.numel()) for _, point_idx in selected_groups]
        stat_groups = selected_groups
        loss_scope = "subtree_output_vs_subtree_input"
    mean_points = sum(point_counts) / float(max(len(point_counts), 1))
    octree_stat = _summarize_subtree_octree_stats(input_xyz, stat_groups, args)
    octree_stat_text = ""
    if octree_stat is not None:
        octree_stat_text = (
            f", octree_node[min/mean/max]={octree_stat['node']}, "
            f"octree_single[min/mean/max]={octree_stat['single']}, "
            f"octree_depth[min/mean/max]={octree_stat['depth']}, "
            f"octree_stat_count={int(octree_stat['count'])}"
        )
    writer.write(
        "SubtreeSelection: "
        f"depth={int(subtree_depth_meta['depth'])} "
        f"(base={int(subtree_depth_meta['base_depth'])}, "
        f"range={int(subtree_depth_meta['min_depth'])}-{int(subtree_depth_meta['max_depth'])}, "
        f"uncapped_range={int(subtree_depth_meta.get('uncapped_min_depth', subtree_depth_meta['min_depth']))}-"
        f"{int(subtree_depth_meta.get('uncapped_max_depth', subtree_depth_meta['max_depth']))}, "
        f"curriculum_phase={float(subtree_depth_meta.get('curriculum_phase', 1.0)):.3f}, "
        f"data_max={int(subtree_depth_meta['data_max_depth'])}, "
        f"percent_mode={bool(subtree_depth_meta.get('depth_percent_curriculum', False))}, "
        f"percent_range={subtree_depth_meta.get('depth_percent_range', 'n/a')}), "
        f"selected={selected_subtree_count}/{eligible_subtree_count} eligible "
        f"(total={total_subtree_count}, min_points={min_subtree_points}), "
        f"points[min/mean/max]={min(point_counts)}/{mean_points:.1f}/{max(point_counts)}, "
        f"anchor_refresh={bool(is_anchor_step)}({anchor_reason}), "
        f"loss_scope={loss_scope}"
        f"{octree_stat_text}"
    )
    return point_counts, stat_groups, loss_scope, mean_points, octree_stat, octree_stat_text

def loss_setup(input_xyz, args):
    L_geom = input_xyz.new_zeros(())
    L_com = input_xyz.new_zeros(())
    L_attr = input_xyz.new_zeros(())
    L_policy = input_xyz.new_zeros(())
    L_actuator = input_xyz.new_zeros(())
    Lp_out = input_xyz.new_zeros(())
    La_fit = input_xyz.new_zeros(())
    La_rep = input_xyz.new_zeros(())
    loss_bit = input_xyz.new_zeros(())
    loss_single = input_xyz.new_zeros(())
    loss_nodes = input_xyz.new_zeros(())
    gen_xyz = None
    final_w = None
    out_label = None
    return L_geom, L_com, L_attr, L_policy, L_actuator, Lp_out, La_fit, La_rep, loss_bit, loss_single, loss_nodes, gen_xyz, final_w, out_label