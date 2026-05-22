                elif args.split2patch:
                    optimizer.zero_grad(set_to_none=True)
                    patch_info = get_patch_info(input_pcd, args, cache_key, patch_info_cache)
                    total_patch_count = int(patch_info["num_patches"])
                    subset_enabled = bool(getattr(args, "train_patch_subset_enable", False))
                    selected_patch_ids = torch.arange( total_patch_count, device=patch_info["patch_xyz"].device, dtype=torch.long)
                    if subset_enabled:
                        is_anchor_step, _ = should_use_full_cloud_anchor( args, global_step=global_train_step, cache_key=cache_key)
                        if not is_anchor_step:
                            selected_patch_ids = select_patch_subset_ids(patch_info, global_train_step, args)
                    selected_patch_count = int(selected_patch_ids.numel())
                    subset_step = bool( subset_enabled and (not is_anchor_step) and selected_patch_count < total_patch_count)
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    pb = effective_patch_batch_size( args, patch_count=selected_patch_count, patch_size=args.num_points, is_train=True, writer=writer)
                    patch_outputs = []
                    patch_count = selected_patch_count
                    geom_weight_sum = 0.0
                    L_geom = input_pcd.new_zeros(())
                    L_attr = input_pcd.new_zeros(())
                    L_policy = input_pcd.new_zeros(())
                    L_actuator = input_pcd.new_zeros(())
                    Lp_out = input_pcd.new_zeros(())
                    La_fit = input_pcd.new_zeros(())
                    La_rep = input_pcd.new_zeros(())
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with autocast_ctx:
                        prev_patch_geom_log = getattr(args, "_log_this_step", True)
                        args._log_this_step = False
                        try:
                            for i in range(0, patch_count, pb):
                                chunk_patch_ids = selected_patch_ids[i:i+pb]
                                chunk_patch_ids_list = chunk_patch_ids.detach().cpu().tolist()
                                patch_xyz = patch_info["patch_xyz"].index_select(0, chunk_patch_ids)
                                patch_attr = patch_info["patch_attr"].index_select(0, chunk_patch_ids)
                                patch_centroid = patch_info["patch_centroid"].index_select(0, chunk_patch_ids)
                                patch_scale = patch_info["patch_scale"].index_select(0, chunk_patch_ids)
                                patch_cache_keys = [ f"{cache_key}|patch={patch_id}" for patch_id in chunk_patch_ids_list]
                                ( gen_chunk, L_attr_chunk, L_policy_chunk, L_actuator_chunk, final_w_chunk, Lp_out_chunk, La_fit_chunk, La_rep_chunk, _, patch_meta_chunk) = model.forward( patch_xyz, patch_attr, cache_key=patch_cache_keys, return_patch_meta=True, coord_scale=patch_scale, return_attr_output=False)
                                if detail_log_this_step:
                                    base_model = model.module if hasattr(model, "module") else model
                                    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
                                gen_chunk = denormalize_patch_output( gen_chunk, patch_centroid, patch_scale)
                                chunk_size = patch_xyz.shape[0]
                                geom_groups = {}
                                L_attr = L_attr + L_attr_chunk * chunk_size
                                L_policy = L_policy + L_policy_chunk * chunk_size
                                L_actuator = L_actuator + L_actuator_chunk * chunk_size
                                Lp_out = Lp_out + Lp_out_chunk * chunk_size
                                La_fit = La_fit + La_fit_chunk * chunk_size
                                La_rep = La_rep + La_rep_chunk * chunk_size

                                for local_idx in range(chunk_size):
                                    patch_id = int(chunk_patch_ids_list[local_idx])
                                    patch_input_idx = patch_info["patch_input_idx"][patch_id]
                                    owned_input_mask = patch_info["owned_input_mask"][patch_id]
                                    anchor_idx_local = patch_meta_chunk["anchor_idx_local"][local_idx].clamp_(0, patch_input_idx.shape[0] - 1)
                                    valid_mask = patch_meta_chunk["output_valid_mask"][local_idx]
                                    owned_output_mask = owned_input_mask.index_select(0, anchor_idx_local)
                                    select_mask = valid_mask & owned_output_mask
                                    selected_pts = gen_chunk[local_idx, :, select_mask]
                                    selected_w = None
                                    if final_w_chunk is not None:
                                        selected_w = final_w_chunk[local_idx, :, select_mask]
                                    represented_owned_mask = torch.zeros_like(owned_input_mask)
                                    if select_mask.any():
                                        represented_owned_mask[anchor_idx_local[select_mask]] = True
                                    missing_owned_mask = owned_input_mask & (~represented_owned_mask)
                                    fallback_pts = None
                                    fallback_w = None
                                    if missing_owned_mask.any():
                                        patch_input_xyz_world = (patch_info["patch_centroid"][patch_id:patch_id+1] + patch_info["patch_xyz"][patch_id:patch_id+1] * patch_info["patch_scale"][patch_id:patch_id+1])
                                        fallback_pts = patch_input_xyz_world[0, :, missing_owned_mask]
                                        if final_w_chunk is not None:
                                            fallback_w = final_w_chunk.new_ones((1, int(missing_owned_mask.sum().item())))

                                    owned_local_idx = torch.nonzero(owned_input_mask, as_tuple=False).flatten()
                                    owned_global_idx = None
                                    owned_out_label = None
                                    if owned_local_idx.numel() > 0:
                                        owned_global_idx = patch_input_idx.index_select(0, owned_local_idx)
                                        if patch_meta_chunk["out_label"] is not None:
                                            owned_out_label = patch_meta_chunk["out_label"][local_idx, owned_local_idx]

                                    if valid_mask.any():
                                        gen_patch_valid = gen_chunk[local_idx:local_idx+1, :3, valid_mask]
                                        if str(getattr(args, "discretelossmode", "hard")).strip().lower() == "hard":
                                            final_w_owned = None
                                        else:
                                            final_w_owned = None if final_w_chunk is None else final_w_chunk[local_idx:local_idx+1, :, valid_mask]

                                        gt_patch_owned = input_pcd[:, :3, patch_input_idx[owned_input_mask]].contiguous()
                                        local_weight = float(max(int(owned_input_mask.sum().item()), 1))
                                        can_batch_geom = ( owned_out_label is None or int(torch.count_nonzero(owned_out_label).detach().cpu()) == 0)
                                        if can_batch_geom:
                                            geom_key = ( int(gen_patch_valid.shape[-1]), int(gt_patch_owned.shape[-1]), final_w_owned is not None)
                                            group = geom_groups.get(geom_key)
                                            if group is None:
                                                group = { "gen": [], "gt": [], "final_w": [] if final_w_owned is not None else None, "weight": 0.0}
                                                geom_groups[geom_key] = group
                                            group["gen"].append(gen_patch_valid)
                                            group["gt"].append(gt_patch_owned)
                                            if final_w_owned is not None:
                                                group["final_w"].append(final_w_owned)
                                            group["weight"] += local_weight
                                        else:
                                            out_label_owned = owned_out_label.unsqueeze(0)
                                            L_geom = L_geom + loss.get_geometry_loss(
                                                args,
                                                gen_pts=gen_patch_valid,
                                                gt_pts=gt_patch_owned,
                                                final_w=final_w_owned,
                                                out_label=out_label_owned,
                                            ) * local_weight
                                            geom_weight_sum += local_weight
                                    patch_outputs.append( { "patch_id": patch_id, "selected_pts": selected_pts, "selected_w": selected_w, "fallback_pts": fallback_pts, "fallback_w": fallback_w, "owned_global_idx": owned_global_idx, "owned_out_label": owned_out_label, "patch_meta": { "anchor_idx_local": anchor_idx_local, "output_valid_mask": valid_mask, "out_label": None if patch_meta_chunk["out_label"] is None else patch_meta_chunk["out_label"][local_idx]}})
                                geom_chunk, geom_chunk_weight = accumulate_grouped_patch_geometry( geom_groups, loss, args)
                                if geom_chunk is not None and geom_chunk_weight > 0.0:
                                    L_geom = L_geom + geom_chunk
                                    geom_weight_sum += geom_chunk_weight
                        finally:
                            args._log_this_step = prev_patch_geom_log
                        if subset_step:
                            gen_pts, compression_gt_pts, final_w, out_label = merge_patch_subset_outputs( patch_info, patch_outputs, input_pcd=input_pcd, device=input_pcd.device, dtype=input_pcd.dtype)
                            compression_cache_key = make_patch_subset_cache_key( cache_key, selected_patch_ids, total_patch_count=total_patch_count)
                        else:
                            gen_pts, final_w, out_label = merge_patch_outputs( patch_info, patch_outputs, device=input_pcd.device, dtype=input_pcd.dtype)
                            compression_gt_pts = input_xyz

                        norm = float(max(patch_count, 1))
                        L_attr = L_attr / norm
                        L_policy = L_policy / norm
                        L_actuator = L_actuator / norm
                        Lp_out = Lp_out / norm
                        La_fit = La_fit / norm
                        La_rep = La_rep / norm
                        if geom_weight_sum > 0:
                            L_geom = L_geom / geom_weight_sum
                    gen_xyz = gen_pts[:, :3, :]
                    train_edit_stats = summarize_point_edits( input_xyz=compression_gt_pts[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args)

                else:
                    optimizer.zero_grad(set_to_none=True)
                    args._log_this_step = bool(getattr(args, "verbose_step_logs", False) and detail_log_this_step)
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with autocast_ctx:
                        gen_patches, L_attr, L_policy, L_actuator, final_w, Lp_out, La_fit, La_rep, out_label = model.forward( patches, None, cache_key=cache_key, coord_scale=fd_xyz, return_attr_output=False)
                    if detail_log_this_step:
                        base_model = model.module if hasattr(model, "module") else model
                        encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))

                    # 元スケールに戻す
                    gen_xyz = centroid_xyz + gen_patches[:, :3, :] * fd_xyz
                    gen_pts = gen_xyz.contiguous()
                    gen_xyz = gen_pts[:, :3, :]
                    train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args)
                    L_geom = None




# 以下は損失の計算のautocast_ctxの下
                    if subtree_mode:
                        pass
                    elif args.split2patch:
                        if compute_compression:
                            L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss( args, gen_xyz=compression_gen_xyz, gt_xyz=compression_gt_pts[:, :3, :], final_w=final_w_for_loss, cache_key=compression_cache_key, refresh_actual_gen=refresh_actual_gen, actual_gen_xyz=gen_xyz)
                        else:
                            zero = gen_xyz.new_zeros(())
                            L_com = zero
                            loss_bit = zero
                            loss_single = zero
                            loss_nodes = zero
                            loss.last_compression_debug = {}
                            loss.last_compression_terms = {}
                    else:
                        L_geom = loss.get_geometry_loss( args, gen_pts=gen_xyz, gt_pts=input_xyz, final_w=final_w_for_loss, out_label=out_label)
                        if compute_compression:
                            L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss( args, gen_xyz=compression_gen_xyz, gt_xyz=input_xyz[:, :3, :], final_w=final_w_for_loss, cache_key=cache_key, refresh_actual_gen=refresh_actual_gen, actual_gen_xyz=gen_xyz)
                        else:
                            zero = gen_xyz.new_zeros(())
                            L_com = zero
                            loss_bit = zero
                            loss_single = zero
                            loss_nodes = zero
                            loss.last_compression_debug = {}
                            loss.last_compression_terms = {}