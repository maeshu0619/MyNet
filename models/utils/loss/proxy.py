import torch


class ProxyCompressionLossMixin:
    def _compression_terms_from_proxy(
        self,
        out,
        bit_ref,
        nodes_ref,
        single_ref,
        args,
        gen_point_count,
        gt_point_count,
    ):
        bit_ref_metric = self._metric_value(
            float(bit_ref),
            own_point_count=gt_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        nodes_ref_metric = self._metric_value(
            float(nodes_ref),
            own_point_count=gt_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        single_ref_metric = self._metric_value(
            float(single_ref),
            own_point_count=gt_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )

        bit = out["rate_entropy"]
        nodes = out["soft_node_count"]
        single = out["soft_single_child_count"]

        bit_metric = self._metric_value(
            bit,
            own_point_count=gen_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        nodes_metric = self._metric_value(
            nodes,
            own_point_count=gen_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        single_metric = self._metric_value(
            single,
            own_point_count=gen_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )

        struct_ref_min = 1.0 if self._compression_rate_metric(args) == "total_bits" else 1e-12
        L_bit = self._relative_percent(bit_metric, bit_ref_metric)
        L_nodes = self._relative_percent(nodes_metric, nodes_ref_metric, ref_min=struct_ref_min)
        L_single = self._relative_percent(single_metric, single_ref_metric, ref_min=struct_ref_min)
        L_com = (
            float(getattr(args, "proxy_lambda_entropy", 1.0)) * L_bit
            + float(getattr(args, "proxy_lambda_node_count", 1.0)) * L_nodes
            + float(getattr(args, "proxy_lambda_single_child", 1.0)) * L_single
        )
        return L_com, L_bit, L_nodes, L_single

    def _get_compression_loss_proxy(
        self,
        args,
        gen_xyz,
        gt_xyz,
        final_w,
        cache_key=None,
        run_grad_probe=True,
        actual_gen_xyz=None,
        subtree_tree=None,
        full_octree_context=None,
        octree_input_mode="auto",
    ):
        self._ensure_rate_proxy_device(gen_xyz.device)
        uses_subtree_tree = subtree_tree is not None
        uses_full_context = full_octree_context is not None
        requested_mode = str(octree_input_mode or "auto").strip().lower()
        if requested_mode == "prebuilt_subtree_tree" and not uses_subtree_tree:
            raise ValueError("octree_input_mode=prebuilt_subtree_tree requires subtree_tree in compression proxy loss.")
        if uses_subtree_tree:
            proxy_input_mode = "prebuilt_subtree_tree"
            fallback_reason = ""
        elif requested_mode == "full_cloud":
            proxy_input_mode = "full_cloud"
            fallback_reason = ""
        else:
            proxy_input_mode = "local_recomputed"
            fallback_reason = "missing_subtree_tree"
        cached_gt = self._get_cached_gt(cache_key, gen_xyz.device)
        if cached_gt is None:
            self.warmup_gt_cache(
                gt_xyz,
                cache_key=cache_key,
                subtree_tree=subtree_tree,
                full_octree_context=full_octree_context,
                octree_input_mode=octree_input_mode,
            )
            cached_gt = self._get_cached_gt(cache_key, gen_xyz.device)
        if cached_gt is None:
            with self._compression_autocast_ctx(gen_xyz.device):
                out_gt, bit_gt, stats_gt = self.rate_proxy.forward_hard_only(
                    gen_xyz=gt_xyz.to(torch.float32),
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
            cached_gt = {
                "rate_gt": self._scalar(out_gt["rate_total"]),
                "single_gt": self._scalar(out_gt["soft_single_child_count"]),
                "nodes_gt": self._scalar(out_gt["soft_node_count"]),
                "bit_gt": self._scalar(bit_gt),
                "point_count_gt": int(gt_xyz.shape[-1]),
                "stats_gt": {k: self._scalar(v) for k, v in stats_gt.items()},
            }
        bit_gt = cached_gt["bit_gt"]
        stats_gt = cached_gt["stats_gt"]
        nodes_gt = cached_gt["nodes_gt"]
        single_gt = cached_gt["single_gt"]
        gt_point_count = int(cached_gt.get("point_count_gt", gt_xyz.shape[-1]))
        gen_point_count = int(gen_xyz.shape[-1])
        mode = self._discrete_loss_mode(args)
        if mode == "hard":
            final_w = None
        use_weighted_forward = mode in {"weighted_soft", "soft", "legacy"} and final_w is not None
        use_ste_hard = mode in {"ste_hard", "hard_ste"} and final_w is not None

        if use_ste_hard:
            with self._compression_autocast_ctx(gen_xyz.device):
                out_gen, out_surrogate, stats_gen = self.rate_proxy.forward_ste_hard_pair(
                    gen_xyz=gen_xyz.to(torch.float32),
                    final_w=final_w.to(torch.float32),
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
        else:
            with self._compression_autocast_ctx(gen_xyz.device):
                out_gen, _, stats_gen = self.rate_proxy(
                    gen_xyz=gen_xyz.to(torch.float32),
                    final_w=final_w.to(torch.float32) if use_weighted_forward else None,
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )

        L_com_forward, L_bit_forward, L_nodes_forward, L_single_forward = self._compression_terms_from_proxy(
            out_gen,
            bit_ref=bit_gt,
            nodes_ref=nodes_gt,
            single_ref=single_gt,
            args=args,
            gen_point_count=gen_point_count,
            gt_point_count=gt_point_count,
        )

        L_com = L_com_forward
        L_bit_objective = L_bit_forward
        L_nodes_objective = L_nodes_forward
        L_single_objective = L_single_forward
        sparse_terms = self._sparsepcgc_aux_feature_terms(args, gen_xyz, gt_xyz, final_w)
        L_sparse_objective = sparse_terms["loss"]
        L_com = L_com + L_sparse_objective
        sparse_aux_uses_voxel_state = sparse_terms.get("sparsepcgc_aux_uses_actuator_voxel_state", None)
        sparse_aux_recomputed = sparse_terms.get("sparsepcgc_aux_final_voxel_recomputed_from_pts_out", None)
        if use_ste_hard:
            L_com_surrogate, L_bit_surrogate, L_nodes_surrogate, L_single_surrogate = self._compression_terms_from_proxy(
                out_surrogate,
                bit_ref=bit_gt,
                nodes_ref=nodes_gt,
                single_ref=single_gt,
                args=args,
                gen_point_count=gen_point_count,
                gt_point_count=gt_point_count,
            )
            L_com = self._compose_discrete_loss(L_com_forward, L_com_surrogate, args)
            L_bit_objective = self._compose_discrete_loss(L_bit_forward, L_bit_surrogate, args)
            L_nodes_objective = self._compose_discrete_loss(L_nodes_forward, L_nodes_surrogate, args)
            L_single_objective = self._compose_discrete_loss(L_single_forward, L_single_surrogate, args)
            L_sparse_objective = sparse_terms["loss"]
            L_com = L_com + L_sparse_objective

        if args.trainORtest == "test":
            self.writer.write(f"=== Compression Stats ===")
            self.writer.write(f"bit                         : {stats_gt['bit']} -> {stats_gen['bit']}")
            self.writer.write(f"bpp                         : {stats_gt['bpp']} -> {stats_gen['bpp']}")
            self.writer.write(f"bpn                         : {stats_gt['bpn']} -> {stats_gen['bpn']}")
            self.writer.write(f"single child node           : {stats_gt['single']} -> {stats_gen['single']}")
            self.writer.write(f"num of nodes                : {stats_gt['node']} -> {stats_gen['node']}")
            self.writer.write(f"num of points               : {gt_xyz.shape[2]} -> {gen_xyz.shape[2]}")

        rate_gt = out_gen["rate_entropy"].new_tensor(cached_gt["rate_gt"])
        single_gt_t = out_gen["soft_single_child_count"].new_tensor(single_gt)
        nodes_gt_t = out_gen["soft_node_count"].new_tensor(nodes_gt)

        rate_gen = out_gen["rate_entropy"].detach()
        single_gen = out_gen["soft_single_child_count"].detach()
        nodes_gen = out_gen["soft_node_count"].detach()

        loss_bit = self._relative_percent(
            self._metric_value(rate_gen, gen_point_count, gt_point_count, args),
            self._metric_value(float(rate_gt), gt_point_count, gt_point_count, args),
        )
        struct_ref_min = 1.0 if self._compression_rate_metric(args) == "total_bits" else 1e-12
        loss_single = self._relative_percent(
            self._metric_value(single_gen, gen_point_count, gt_point_count, args),
            self._metric_value(float(single_gt_t), gt_point_count, gt_point_count, args),
            ref_min=struct_ref_min,
        )
        loss_nodes = self._relative_percent(
            self._metric_value(nodes_gen, gen_point_count, gt_point_count, args),
            self._metric_value(float(nodes_gt_t), gt_point_count, gt_point_count, args),
            ref_min=struct_ref_min,
        )
        loss_total_bit = self._relative_percent(rate_gen, float(rate_gt))
        loss_bpp = self._relative_percent(
            rate_gen / self._positive_count(gen_point_count),
            float(rate_gt) / self._positive_count(gt_point_count),
        )
        self.last_compression_debug = {
            "metric": self._compression_rate_metric(args),
            "total_bit": self._scalar(loss_total_bit),
            "compression_objective": self._scalar(L_com.detach()),
            "bpp": self._scalar(loss_bpp),
            "gt_points": gt_point_count,
            "gen_points": gen_point_count,
            "gt_bit_abs": float(stats_gt.get("bit", 0.0)),
            "gen_bit_abs": self._scalar(stats_gen.get("bit", 0.0)),
            "gt_bpp_abs": float(stats_gt.get("bpp", 0.0)),
            "gen_bpp_abs": self._scalar(stats_gen.get("bpp", 0.0)),
            "gt_bpn_abs": float(stats_gt.get("bpn", 0.0)),
            "gen_bpn_abs": self._scalar(stats_gen.get("bpn", 0.0)),
            "gt_single_abs": float(stats_gt.get("single", 0.0)),
            "gen_single_abs": self._scalar(stats_gen.get("single", 0.0)),
            "gt_node_abs": float(stats_gt.get("node", 0.0)),
            "gen_node_abs": self._scalar(stats_gen.get("node", 0.0)),
            "rate_proxy_before": float(stats_gt.get("bit", 0.0)),
            "rate_proxy_after": self._scalar(stats_gen.get("bit", 0.0)),
            "rate_proxy_delta": self._scalar(loss_total_bit),
            "actual_value_is_fresh": False,
            "actual_value_source": "proxy",
            "node_delta": self._scalar(nodes_gen - nodes_gt_t),
            "single_delta": self._scalar(single_gen - single_gt_t),
            "sparsepcgc_aux_loss": self._scalar(sparse_terms["loss"].detach()),
            "sparsepcgc_active_coord_loss": self._scalar(sparse_terms["active"].detach()),
            "sparsepcgc_isolated_proxy_loss": self._scalar(sparse_terms["single"].detach()),
            "sparsepcgc_entropy_proxy_loss": self._scalar(sparse_terms["entropy"].detach()),
            "sparsepcgc_density_proxy_loss": self._scalar(sparse_terms["density"].detach()),
            "compression_proxy_input_mode": proxy_input_mode,
            "compression_proxy_uses_subtree_tree": bool(uses_subtree_tree),
            "compression_proxy_uses_full_context": bool(uses_full_context),
            "rate_proxy_source": proxy_input_mode,
            "L_com_source": proxy_input_mode,
            "loss_nodes_source": proxy_input_mode,
            "loss_single_source": proxy_input_mode,
            "compression_proxy_fallback_reason": fallback_reason,
            "prebuilt_node_count_used": self._scalar(stats_gen.get("prebuilt_node_count", 0.0)),
            "prebuilt_single_child_count_used": self._scalar(stats_gen.get("prebuilt_single_child_count", 0.0)),
            "rate_proxy_node_count_used": self._scalar(stats_gen.get("node", 0.0)),
            "loss_nodes_node_count_used": self._scalar(stats_gen.get("node", 0.0)),
            "sparsepcgc_aux_uses_actuator_voxel_state": bool(
                float(sparse_aux_uses_voxel_state.detach().cpu()) > 0.5
            ) if torch.is_tensor(sparse_aux_uses_voxel_state) else False,
            "sparsepcgc_aux_final_voxel_recomputed_from_pts_out": bool(
                float(sparse_aux_recomputed.detach().cpu()) > 0.5
            ) if torch.is_tensor(sparse_aux_recomputed) else True,
        }
        debug_gen_xyz = gen_xyz if actual_gen_xyz is None else actual_gen_xyz
        self._maybe_update_sparsepcgc_debug(
            args,
            self.last_compression_debug,
            gen_xyz=debug_gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
        )
        self._store_compression_terms(
            main=L_bit_objective,
            bit=L_bit_objective,
            single=L_single_objective,
            node=L_nodes_objective,
            bpn=gen_xyz.new_zeros(()),
            sparsepcgc=L_sparse_objective,
            objective=L_com,
            backend="proxy",
        )

        if run_grad_probe:
            self._log_compression_grad_probe(args, "proxy", L_com, gen_xyz)

        return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt
