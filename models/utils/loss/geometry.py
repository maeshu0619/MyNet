from contextlib import nullcontext

import torch

from .utils_loss import (
    chamfer_l2_loss,
    chamfer_l2_loss_and_weight_surrogate,
    chamfer_dist,
    compute_d2_psnr,
    remove_outlier_points_by_label,
)


class GeometryLossMixin:
    def _set_geometry_debug(self, **kwargs):
        self.last_geometry_debug = kwargs

    @staticmethod
    def _geometry_autocast_ctx(tensor):
        if torch.is_tensor(tensor) and tensor.device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    @staticmethod
    def _as_geometry_scalar(value, reference):
        if not torch.is_tensor(value):
            return reference.new_zeros(())
        value = value.to(device=reference.device, dtype=reference.dtype)
        return value.reshape(()) if value.numel() == 1 else value.mean()

    @staticmethod
    def _debug_scalar(value):
        if not torch.is_tensor(value):
            return None
        try:
            return float(value.detach().float().mean().cpu())
        except Exception:
            return None

    @staticmethod
    def _debug_requires_grad(value):
        return bool(torch.is_tensor(value) and value.requires_grad)

    @staticmethod
    def _fit_proxy_loss(gen_pts_f, gt_pts_f):
        gen = gen_pts_f.transpose(1, 2).contiguous()
        gt = gt_pts_f.transpose(1, 2).contiguous()
        _, dist2, _, _ = chamfer_dist(gen, gt)
        return dist2.mean()

    @staticmethod
    def _sorted_membership(source_keys, target_keys):
        """sourceの各keyがtargetに存在するかをTorch 1.11互換で返す。"""
        if target_keys.numel() == 0:
            return torch.zeros_like(source_keys, dtype=torch.bool)
        target_sorted = torch.sort(target_keys).values
        positions = torch.searchsorted(target_sorted, source_keys)
        in_bounds = positions < target_sorted.numel()
        safe = positions.clamp(max=max(int(target_sorted.numel()) - 1, 0))
        return in_bounds & target_sorted.index_select(0, safe).eq(source_keys)

    @staticmethod
    def _voxel_keys(initial_rows, final_rows):
        combined = torch.cat([initial_rows, final_rows], dim=0)
        minimum = combined.amin(dim=0)
        span = (combined.amax(dim=0) - minimum + 1).to(dtype=torch.int64).clamp_min(1)

        def encode(rows):
            shifted = rows.to(dtype=torch.int64) - minimum
            return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

        return encode(initial_rows), encode(final_rows)

    @staticmethod
    def _voxel_xyz(rows, step, offset, reference):
        xyz = rows.to(device=reference.device, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        return xyz * step.to(device=reference.device, dtype=torch.float32) + offset.to(
            device=reference.device, dtype=torch.float32
        )

    def _exact_sparse_edit_chamfer(self, args, gen_pts_f, gt_pts_f, final_w_f):
        """不変Voxelの距離0を省略し、通常Chamferと同じ値を差分集合だけで計算する。"""
        if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() not in {
            "ana_den6_online", "network_only_codec_policy", "network_k_proposal_policy", "single_plan_student"
        }:
            return None
        if gen_pts_f.shape[0] != 1 or gt_pts_f.shape[0] != 1:
            return None
        state = getattr(args, "_last_actuator_voxel_state", None)
        if not isinstance(state, dict) or not bool(state.get("voxel_edit_state_enabled", False)):
            return None
        initial = state.get("initial_voxel_coords", state.get("voxel_edit_initial_coords"))
        final = state.get("final_voxel_coords", state.get("voxel_edit_final_coords"))
        meta = state.get("voxel_restore_meta", {})
        if not torch.is_tensor(initial) or not torch.is_tensor(final) or not isinstance(meta, dict):
            return None
        if initial.ndim != 3 or final.ndim != 3 or initial.shape[0] != 1 or final.shape[0] != 1:
            return None
        valid = state.get("final_voxel_valid_mask", state.get("voxel_edit_valid_mask", None))
        initial_rows = initial[0].transpose(0, 1).detach().to(device=gen_pts_f.device, dtype=torch.int64)
        final_rows = final[0].transpose(0, 1).detach().to(device=gen_pts_f.device, dtype=torch.int64)
        if torch.is_tensor(valid):
            valid_b = valid[0].to(device=gen_pts_f.device, dtype=torch.bool).reshape(-1)
            if valid_b.numel() == final_rows.shape[0]:
                final_rows = final_rows[valid_b]
        if (
            int(initial_rows.shape[0]) != int(gt_pts_f.shape[-1])
            or int(final_rows.shape[0]) != int(gen_pts_f.shape[-1])
            or initial_rows.numel() == 0
            or final_rows.numel() == 0
        ):
            return None

        step = meta.get(
            "effective_qs_tensor",
            meta.get("global_qs", state.get("voxel_step", None)),
        )
        offset = meta.get(
            "global_offset_tensor",
            meta.get("global_offset", state.get("voxel_offset", None)),
        )
        if not torch.is_tensor(step):
            step = gen_pts_f.new_tensor(float(step if step is not None else 1.0)).reshape(1, 1, 1)
        else:
            step = step.reshape(-1, 1, 1)
        if not torch.is_tensor(offset):
            offset = gen_pts_f.new_zeros((1, 3, 1)) if offset is None else gen_pts_f.new_tensor(offset).reshape(1, 3, 1)
        else:
            offset = offset.reshape(1, 3, 1)

        initial_keys, final_keys = self._voxel_keys(initial_rows, final_rows)
        removed_mask = ~self._sorted_membership(initial_keys, final_keys)
        added_mask = ~self._sorted_membership(final_keys, initial_keys)
        removed_rows = initial_rows[removed_mask]
        added_rows = final_rows[added_mask]
        initial_xyz = self._voxel_xyz(initial_rows, step, offset, gt_pts_f)
        final_xyz = self._voxel_xyz(final_rows, step, offset, gen_pts_f)
        # Voxel状態とLoss入力の順序・座標が一致するときだけ高速経路を使う。
        # これにより値だけでなく、変更点とその最近傍に対する位置勾配も
        # 通常のfull-cloud Chamferと同じ入力Tensorへ流せる。
        if not torch.allclose(initial_xyz, gt_pts_f, rtol=1e-5, atol=1e-4):
            return None
        if not torch.allclose(final_xyz, gen_pts_f, rtol=1e-5, atol=1e-4):
            return None
        initial_xyz = gt_pts_f
        final_xyz = gen_pts_f

        zero = gen_pts_f.new_zeros(())
        added_dist = gen_pts_f.new_empty((0,))
        removed_dist = gen_pts_f.new_empty((0,))
        if added_rows.numel() > 0:
            added_xyz = gen_pts_f[:, :, added_mask]
            added_dist, _, _, _ = chamfer_dist(
                added_xyz.transpose(1, 2).contiguous(),
                initial_xyz.transpose(1, 2).contiguous(),
            )
            added_dist = added_dist.reshape(-1)
        if removed_rows.numel() > 0:
            removed_xyz = gt_pts_f[:, :, removed_mask]
            removed_dist, _, _, _ = chamfer_dist(
                removed_xyz.transpose(1, 2).contiguous(),
                final_xyz.transpose(1, 2).contiguous(),
            )
            removed_dist = removed_dist.reshape(-1)

        gen_count = max(int(final_rows.shape[0]), 1)
        gt_count = max(int(initial_rows.shape[0]), 1)
        added_sum = added_dist.sum() if added_dist.numel() > 0 else zero
        removed_sum = removed_dist.sum() if removed_dist.numel() > 0 else zero
        hard = added_sum / float(gen_count) + removed_sum / float(gt_count)
        fit = removed_sum / float(gt_count)

        if final_w_f is None:
            surrogate = hard
            weighted = hard
        else:
            weights = final_w_f.reshape(-1).clamp(0.0, 1.0).to(dtype=hard.dtype)
            if weights.numel() != final_rows.shape[0]:
                return None
            added_weighted = (
                (added_dist.detach() * weights[added_mask]).sum()
                if added_dist.numel() > 0
                else zero
            )
            surrogate = added_weighted / weights.sum().clamp_min(1e-12) + fit.detach()
            weighted = surrogate
        return {
            "hard": hard,
            "surrogate": surrogate,
            "weighted": weighted,
            "fit": fit,
            "removed_count": int(removed_rows.shape[0]),
            "added_count": int(added_rows.shape[0]),
        }

    def _soft_actuator_geometry_proxy(self, args, reference):
        if getattr(args, "trainORtest", "train") != "train":
            return reference.new_zeros(())
        terms = getattr(args, "_last_actuator_soft_terms", {}) or {}
        if not isinstance(terms, dict):
            return reference.new_zeros(())

        add_proxy = (
            self._as_geometry_scalar(terms.get("add_shape_guard"), reference)
            + 0.1 * self._as_geometry_scalar(terms.get("add_direction_ce"), reference)
            + 0.1 * self._as_geometry_scalar(terms.get("add_prob_mean"), reference)
        )
        prune_soft_geom = self._as_geometry_scalar(terms.get("prune_soft_geom"), reference)
        prune_proxy = prune_soft_geom + 0.1 * self._as_geometry_scalar(terms.get("drop_shape_guard"), reference)
        move_proxy = 0.1 * self._as_geometry_scalar(terms.get("move_direction_ce"), reference)
        proxy = (
            float(getattr(args, "geometry_soft_add_proxy_weight", 1e-3)) * add_proxy
            + float(getattr(args, "geometry_soft_prune_proxy_weight", 1.0)) * prune_proxy
            + float(getattr(args, "geometry_soft_move_proxy_weight", 1e-3)) * move_proxy
        )
        proxy = torch.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            setattr(
                args,
                "_soft_proxy_geom_debug",
                {
                    "soft_proxy_geom_requires_grad": self._debug_requires_grad(proxy),
                    "soft_proxy_prune_geom_requires_grad": self._debug_requires_grad(prune_proxy),
                    "drop_prob_requires_grad": self._debug_requires_grad(terms.get("drop_prob")),
                    "keep_prob_requires_grad": self._debug_requires_grad(terms.get("keep_prob")),
                    "drop_prob_mean": self._debug_scalar(terms.get("drop_prob_mean")),
                    "drop_prob_min": self._debug_scalar(terms.get("drop_prob_min")),
                    "drop_prob_max": self._debug_scalar(terms.get("drop_prob_max")),
                    "drop_prob_proxy_mean": self._debug_scalar(terms.get("drop_prob_proxy_mean")),
                    "drop_prob_proxy_min": self._debug_scalar(terms.get("drop_prob_proxy_min")),
                    "drop_prob_proxy_max": self._debug_scalar(terms.get("drop_prob_proxy_max")),
                    "keep_prob_mean": self._debug_scalar(terms.get("keep_prob")),
                    "prune_soft_geom_value": self._debug_scalar(prune_proxy),
                },
            )
        except Exception:
            pass
        return proxy

    def get_geometry_loss(self, args, gen_pts, gt_pts, final_w=None, out_label=None):
        use_torch_d2 = args.trainORtest == "train"
        audit_enabled = self._should_verbose_step(args) or args.trainORtest != "train"
        # Phase2/Phase6では out_label がdictになる。
        # geometry lossで使う外れ点ラベルは従来互換の point_label だけである。
        if isinstance(out_label, dict):
            out_label = out_label.get("point_label", None)
        # Networkの現行full-cloud経路は互換用point_labelを常に全ゼロで返す。
        # 全ゼロなら外れ点除去の前後は完全に同一なので、label無しとして扱い、
        # occupied-voxel差分だけを評価する厳密Chamfer高速経路を利用できる。
        if (
            str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
            in {"ana_den6_online", "network_only_codec_policy", "network_k_proposal_policy", "single_plan_student"}
            and torch.is_tensor(out_label)
            and not bool(torch.any(out_label >= 0.5).item())
        ):
            out_label = None
        if gen_pts.shape[-1] == 0 or gt_pts.shape[-1] == 0:
            self._set_geometry_debug(
                mode="empty",
                value=0.0,
                hard=0.0,
                surrogate=0.0,
                weighted=0.0,
                gen_points=int(gen_pts.shape[-1]),
                gt_points=int(gt_pts.shape[-1]),
            )
            return gt_pts.new_zeros(())

        L_geom = 0.0
        mode = self._discrete_loss_mode(args)
        if mode == "hard":
            final_w = None
        use_weighted_forward = mode in {"weighted_soft", "soft", "legacy"} and final_w is not None
        use_ste_hard = mode in {"ste_hard", "hard_ste"} and final_w is not None
        forward_w = final_w if use_weighted_forward else None
        if args.loss_type == "cd":
            if out_label is None:
                gt_inlinear = gt_pts
            else:
                gt_inlinear = gt_pts

                if out_label is not None:
                    if out_label.dim() == 3:
                        label_b = out_label.size(0)
                        label_n = out_label.size(2)
                    elif out_label.dim() == 2:
                        label_b = out_label.size(0)
                        label_n = out_label.size(1)
                    else:
                        label_b = None
                        label_n = None

                    if label_b == gt_pts.size(0) and label_n == gt_pts.size(2):
                        gt_inlinear = remove_outlier_points_by_label(gt_pts, out_label)
                    else:
                        # gt_pts が 20000点にサンプリング済みで、
                        # out_label が full-cloud 全体のラベルの場合は対応関係がない。
                        # そのため外れ点除去はスキップする。
                        pass
            if gt_inlinear.shape[-1] == 0:
                self._set_geometry_debug(
                    mode="empty_gt_after_filter",
                    value=0.0,
                    hard=0.0,
                    surrogate=0.0,
                    weighted=0.0,
                    gen_points=int(gen_pts.shape[-1]),
                    gt_points=0,
                )
                return gt_pts.new_zeros(())
            with self._geometry_autocast_ctx(gen_pts):
                gen_pts_f = gen_pts.to(torch.float32)
                gt_inlinear_f = gt_inlinear.to(torch.float32)
                final_w_f = None if final_w is None else final_w.to(torch.float32)
                sparse_edit = self._exact_sparse_edit_chamfer(
                    args, gen_pts_f, gt_inlinear_f, final_w_f
                ) if out_label is None else None
                if sparse_edit is not None:
                    L_cd_hard = sparse_edit["hard"]
                    L_cd_surrogate = sparse_edit["surrogate"]
                    L_cd = (
                        self._compose_discrete_loss(L_cd_hard, L_cd_surrogate, args)
                        if use_ste_hard
                        else sparse_edit["weighted"] if use_weighted_forward
                        else L_cd_hard
                    )
                    hard_cd = L_cd_hard.detach()
                    weighted_cd = sparse_edit["weighted"].detach()
                    surrogate_cd = L_cd_surrogate.detach()
                    L_fit = sparse_edit["fit"]
                    mode_name = "exact_sparse_edit"
                elif use_ste_hard:
                    L_cd_hard, L_cd_surrogate = chamfer_l2_loss_and_weight_surrogate(
                        gen_pts_f,
                        gt_inlinear_f,
                        final_w_f,
                    )
                    L_cd = self._compose_discrete_loss(L_cd_hard, L_cd_surrogate, args)
                    hard_cd = chamfer_l2_loss(gen_pts_f, gt_inlinear_f, None).detach() if audit_enabled else L_cd.detach()
                    weighted_cd = (
                        chamfer_l2_loss(gen_pts_f, gt_inlinear_f, final_w_f).detach()
                        if audit_enabled
                        else L_cd.detach()
                    )
                    surrogate_cd = L_cd_surrogate.detach()
                    mode_name = "ste_hard"
                else:
                    forward_w_f = None if forward_w is None else forward_w.to(torch.float32)
                    L_cd = chamfer_l2_loss(gen_pts_f, gt_inlinear_f, forward_w_f)
                    hard_cd = chamfer_l2_loss(gen_pts_f, gt_inlinear_f, None).detach() if audit_enabled else L_cd.detach()
                    weighted_cd = hard_cd if final_w_f is None else (
                        chamfer_l2_loss(gen_pts_f, gt_inlinear_f, final_w_f).detach()
                        if audit_enabled
                        else L_cd.detach()
                    )
                    surrogate_cd = weighted_cd
                    mode_name = "weighted_soft" if use_weighted_forward else "hard"
                    L_fit = self._fit_proxy_loss(gen_pts_f, gt_inlinear_f)
                if sparse_edit is None and use_ste_hard:
                    L_fit = self._fit_proxy_loss(gen_pts_f, gt_inlinear_f)
            L_geom = L_cd
            fit_weight = max(float(getattr(args, "geometry_fit_weight", 0.05)), 0.0)
            if fit_weight > 0.0:
                L_geom = L_geom + fit_weight * L_fit
            self._set_geometry_debug(
                mode=mode_name,
                value=float(L_geom.detach()),
                hard=float(hard_cd),
                surrogate=float(surrogate_cd),
                weighted=float(weighted_cd),
                fit=float(L_fit.detach()),
                fit_weight=float(fit_weight),
                gen_points=int(gen_pts.shape[-1]),
                gt_points=int(gt_inlinear.shape[-1]),
                sparse_added_points=(
                    int(sparse_edit["added_count"]) if sparse_edit is not None else None
                ),
                sparse_removed_points=(
                    int(sparse_edit["removed_count"]) if sparse_edit is not None else None
                ),
            )
            if self._should_verbose_step(args):
                self.writer.write(
                    f"L_geom  :{self._scalar(L_geom):.4f}"
                    f" (hard:{float(hard_cd):.4f}, weighted:{float(weighted_cd):.4f}, fit:{float(L_fit):.4f}, w_fit:{fit_weight:.4f})"
                )
        elif args.loss_type == "cd+d2":
            with self._geometry_autocast_ctx(gen_pts):
                gen_pts_f = gen_pts.to(torch.float32)
                gt_pts_f = gt_pts.to(torch.float32)
                final_w_f = None if final_w is None else final_w.to(torch.float32)
                if use_ste_hard:
                    L_cd_hard, L_cd_surrogate = chamfer_l2_loss_and_weight_surrogate(
                        gen_pts_f,
                        gt_pts_f,
                        final_w_f,
                    )
                    L_cd = self._compose_discrete_loss(L_cd_hard, L_cd_surrogate, args)
                elif use_weighted_forward:
                    L_cd_hard = chamfer_l2_loss(gen_pts_f, gt_pts_f)
                    L_cd_soft = chamfer_l2_loss(gen_pts_f, gt_pts_f, final_w_f)
                    L_cd = self.lambda_p * L_cd_hard + L_cd_soft
                else:
                    L_cd_hard = chamfer_l2_loss(gen_pts_f, gt_pts_f)
                    L_cd = L_cd_hard

                L_fit = self._fit_proxy_loss(gen_pts_f, gt_pts_f)

                if bool(getattr(args, "geometry_use_d2", False)):
                    L_d2_hard = compute_d2_psnr(gen_pts_f, gt_pts_f, use_torch_ops=use_torch_d2)
                    if use_weighted_forward:
                        L_d2_soft = compute_d2_psnr(gen_pts_f, gt_pts_f, final_w=final_w_f, use_torch_ops=use_torch_d2)
                        L_d2_psnr = self.lambda_p * L_d2_hard + L_d2_soft
                    else:
                        L_d2_psnr = L_d2_hard
                    L_d2_term = -float(getattr(args, "geom_d2_weight", 0.0)) * L_d2_psnr
                else:
                    L_d2_psnr = gen_pts_f.new_zeros(())
                    L_d2_term = gen_pts_f.new_zeros(())

            fit_weight = max(float(getattr(args, "geometry_fit_weight", 0.05)), 0.0)
            L_geom = L_cd + fit_weight * L_fit + L_d2_term
            self._set_geometry_debug(
                mode="cd+d2",
                value=float(L_geom.detach()),
                hard=float(L_cd_hard.detach() if 'L_cd_hard' in locals() else L_cd.detach()),
                surrogate=float(L_cd_surrogate.detach() if 'L_cd_surrogate' in locals() else L_cd.detach()),
                weighted=float(L_cd.detach()),
                fit=float(L_fit.detach()),
                fit_weight=float(fit_weight),
                d2_psnr=float(L_d2_psnr.detach()),
                d2_term=float(L_d2_term.detach()),
                gen_points=int(gen_pts.shape[-1]),
                gt_points=int(gt_pts.shape[-1]),
            )
            if self._should_verbose_step(args):
                self.writer.write(
                    f"L_geom  :{self._scalar(L_geom):.4f}->"
                    f"L_cd:{self._scalar(L_cd):.4f}, "
                    f"Fit:{self._scalar(L_fit):.4f}, "
                    f"D2PSNR:{self._scalar(L_d2_psnr):.4f}, "
                    f"L_d2_term:{self._scalar(L_d2_term):.4f}"
                )

        soft_proxy = self._soft_actuator_geometry_proxy(args, L_geom)
        if torch.is_tensor(soft_proxy) and soft_proxy.requires_grad:
            L_geom = L_geom + (soft_proxy - soft_proxy.detach())

        return L_geom
