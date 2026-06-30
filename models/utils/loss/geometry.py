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
                if use_ste_hard:
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
