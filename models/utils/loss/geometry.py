from contextlib import nullcontext

import torch

from .utils_loss import (
    chamfer_l2_loss,
    chamfer_l2_loss_and_weight_surrogate,
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

    def get_geometry_loss(self, args, gen_pts, gt_pts, final_w=None, out_label=None):
        use_torch_d2 = args.trainORtest == "train"
        audit_enabled = self._should_verbose_step(args) or args.trainORtest != "train"
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
                gt_inlinear = remove_outlier_points_by_label(gt_pts, out_label)
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
            L_geom = L_cd
            self._set_geometry_debug(
                mode=mode_name,
                value=float(L_geom.detach()),
                hard=float(hard_cd),
                surrogate=float(surrogate_cd),
                weighted=float(weighted_cd),
                gen_points=int(gen_pts.shape[-1]),
                gt_points=int(gt_inlinear.shape[-1]),
            )
            if self._should_verbose_step(args):
                self.writer.write(
                    f"L_geom  :{self._scalar(L_geom):.4f}"
                    f" (hard:{float(hard_cd):.4f}, weighted:{float(weighted_cd):.4f})"
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

                L_d2_hard = compute_d2_psnr(gen_pts_f, gt_pts_f, use_torch_ops=use_torch_d2)
                if use_weighted_forward:
                    L_d2_soft = compute_d2_psnr(gen_pts_f, gt_pts_f, final_w=final_w_f, use_torch_ops=use_torch_d2)
                    L_d2_psnr = self.lambda_p * L_d2_hard + L_d2_soft
                else:
                    L_d2_psnr = L_d2_hard

            L_d2_term = -float(getattr(args, "geom_d2_weight", 0.2)) * L_d2_psnr
            L_geom += L_cd + L_d2_term
            self._set_geometry_debug(
                mode="cd+d2",
                value=float(L_geom.detach()),
                hard=float(L_cd_hard.detach() if 'L_cd_hard' in locals() else L_cd.detach()),
                surrogate=float(L_cd_surrogate.detach() if 'L_cd_surrogate' in locals() else L_cd.detach()),
                weighted=float(L_cd.detach()),
                d2_psnr=float(L_d2_psnr.detach()),
                d2_term=float(L_d2_term.detach()),
                gen_points=int(gen_pts.shape[-1]),
                gt_points=int(gt_pts.shape[-1]),
            )
            if self._should_verbose_step(args):
                self.writer.write(
                    f"L_geom  :{self._scalar(L_geom):.4f}->"
                    f"L_cd:{self._scalar(L_cd):.4f}, "
                    f"D2PSNR:{self._scalar(L_d2_psnr):.4f}, "
                    f"L_d2_term:{self._scalar(L_d2_term):.4f}"
                )

        return L_geom
