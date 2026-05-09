import numpy as np
import torch

from .actual_encoder import build_actual_encoder
from models.utils.compression.octree_stats import hard_octree_occupancy_stats


class CompressionLossMixin:
    def _store_compression_terms(self, **terms):
        self.last_compression_terms = dict(terms)

    @staticmethod
    def _compression_rate_metric(args):
        return str(getattr(args, "compression_rate_metric", "bits_per_point")).strip().lower()

    @staticmethod
    def _compression_loss_backend(args):
        return str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()

    @staticmethod
    def _positive_count(count):
        return max(float(count), 1.0)

    def _metric_value(self, value, own_point_count, ref_point_count, args):
        mode = self._compression_rate_metric(args)
        if mode == "total_bits":
            return value
        if mode == "bits_per_input_point":
            return value / self._positive_count(ref_point_count)
        return value / self._positive_count(own_point_count)

    @staticmethod
    def _relative_ratio(value, ref, ref_min=1e-12):
        if torch.is_tensor(value):
            ref_t = value.new_tensor(float(ref))
            denom = ref_t.abs().clamp_min(float(ref_min))
            return (value - ref_t) / denom
        denom = max(abs(float(ref)), float(ref_min))
        return (float(value) - float(ref)) / denom

    @staticmethod
    def _relative_percent(value, ref, ref_min=1e-12):
        return 100.0 * CompressionLossMixin._relative_ratio(value, ref, ref_min)

    def _get_cached_actual_gt(self, cache_key):
        if not self.gt_cache_enabled or not cache_key:
            return None
        cache_entry = self.actual_gt_cache.get(cache_key)
        if cache_entry is None:
            return None
        self.actual_gt_cache.move_to_end(cache_key)
        return dict(cache_entry)

    def _store_cached_actual_gt(self, cache_key, cache_entry):
        if not self.gt_cache_enabled or not cache_key or self.gt_cache_max_entries <= 0:
            return
        self.actual_gt_cache[cache_key] = dict(cache_entry)
        self.actual_gt_cache.move_to_end(cache_key)
        while len(self.actual_gt_cache) > self.gt_cache_max_entries:
            self.actual_gt_cache.popitem(last=False)

    def _get_actual_encoder(self, args):
        backend = self._compression_loss_backend(args)
        compress_key = str(getattr(args, "compress", "OctAttention")).strip().lower().replace("_", "").replace("-", "")
        if backend.startswith("sparsepcgc") or compress_key == "sparsepcgc":
            codec_key = "sparsepcgc"
        elif backend.startswith("gpcc") or compress_key == "gpcc":
            codec_key = "gpcc"
        else:
            codec_key = "octattention"
        if self.actual_encoder is None or getattr(self, "actual_encoder_codec_key", None) != codec_key:
            old_encoder = self.actual_encoder
            if old_encoder is not None and hasattr(old_encoder, "close"):
                old_encoder.close()
            self.actual_encoder = build_actual_encoder(args, writer=self.writer)
            self.actual_encoder_codec_key = codec_key
            if hasattr(self, "actual_gt_cache"):
                self.actual_gt_cache.clear()
            if hasattr(self, "surrogate_target_cache"):
                self.surrogate_target_cache.clear()
            if hasattr(self, "last_surrogate_target_entry"):
                self.last_surrogate_target_entry = None
        return self.actual_encoder

    @staticmethod
    def _effective_keep_mask_from_weights(weights, args):
        weights = torch.nan_to_num(weights.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        point_count = int(weights.numel())
        if point_count <= 0:
            return weights.new_zeros((0,), dtype=torch.bool)
        threshold = float(
            getattr(args, "operation_count_drop_threshold", getattr(args, "test_drop_threshold", 0.5))
        )
        keep_mask = weights >= threshold
        keep_count = int(keep_mask.sum().item())
        if keep_count <= 0 or keep_count >= point_count:
            expected_keep = int(round(float(weights.sum().item())))
            expected_keep = min(max(expected_keep, 1), point_count)
            if 0 < expected_keep < point_count:
                topk_idx = torch.topk(weights, k=expected_keep, largest=True, sorted=False).indices
                keep_mask = torch.zeros_like(weights, dtype=torch.bool)
                keep_mask.scatter_(0, topk_idx, True)
        if not bool(keep_mask.any().item()):
            keep_mask[torch.argmax(weights)] = True
        return keep_mask

    def _select_actual_points(self, xyz_3n, final_w, args, batch_idx):
        if final_w is None:
            return xyz_3n
        if final_w.ndim == 3:
            weights = final_w[batch_idx].squeeze(0)
        elif final_w.ndim == 2:
            weights = final_w[batch_idx]
        else:
            weights = final_w.reshape(-1)
        point_count = int(xyz_3n.shape[-1])
        weights = weights.to(device=xyz_3n.device, dtype=torch.float32).reshape(-1)
        if weights.numel() > point_count:
            weights = weights[:point_count]
        elif weights.numel() < point_count:
            pad = weights.new_ones((point_count - int(weights.numel()),))
            weights = torch.cat([weights, pad], dim=0)
        keep_mask = self._effective_keep_mask_from_weights(weights, args)
        return xyz_3n[:, keep_mask]

    def _encode_actual_batch(self, args, xyz, final_w=None):
        encoder = self._get_actual_encoder(args)
        stats_list = []
        for b in range(xyz.shape[0]):
            pts_b = self._select_actual_points(xyz[b].to(torch.float32), final_w, args, b)
            stats = dict(encoder.encode_bits(pts_b))
            stats = self._attach_octree_aux_stats(args, pts_b, stats)
            stats_list.append(stats)
        total_bit = sum(s["bit"] for s in stats_list)
        total_single = sum(s["single"] for s in stats_list)
        total_node = sum(s["node"] for s in stats_list)
        total_points = sum(s["point_count"] for s in stats_list)
        total_octree_single = sum(float(s.get("octree_single", s.get("single", 0.0))) for s in stats_list)
        total_octree_node = sum(float(s.get("octree_node", s.get("node", 0.0))) for s in stats_list)
        max_octree_depth = max((int(s.get("octree_depth", 0)) for s in stats_list), default=0)
        total_leaf = sum(int(s.get("octree_leaf_count", s.get("point_count", 0))) for s in stats_list)
        return {
            "bit": float(total_bit),
            "bpp": float(total_bit) / max(float(total_points), 1.0),
            "bpn": float(total_bit) / max(float(total_node), 1.0),
            "single": float(total_single),
            "node": float(total_node),
            "octree_single": float(total_octree_single),
            "octree_node": float(total_octree_node),
            "octree_depth": int(max_octree_depth),
            "octree_leaf_count": int(total_leaf),
            "point_count": int(total_points),
            "codec": str(getattr(encoder, "codec_name", "octattention")),
            "per_batch": stats_list,
        }

    def _actual_octree_stat_qs(self, args, codec_name):
        codec_key = str(codec_name).strip().lower()
        if codec_key == "sparsepcgc":
            return max(
                float(getattr(args, "sparsepcgc_effective_qs", 0.0))
                or float(getattr(args, "sparsepcgc_voxel_size", 1.0)) * float(getattr(args, "sparsepcgc_pos_quantscale", 1)),
                1e-9,
            )
        if codec_key == "gpcc":
            return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 1.0))), 1e-9)
        return max(float(getattr(args, "qs", 1.0)), 1e-9)

    def _attach_octree_aux_stats(self, args, pts_3n, stats):
        codec_name = str(stats.get("codec", getattr(self, "actual_encoder_codec_key", "octattention"))).strip().lower()
        need_aux = (
            bool(getattr(args, "compression_octree_stat_force", True))
            or float(stats.get("node", 0.0)) <= 0.0
            or float(stats.get("single", 0.0)) <= 0.0
            or codec_name in {"sparsepcgc", "gpcc"}
        )
        if not need_aux:
            stats.setdefault("octree_node", float(stats.get("node", 0.0)))
            stats.setdefault("octree_single", float(stats.get("single", 0.0)))
            stats.setdefault("octree_depth", 0)
            stats.setdefault("octree_leaf_count", int(stats.get("point_count", 0)))
            return stats
        aux = hard_octree_occupancy_stats(
            pts_3n,
            qs=self._actual_octree_stat_qs(args, codec_name),
            max_depth=int(getattr(args, "compression_octree_stat_depth", 0)),
        )
        aux_node = float(aux["node_count"])
        aux_single = float(aux["single_child_count"])
        stats["octree_node"] = aux_node
        stats["octree_single"] = aux_single
        stats["octree_depth"] = int(aux["max_depth"])
        stats["octree_leaf_count"] = int(aux["leaf_count"])
        stats["octree_single_ratio"] = float(aux["single_ratio"])
        stats["octree_mean_children"] = float(aux["mean_children"])
        if float(stats.get("node", 0.0)) <= 0.0 or codec_name in {"sparsepcgc", "gpcc"}:
            stats["node"] = aux_node
        if float(stats.get("single", 0.0)) <= 0.0 or codec_name in {"sparsepcgc", "gpcc"}:
            stats["single"] = aux_single
        stats["bpn"] = float(stats.get("bit", 0.0)) / max(float(stats.get("node", 0.0)), 1.0)
        return stats

    def _log_compression_grad_probe(self, args, label, L_com, gen_xyz):
        if not bool(getattr(args, "compression_grad_probe", True)):
            return
        every = max(int(getattr(args, "compression_grad_probe_every", 1)), 1)
        self._compression_grad_probe_count += 1
        if self._compression_grad_probe_count % every != 0:
            return

        requires_grad = bool(torch.is_tensor(L_com) and L_com.requires_grad)
        grad_fn = type(L_com.grad_fn).__name__ if requires_grad and L_com.grad_fn is not None else "None"
        grad_norm = None
        grad_ok = False
        err = None

        if requires_grad:
            try:
                grad = torch.autograd.grad(
                    L_com,
                    gen_xyz,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    grad_norm = float(grad.detach().norm().cpu())
                    grad_ok = grad_norm > 0.0 and np.isfinite(grad_norm)
            except Exception as exc:
                err = str(exc)

        msg = (
            f"[GradCheck][L_com:{label}] "
            f"requires_grad={requires_grad}, grad_fn={grad_fn}, "
            f"grad_to_gen_xyz={'OK' if grad_ok else 'NG'}, "
            f"grad_norm={grad_norm if grad_norm is not None else 'None'}"
        )
        if err is not None:
            msg += f", err={err}"
        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(msg)

    def _get_compression_loss_actual_codec(self, args, gen_xyz, gt_xyz, final_w, cache_key=None, use_proxy_surrogate=False):
        cached_gt = self._get_cached_actual_gt(cache_key)
        if cached_gt is None:
            cached_gt = self._encode_actual_batch(args, gt_xyz)
            self._store_cached_actual_gt(cache_key, cached_gt)

        stats_gen = self._encode_actual_batch(args, gen_xyz, final_w=final_w)
        codec_name = str(stats_gen.get("codec", cached_gt.get("codec", "octattention"))).strip().lower()
        backend_label = f"{codec_name}_actual_ste" if use_proxy_surrogate else f"{codec_name}_actual"
        gt_bit = float(cached_gt["bit"])
        gen_bit = float(stats_gen["bit"])
        loss_bit_ratio = self._relative_ratio(gen_bit, gt_bit)
        loss_bit_percent = 100.0 * loss_bit_ratio

        L_com_hard = gen_xyz.new_tensor(loss_bit_percent)
        L_com = L_com_hard

        proxy_debug = None
        if use_proxy_surrogate:
            proxy_L_com, proxy_loss_bit, proxy_loss_single, proxy_loss_nodes, _, _ = self._get_compression_loss_proxy(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                run_grad_probe=False,
            )
            proxy_terms = dict(getattr(self, "last_compression_terms", {}) or {})
            proxy_bit = proxy_terms.get("bit")
            surrogate_loss = proxy_bit if torch.is_tensor(proxy_bit) else None
            L_com = self._compose_discrete_loss(L_com_hard, surrogate_loss, args)
            proxy_debug = {
                "L_com": self._scalar(proxy_L_com),
                "loss_bit": self._scalar(proxy_loss_bit),
                "loss_single": self._scalar(proxy_loss_single),
                "loss_nodes": self._scalar(proxy_loss_nodes),
            }

        loss_bit = gen_xyz.new_tensor(loss_bit_percent)
        loss_single = gen_xyz.new_tensor(
            self._relative_percent(float(stats_gen["single"]), float(cached_gt["single"]), ref_min=1.0)
        )
        loss_nodes = gen_xyz.new_tensor(
            self._relative_percent(float(stats_gen["node"]), float(cached_gt["node"]), ref_min=1.0)
        )
        self._store_compression_terms(
            bit=L_com_hard,
            single=gen_xyz.new_zeros(()),
            node=gen_xyz.new_zeros(()),
            bpn=gen_xyz.new_zeros(()),
            objective=L_com,
            backend=backend_label,
        )
        self.last_compression_debug = {
            "metric": "actual_total_bit_percent",
            "teacher_codec": codec_name,
            "total_bit": loss_bit_percent,
            "bpp": self._relative_percent(float(stats_gen["bpp"]), float(cached_gt["bpp"])),
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": int(stats_gen["point_count"]),
            "gt_actual_bit": gt_bit,
            "gen_actual_bit": gen_bit,
            "actual_total_bit_percent": loss_bit_percent,
            "proxy_surrogate": proxy_debug,
        }

        if self._should_verbose_step(args):
            surrogate_msg = ""
            if proxy_debug is not None:
                surrogate_msg = (
                    f", proxy_surrogate_L:{proxy_debug['L_com']:.4f}, "
                    f"proxy_surrogate_bit:{proxy_debug['loss_bit']:.4f}"
                )
            self.writer.write(
                f"L_com(actual {codec_name}):{self._scalar(L_com):.6f}->"
                f"bit:{gt_bit:.1f}->{gen_bit:.1f}, "
                f"rel:{loss_bit_percent:.4f}%"
                f"{surrogate_msg}"
            )

        self._log_compression_grad_probe(args, backend_label, L_com, gen_xyz)

        stats_gt = {
            "bit": gt_bit,
            "bpp": float(cached_gt["bpp"]),
            "bpn": float(cached_gt["bpn"]),
            "single": float(cached_gt["single"]),
            "node": float(cached_gt["node"]),
        }
        return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt

    def get_compression_loss(self, args, gen_xyz, gt_xyz, final_w, cache_key=None, refresh_actual_gen=True):
        self._store_compression_terms()
        backend = self._compression_loss_backend(args)
        if backend in {"octattention_surrogate", "sparsepcgc_surrogate", "gpcc_surrogate", "surrogate", "soft_surrogate"}:
            return self._get_compression_loss_surrogate(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                refresh_actual_gen=refresh_actual_gen,
            )
        if backend in {"octattention_actual", "actual_octattention", "real_octattention", "sparsepcgc_actual", "gpcc_actual"}:
            return self._get_compression_loss_actual_codec(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                use_proxy_surrogate=False,
            )
        if backend in {"octattention_actual_ste", "actual_octattention_ste", "real_octattention_ste", "sparsepcgc_actual_ste", "gpcc_actual_ste"}:
            return self._get_compression_loss_actual_codec(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                use_proxy_surrogate=True,
            )
        if backend != "proxy":
            raise ValueError(
                "--compression_loss_backend must be one of: proxy, "
                "octattention_actual, octattention_actual_ste, octattention_surrogate, "
                "sparsepcgc_actual, sparsepcgc_actual_ste, sparsepcgc_surrogate, "
                "gpcc_actual, gpcc_actual_ste, gpcc_surrogate "
                f"(got {backend})"
            )
        return self._get_compression_loss_proxy(
            args,
            gen_xyz=gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            cache_key=cache_key,
            run_grad_probe=True,
        )
