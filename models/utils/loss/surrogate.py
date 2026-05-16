import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CompressionSurrogateNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, pred_clip=2.0):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 16)
        self.pred_clip = float(pred_clip)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        raw = self.net(x)
        if self.pred_clip > 0:
            return self.pred_clip * torch.tanh(raw / self.pred_clip)
        return raw


class SurrogateCompressionLossMixin:
    @staticmethod
    def _parse_surrogate_levels(args):
        raw = getattr(args, "compression_surrogate_levels", "4,6,8")
        if isinstance(raw, (list, tuple)):
            vals = raw
        else:
            vals = str(raw).replace(" ", "").split(",")
        levels = []
        for val in vals:
            if val == "":
                continue
            level = int(val)
            if level > 0:
                levels.append(level)
        return levels or [4, 6, 8]

    def _ensure_surrogate_device(self, device):
        first_param = next(self.compression_surrogate.parameters())
        if first_param.device != device:
            self.compression_surrogate = self.compression_surrogate.to(device)
            state = self.surrogate_optimizer.state
            for value in state.values():
                for key, item in value.items():
                    if torch.is_tensor(item):
                        value[key] = item.to(device)

    @staticmethod
    def _all_finite(*values):
        for value in values:
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return False
        return True

    def _log_surrogate_event(self, message):
        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(f"[CompressionSurrogate] {message}")

    def _surrogate_state_is_finite(self):
        for param in self.compression_surrogate.parameters():
            if not torch.isfinite(param.detach()).all():
                return False
        for state in self.surrogate_optimizer.state.values():
            for item in state.values():
                if torch.is_tensor(item) and not torch.isfinite(item.detach()).all():
                    return False
        return True

    def _reset_compression_surrogate(self, reason):
        device = next(self.compression_surrogate.parameters()).device
        self.compression_surrogate = _CompressionSurrogateNet(
            in_dim=self.surrogate_feature_dim,
            hidden_dim=int(getattr(self.args, "compression_surrogate_hidden_dim", 128)),
            pred_clip=float(getattr(self.args, "compression_surrogate_pred_clip", 2.0)),
        ).to(device)
        self.surrogate_optimizer = torch.optim.Adam(
            self.compression_surrogate.parameters(),
            lr=float(getattr(self.args, "compression_surrogate_lr", 1e-3)),
            weight_decay=float(getattr(self.args, "compression_surrogate_weight_decay", 1e-5)),
        )
        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        self._log_surrogate_event(f"reset network ({reason}).")

    def _surrogate_target_from_actual(self, args, stats_gen, stats_ref, device):
        target_percent = self._relative_percent(float(stats_gen["bit"]), float(stats_ref["bit"]))
        clip = float(getattr(args, "compression_surrogate_pred_clip", 0.0))
        target = torch.tensor([[target_percent]], device=device, dtype=torch.float32)
        if clip > 0:
            target = target.clamp(min=-clip, max=clip)
        return target

    @staticmethod
    def _decode_keys(keys, grid):
        grid_t = torch.as_tensor(grid, device=keys.device, dtype=torch.long)
        xy = grid_t * grid_t
        z = torch.div(keys, xy, rounding_mode="floor")
        rem = keys - z * xy
        y = torch.div(rem, grid_t, rounding_mode="floor")
        x = rem - y * grid_t
        return torch.stack([x, y, z], dim=1)

    def _soft_level_stats(self, coords_norm, weights, level, args):
        grid = 2 ** int(level)
        scaled = (coords_norm * float(grid - 1)).clamp(0.0, float(grid - 1))
        base = torch.floor(scaled).to(torch.long)
        frac = scaled - base.to(scaled.dtype)

        masses = []
        keys = []
        for bx in (0, 1):
            wx = frac[0] if bx else (1.0 - frac[0])
            ix = (base[0] + bx).clamp(0, grid - 1)
            for by in (0, 1):
                wy = frac[1] if by else (1.0 - frac[1])
                iy = (base[1] + by).clamp(0, grid - 1)
                for bz in (0, 1):
                    wz = frac[2] if bz else (1.0 - frac[2])
                    iz = (base[2] + bz).clamp(0, grid - 1)
                    corner_weight = (wx * wy * wz).clamp_min(0.0)
                    masses.append(weights * corner_weight)
                    keys.append(ix + grid * (iy + grid * iz))

        mass = torch.cat(masses, dim=0)
        key = torch.cat(keys, dim=0)
        order = torch.argsort(key)
        key = key[order]
        mass = mass[order]
        unique_key, inverse = torch.unique_consecutive(key, return_inverse=True)
        voxel_mass = mass.new_zeros(unique_key.shape[0]).scatter_add(0, inverse, mass)

        gain = float(getattr(args, "compression_surrogate_occ_gain", 1.0))
        occ = 1.0 - torch.exp(-gain * voxel_mass.clamp_min(0.0))
        occ = occ.clamp(1e-6, 1.0 - 1e-6)

        node = occ.sum()
        entropy = -(occ * torch.log2(occ) + (1.0 - occ) * torch.log2(1.0 - occ)).sum()
        mass_total = voxel_mass.sum()

        if level <= 1 or unique_key.numel() == 0:
            single = node.new_zeros(())
        else:
            coords = self._decode_keys(unique_key, grid)
            parent_grid = grid // 2
            parent = torch.div(coords, 2, rounding_mode="floor")
            child_bits = coords - parent * 2
            child_id = child_bits[:, 0] * 4 + child_bits[:, 1] * 2 + child_bits[:, 2]
            parent_key = parent[:, 0] + parent_grid * (parent[:, 1] + parent_grid * parent[:, 2])
            parent_unique, parent_inv = torch.unique(parent_key, sorted=True, return_inverse=True)
            child_occ = occ.new_zeros(parent_unique.numel() * 8)
            flat_child_idx = parent_inv * 8 + child_id
            child_occ = child_occ.scatter_add(0, flat_child_idx, occ).view(-1, 8).clamp(1e-6, 1.0 - 1e-6)
            not_occ = (1.0 - child_occ).clamp(1e-6, 1.0)
            prod_not = not_occ.prod(dim=1, keepdim=True)
            single_prob = (child_occ * prod_not / not_occ).sum(dim=1)
            single = single_prob.sum()

        node_safe = node.clamp_min(1e-6)
        grid_total = float(grid ** 3)
        return torch.stack(
            [
                torch.log1p(node),
                torch.log1p(single),
                entropy / node_safe,
                torch.log1p(mass_total),
                node / max(grid_total, 1.0),
            ]
        )

    def _build_soft_compression_features(self, args, gen_xyz, gt_xyz, final_w):
        gen_xyz = torch.nan_to_num(gen_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        gt_xyz = torch.nan_to_num(gt_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        B, _, N = gen_xyz.shape
        if final_w is None:
            weights_all = gen_xyz.new_ones(B, N)
        else:
            weights_all = final_w.squeeze(1).to(device=gen_xyz.device, dtype=gen_xyz.dtype)
            if weights_all.shape[-1] > N:
                weights_all = weights_all[..., :N]
            elif weights_all.shape[-1] < N:
                pad = weights_all.new_ones(*weights_all.shape[:-1], N - weights_all.shape[-1])
                weights_all = torch.cat([weights_all, pad], dim=-1)
            weights_all = torch.nan_to_num(weights_all, nan=0.0, posinf=1.0, neginf=0.0)
            weights_all = weights_all.clamp(0.0, 1.0)

        features = []
        ref_min = gt_xyz.detach().amin(dim=2)
        ref_max = gt_xyz.detach().amax(dim=2)
        ref_span = (ref_max - ref_min).clamp_min(1e-6)
        codec_key = self._surrogate_backend_label(args).replace("_surrogate", "").replace("_actual", "")
        is_sparsepcgc = 1.0 if codec_key == "sparsepcgc" else 0.0
        is_gpcc = 1.0 if codec_key == "gpcc" else 0.0
        is_draco = 1.0 if codec_key == "draco" else 0.0
        effective_qs = self._surrogate_effective_qs(args, codec_key)

        for b in range(B):
            pts = gen_xyz[b].to(torch.float32)
            weights = weights_all[b].to(torch.float32)
            coords_norm = ((pts - ref_min[b].to(pts.dtype).unsqueeze(1)) / ref_span[b].to(pts.dtype).unsqueeze(1)).clamp(0.0, 1.0)
            w_sum = weights.sum().clamp_min(1e-6)
            w_mean = weights.mean()
            w_std = weights.std(unbiased=False)
            mean_xyz = (coords_norm * weights.unsqueeze(0)).sum(dim=1) / w_sum
            centered = coords_norm - mean_xyz.unsqueeze(1)
            std_xyz = torch.sqrt((centered.pow(2) * weights.unsqueeze(0)).sum(dim=1) / w_sum + 1e-8)
            bbox = (coords_norm.amax(dim=1) - coords_norm.amin(dim=1)).clamp_min(1e-6)
            global_feat = [
                torch.log1p(gen_xyz.new_tensor(float(N), dtype=torch.float32)),
                torch.log1p(w_sum),
                w_mean,
                w_std,
                mean_xyz[0], mean_xyz[1], mean_xyz[2],
                std_xyz[0], std_xyz[1], std_xyz[2],
                torch.log1p(bbox.prod()),
            ]
            level_feat = [
                self._soft_level_stats(coords_norm, weights, level, args)
                for level in self.surrogate_levels
            ]
            span_max = float(ref_span[b].detach().max().cpu())
            span_mean = ref_span[b].detach().mean().clamp_min(1e-6)
            q_level = max(int(math.ceil(math.log2(max(span_max / max(effective_qs, 1e-9) + 1.0, 2.0)))), 1)
            q_level = min(q_level, max(int(getattr(args, "proxy_max_depth", 12)), 1))
            qstep_feat = self._soft_level_stats(coords_norm, weights, q_level, args)
            q_norm = (gen_xyz.new_tensor(float(effective_qs), dtype=torch.float32) / span_mean.to(device=gen_xyz.device, dtype=torch.float32)).clamp_min(1e-9)
            codec_feat = torch.stack(
                [
                    torch.log1p(q_norm),
                    torch.log1p(1.0 / q_norm),
                    gen_xyz.new_tensor(float(q_level) / max(float(getattr(args, "proxy_max_depth", 12)), 1.0), dtype=torch.float32),
                    gen_xyz.new_tensor(is_sparsepcgc, dtype=torch.float32),
                    gen_xyz.new_tensor(is_gpcc, dtype=torch.float32),
                    gen_xyz.new_tensor(is_draco, dtype=torch.float32),
                ]
            ).to(device=gen_xyz.device, dtype=torch.float32)
            features.append(torch.cat([torch.stack(global_feat), *level_feat, qstep_feat, codec_feat], dim=0))

        return torch.stack(features, dim=0).to(device=gen_xyz.device, dtype=torch.float32)

    @staticmethod
    def _surrogate_effective_qs(args, codec_key):
        if str(codec_key).strip().lower() == "sparsepcgc":
            return max(
                float(getattr(args, "sparsepcgc_effective_qs", 0.0))
                or float(getattr(args, "sparsepcgc_voxel_size", 1.0)) * float(getattr(args, "sparsepcgc_pos_quantscale", 1)),
                1e-9,
            )
        if str(codec_key).strip().lower() == "gpcc":
            return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 1.0))), 1e-9)
        if str(codec_key).strip().lower() == "draco":
            return max(float(getattr(args, "draco_effective_qs", getattr(args, "qs", 1.0))), 1e-9)
        return max(float(getattr(args, "qs", 1.0)), 1e-9)

    def _soft_aux_percent_from_features(self, x_soft, x_ref):
        level_dim = 5 * len(self.surrogate_levels)
        if x_soft.shape[1] < 11 + level_dim or x_ref.shape[1] < 11 + level_dim:
            zero = x_soft.new_zeros(())
            return zero, zero
        gen_levels = x_soft[:, 11:11 + level_dim].reshape(x_soft.shape[0], len(self.surrogate_levels), 5)
        ref_levels = x_ref[:, 11:11 + level_dim].reshape(x_ref.shape[0], len(self.surrogate_levels), 5)
        gen_node = gen_levels[:, :, 0].mean()
        gen_single = gen_levels[:, :, 1].mean()
        ref_node = ref_levels[:, :, 0].detach().mean().clamp_min(1e-6)
        ref_single = ref_levels[:, :, 1].detach().mean().clamp_min(1e-6)
        node_percent = 100.0 * (gen_node - ref_node) / ref_node.abs().clamp_min(1e-6)
        single_percent = 100.0 * (gen_single - ref_single) / ref_single.abs().clamp_min(1e-6)
        return node_percent, single_percent

    def _set_surrogate_trainable(self, trainable):
        for param in self.compression_surrogate.parameters():
            param.requires_grad_(trainable)

    @staticmethod
    def _can_update_compression_surrogate(args):
        if str(getattr(args, "trainORtest", "train")).strip().lower() != "train":
            return False
        if not torch.is_grad_enabled():
            return False
        inference_enabled = getattr(torch, "is_inference_mode_enabled", None)
        if callable(inference_enabled) and inference_enabled():
            return False
        return True

    def _surrogate_target_cache_key(self, args, cache_key):
        backend = self._surrogate_backend_label(args)
        return f"{backend}|{cache_key or ''}"

    def _surrogate_target_is_current_step(self, args, entry):
        if entry is None:
            return False
        current_step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
        return int(entry.get("global_step", -1)) == current_step

    def _get_cached_surrogate_target(self, args, cache_key):
        entry = None
        key = self._surrogate_target_cache_key(args, cache_key)
        backend = self._surrogate_backend_label(args)
        cache = getattr(self, "surrogate_target_cache", None)
        if cache is not None and key in cache:
            cache.move_to_end(key)
            cached_entry = dict(cache[key])
            entry = cached_entry
            entry["cache_hit"] = "exact" if self._surrogate_target_is_current_step(args, cached_entry) else "stale"
        if entry is None and not cache_key and bool(getattr(args, "compression_surrogate_reuse_last_target", False)):
            last_entry = getattr(self, "last_surrogate_target_entry", None)
            if (
                last_entry is not None
                and str(last_entry.get("backend_label", backend)) == backend
            ):
                entry = dict(last_entry)
                entry["cache_hit"] = "last" if self._surrogate_target_is_current_step(args, last_entry) else "last_stale"
        return entry

    def _store_cached_surrogate_target(self, args, cache_key, entry):
        stored = dict(entry)
        stored.pop("cache_hit", None)
        self.last_surrogate_target_entry = dict(stored)
        key = self._surrogate_target_cache_key(args, cache_key)
        cache = getattr(self, "surrogate_target_cache", None)
        max_entries = max(int(getattr(self, "surrogate_target_cache_max_entries", 0)), 0)
        if cache is None or not cache_key or max_entries <= 0:
            return
        cache[key] = dict(stored)
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)

    def _store_surrogate_replay(self, args, x_soft, target):
        max_entries = max(int(getattr(self, "surrogate_replay_max_entries", 0)), 0)
        if max_entries <= 0 or not self._all_finite(x_soft, target):
            return
        x_cpu = x_soft.detach().to(device="cpu", dtype=torch.float32)
        y_cpu = target.detach().to(device="cpu", dtype=torch.float32)
        if y_cpu.shape[0] == 1 and x_cpu.shape[0] > 1:
            y_cpu = y_cpu.expand(x_cpu.shape[0], -1).contiguous()
        for idx in range(x_cpu.shape[0]):
            entry = (x_cpu[idx].clone(), y_cpu[min(idx, y_cpu.shape[0] - 1)].clone())
            if len(self.surrogate_replay) < max_entries:
                self.surrogate_replay.append(entry)
            else:
                self.surrogate_replay[self.surrogate_replay_next] = entry
                self.surrogate_replay_next = (self.surrogate_replay_next + 1) % max_entries

    def _sample_surrogate_replay(self, args, device):
        replay = getattr(self, "surrogate_replay", None)
        if not replay:
            return None, None
        batch = min(max(int(getattr(args, "compression_surrogate_replay_batch", 8)), 1), len(replay))
        start = int(getattr(self, "_surrogate_call_count", 0)) % len(replay)
        indices = [(start + offset) % len(replay) for offset in range(batch)]
        x = torch.stack([replay[idx][0] for idx in indices], dim=0).to(device=device, dtype=torch.float32)
        y = torch.stack([replay[idx][1] for idx in indices], dim=0).to(device=device, dtype=torch.float32)
        return x, y

    def _train_surrogate_replay(self, args, device):
        replay_steps = max(int(getattr(args, "compression_surrogate_replay_steps", 0)), 0)
        if replay_steps <= 0:
            return None
        x_replay, y_replay = self._sample_surrogate_replay(args, device)
        if x_replay is None:
            return None
        return self._train_compression_surrogate(args, x_replay, y_replay, train_steps=replay_steps)

    @staticmethod
    def _surrogate_cache_stats_from_entry(entry):
        return {
            "bit": float(entry.get("gt_actual_bit", 0.0)),
            "bpp": float(entry.get("gt_bpp", 0.0)),
            "bpn": float(entry.get("gt_bpn", 0.0)),
            "single": float(entry.get("gt_single", 0.0)),
            "node": float(entry.get("gt_node", 0.0)),
            "octree_single": float(entry.get("gt_octree_single", entry.get("gt_single", 0.0))),
            "octree_node": float(entry.get("gt_octree_node", entry.get("gt_node", 0.0))),
            "octree_depth": int(entry.get("gt_octree_depth", 0)),
            "point_count": int(entry.get("gt_points", 0)),
            "codec": str(entry.get("teacher_codec", "octattention")),
        }

    def _should_refresh_surrogate_teacher(self, args, entry, refresh_actual_gen=True):
        if isinstance(refresh_actual_gen, str) and refresh_actual_gen.strip().lower() == "always":
            return True
        if not bool(refresh_actual_gen):
            return False
        interval = max(int(getattr(args, "compression_surrogate_refresh_interval", 0)), 0)
        step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
        if entry is None:
            # 新しいsampleでも毎回actual codecを呼ばず、global step間隔で教師を更新する。
            # これにより外部codecの重複実行を避け、GPU/CPU時間がstepごとに膨らむのを防ぐ。
            return step == 0 or (interval > 0 and step % interval == 0)
        if interval <= 0:
            return False
        last_step = int(entry.get("global_step", -10**12))
        return (step - last_step) >= interval

    def _surrogate_refresh_policy_label(self, args):
        return (
            f"periodic_interval={int(getattr(args, 'compression_surrogate_refresh_interval', 0))},"
            f"warmup_steps={int(getattr(args, 'compression_surrogate_warmup_steps', 0))},"
            f"replay_steps={int(getattr(args, 'compression_surrogate_replay_steps', 0))},"
            f"forward={getattr(args, 'compression_surrogate_forward_mode', 'surrogate')},"
            f"reuse_last_target={bool(getattr(args, 'compression_surrogate_reuse_last_target', True))}"
        )

    def _train_compression_surrogate(self, args, x_soft, target, train_steps=None):
        if train_steps is None:
            train_steps = max(int(getattr(args, "compression_surrogate_train_steps", 2)), 0)
        else:
            train_steps = max(int(train_steps), 0)
        if train_steps <= 0:
            return x_soft.new_zeros(())
        if not self._can_update_compression_surrogate(args):
            return x_soft.new_zeros(())
        if not self._all_finite(x_soft, target):
            self._log_surrogate_event("skipped train step because x_soft/target was non-finite.")
            if not self._surrogate_state_is_finite():
                self._reset_compression_surrogate("non-finite state while skipping train")
            return x_soft.new_zeros(())

        self._ensure_surrogate_device(x_soft.device)
        if not self._surrogate_state_is_finite():
            self._reset_compression_surrogate("non-finite params before train")
        self.compression_surrogate.train()
        self._set_surrogate_trainable(True)
        x_det = x_soft.detach()
        y_det = target.detach().expand(x_det.shape[0], -1)
        last_loss = x_soft.new_zeros(())
        weight = float(getattr(args, "compression_surrogate_bit_weight", 1.0))

        with self._compression_autocast_ctx(x_soft.device):
            for _ in range(train_steps):
                self.surrogate_optimizer.zero_grad(set_to_none=True)
                pred = self.compression_surrogate(x_det)
                if not self._all_finite(pred):
                    self.surrogate_optimizer.zero_grad(set_to_none=True)
                    self._reset_compression_surrogate("non-finite prediction during train")
                    return x_soft.new_zeros(())
                loss = float(weight) * F.smooth_l1_loss(pred, y_det, reduction="mean")
                if not torch.isfinite(loss):
                    self.surrogate_optimizer.zero_grad(set_to_none=True)
                    self._reset_compression_surrogate("non-finite loss during train")
                    return x_soft.new_zeros(())
                loss.backward()
                grad_finite = True
                for param in self.compression_surrogate.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        grad_finite = False
                        break
                if not grad_finite:
                    self.surrogate_optimizer.zero_grad(set_to_none=True)
                    self._reset_compression_surrogate("non-finite gradient during train")
                    return x_soft.new_zeros(())
                torch.nn.utils.clip_grad_norm_(
                    self.compression_surrogate.parameters(),
                    float(getattr(args, "compression_surrogate_grad_clip", 10.0)),
                )
                self.surrogate_optimizer.step()
                if not self._surrogate_state_is_finite():
                    self._reset_compression_surrogate("non-finite params after optimizer step")
                    return x_soft.new_zeros(())
                last_loss = loss.detach()

        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        self._surrogate_step += train_steps
        return last_loss

    def _surrogate_backend_label(self, args, codec_name=None):
        backend = self._compression_loss_backend(args)
        if backend in {"surrogate", "soft_surrogate"}:
            codec = codec_name or str(getattr(args, "compress", "OctAttention")).strip().lower()
            codec = codec.replace("_", "").replace("-", "")
            if codec == "sparsepcgc":
                return "sparsepcgc_surrogate"
            if codec == "gpcc":
                return "gpcc_surrogate"
            if codec == "draco":
                return "draco_surrogate"
            return "octattention_surrogate"
        return backend

    def _get_compression_loss_surrogate(
        self,
        args,
        gen_xyz,
        gt_xyz,
        final_w,
        cache_key=None,
        refresh_actual_gen=True,
        actual_gen_xyz=None,
    ):
        timing_enabled = bool(getattr(args, "debug_timing", False))
        timing = {}

        def _sync_timing():
            if timing_enabled and torch.is_tensor(gen_xyz) and gen_xyz.is_cuda:
                torch.cuda.synchronize(gen_xyz.device)

        def _mark_timing(name, start_time):
            if not timing_enabled:
                return start_time
            _sync_timing()
            now = time.time()
            timing[name] = now - start_time
            return now

        if timing_enabled:
            _sync_timing()
            timing_cursor = time.time()
        else:
            timing_cursor = 0.0
        self._surrogate_call_count = int(getattr(self, "_surrogate_call_count", 0)) + 1
        x_soft = self._build_soft_compression_features(args, gen_xyz, gt_xyz, final_w)
        timing_cursor = _mark_timing("feature_gen", timing_cursor)
        aux_node_weight = float(getattr(args, "compression_surrogate_aux_node_weight", 0.0))
        aux_single_weight = float(getattr(args, "compression_surrogate_aux_single_weight", 0.0))
        if aux_node_weight > 0.0 or aux_single_weight > 0.0:
            x_ref = self._build_soft_compression_features(args, gt_xyz, gt_xyz, None)
            soft_node_percent, soft_single_percent = self._soft_aux_percent_from_features(x_soft, x_ref)
        else:
            soft_node_percent = x_soft.new_zeros(())
            soft_single_percent = x_soft.new_zeros(())
        timing_cursor = _mark_timing("feature_ref_aux", timing_cursor)
        inputs_finite = self._all_finite(gen_xyz, gt_xyz, x_soft)
        target = None
        stats_gen = None
        target_entry = self._get_cached_surrogate_target(args, cache_key)
        target_cache_hit = str(target_entry.get("cache_hit", "miss")) if target_entry is not None else "miss"
        actual_bit_percent = 0.0
        actual_bpp_percent = 0.0
        actual_single_percent = 0.0
        actual_node_percent = 0.0
        target_percent_value = 0.0
        gen_points = int(gen_xyz.shape[-1])
        gen_actual_bit = float("nan")
        cached_gt = None
        if not self._surrogate_state_is_finite():
            self._reset_compression_surrogate("non-finite params before inference")

        teacher_refreshed = bool(inputs_finite and self._should_refresh_surrogate_teacher(args, target_entry, refresh_actual_gen))
        if teacher_refreshed:
            actual_t0 = time.time() if timing_enabled else 0.0
            cached_gt = self._get_cached_actual_gt(cache_key)
            if cached_gt is None:
                cached_gt = self._encode_actual_batch(args, gt_xyz)
                self._store_cached_actual_gt(cache_key, cached_gt)
            # actual codec教師は評価指標なので、train用ノイズなしの編集点群で測る。
            actual_xyz = gen_xyz if actual_gen_xyz is None else actual_gen_xyz
            stats_gen = self._encode_actual_batch(args, actual_xyz, final_w=final_w)
            if timing_enabled:
                timing["actual_encode"] = time.time() - actual_t0
                timing_cursor = time.time()
            teacher_codec = str(stats_gen.get("codec", cached_gt.get("codec", "octattention"))).strip().lower()
            actual_bit_percent = self._relative_percent(float(stats_gen["bit"]), float(cached_gt["bit"]))
            actual_bpp_percent = self._relative_percent(float(stats_gen["bpp"]), float(cached_gt["bpp"]))
            actual_single_percent = self._relative_percent(
                float(stats_gen["single"]),
                float(cached_gt["single"]),
                ref_min=1.0,
            )
            actual_node_percent = self._relative_percent(
                float(stats_gen["node"]),
                float(cached_gt["node"]),
                ref_min=1.0,
            )
            target = self._surrogate_target_from_actual(args, stats_gen, cached_gt, gen_xyz.device)
            warmup_steps = max(
                int(getattr(args, "compression_surrogate_warmup_steps", getattr(args, "compression_surrogate_train_steps", 2))),
                0,
            )
            L_sur = self._train_compression_surrogate(args, x_soft, target, train_steps=warmup_steps)
            timing_cursor = _mark_timing("surrogate_fit", timing_cursor)
            self._store_surrogate_replay(args, x_soft, target)
            target_percent_value = float(actual_bit_percent)
            gen_points = int(stats_gen["point_count"])
            gen_actual_bit = float(stats_gen["bit"])
            current_step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
            self._store_cached_surrogate_target(
                args,
                cache_key,
                {
                    "backend_label": self._surrogate_backend_label(args, teacher_codec),
                    "teacher_codec": teacher_codec,
                    "actual_bit_percent": float(actual_bit_percent),
                    "actual_bpp_percent": float(actual_bpp_percent),
                    "actual_single_percent": float(actual_single_percent),
                    "actual_node_percent": float(actual_node_percent),
                    "target_bit_percent": float(target_percent_value),
                    "gt_points": int(cached_gt["point_count"]),
                    "gen_points": int(gen_points),
                    "gt_actual_bit": float(cached_gt["bit"]),
                    "gen_actual_bit": float(gen_actual_bit),
                    "gt_bpp": float(cached_gt["bpp"]),
                    "gen_bpp": float(stats_gen["bpp"]),
                    "gt_bpn": float(cached_gt["bpn"]),
                    "gen_bpn": float(stats_gen["bpn"]),
                    "gt_single": float(cached_gt["single"]),
                    "gen_single": float(stats_gen["single"]),
                    "gt_node": float(cached_gt["node"]),
                    "gen_node": float(stats_gen["node"]),
                    "gt_octree_single": float(cached_gt.get("octree_single", cached_gt["single"])),
                    "gen_octree_single": float(stats_gen.get("octree_single", stats_gen["single"])),
                    "gt_octree_node": float(cached_gt.get("octree_node", cached_gt["node"])),
                    "gen_octree_node": float(stats_gen.get("octree_node", stats_gen["node"])),
                    "gt_octree_depth": int(cached_gt.get("octree_depth", 0)),
                    "gen_octree_depth": int(stats_gen.get("octree_depth", 0)),
                    "global_step": int(current_step),
                    "surrogate_step": int(getattr(self, "_surrogate_step", 0)),
                },
            )
        elif target_entry is not None:
            teacher_codec = str(target_entry.get("teacher_codec", "octattention")).strip().lower()
            cached_gt = self._surrogate_cache_stats_from_entry(target_entry)
            actual_bit_percent = float(target_entry.get("actual_bit_percent", 0.0))
            actual_bpp_percent = float(target_entry.get("actual_bpp_percent", 0.0))
            actual_single_percent = float(target_entry.get("actual_single_percent", 0.0))
            actual_node_percent = float(target_entry.get("actual_node_percent", 0.0))
            target_percent_value = float(target_entry.get("target_bit_percent", actual_bit_percent))
            gen_points = int(target_entry.get("gen_points", gen_points))
            gen_actual_bit = float(target_entry.get("gen_actual_bit", float("nan")))
            L_sur = x_soft.new_zeros(())
            timing_cursor = _mark_timing("target_cache", timing_cursor)
        else:
            teacher_codec = self._surrogate_backend_label(args).replace("_surrogate", "")
            cached_gt = {
                "bit": 0.0,
                "bpp": 0.0,
                "bpn": 0.0,
                "single": 0.0,
                "node": 0.0,
                "point_count": int(gt_xyz.shape[-1]),
                "codec": teacher_codec,
            }
            if not inputs_finite:
                self._log_surrogate_event("skipped teacher refresh because generator features were non-finite.")
            L_sur = x_soft.new_zeros(())
            timing_cursor = _mark_timing("target_missing", timing_cursor)

        replay_loss = self._train_surrogate_replay(args, gen_xyz.device)
        if replay_loss is not None:
            L_sur = 0.5 * (L_sur + replay_loss) if teacher_refreshed else replay_loss
        timing_cursor = _mark_timing("surrogate_replay", timing_cursor)

        self._ensure_surrogate_device(gen_xyz.device)
        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        if inputs_finite:
            pred = self.compression_surrogate(x_soft)
            if not self._all_finite(pred):
                self._reset_compression_surrogate("non-finite prediction during inference")
                pred = self.compression_surrogate(x_soft)
            if not self._all_finite(pred):
                self._log_surrogate_event("using zero prediction because inference stayed non-finite after reset.")
                pred = x_soft.new_zeros((x_soft.shape[0], 1), dtype=torch.float32)
        else:
            pred = x_soft.new_zeros((x_soft.shape[0], 1), dtype=torch.float32)
        timing_cursor = _mark_timing("surrogate_predict", timing_cursor)

        surrogate_bit_percent = pred.reshape(-1).mean() if pred.numel() > 0 else x_soft.new_zeros(())
        actual_bit_percent_t = gen_xyz.new_tensor(float(actual_bit_percent), dtype=torch.float32)
        loss_single = soft_single_percent.to(device=gen_xyz.device, dtype=torch.float32)
        loss_nodes = soft_node_percent.to(device=gen_xyz.device, dtype=torch.float32)
        forward_mode = str(getattr(args, "compression_surrogate_forward_mode", "surrogate")).strip().lower()
        if forward_mode == "teacher_ste":
            surrogate_loss = surrogate_bit_percent if inputs_finite else None
            main_loss = self._compose_discrete_loss(actual_bit_percent_t, surrogate_loss, args)
        else:
            main_loss = surrogate_bit_percent if inputs_finite else actual_bit_percent_t
        aux_loss = aux_node_weight * loss_nodes + aux_single_weight * loss_single
        sparse_terms = self._sparsepcgc_aux_feature_terms(args, gen_xyz, gt_xyz, final_w)
        sparse_aux_loss = float(getattr(args, "com_sparsepcgc", 1.0)) * sparse_terms["loss"]
        L_com = main_loss + aux_loss + sparse_aux_loss
        backend_label = self._surrogate_backend_label(args, teacher_codec)
        self._store_compression_terms(
            bit=surrogate_bit_percent,
            node=loss_nodes,
            single=loss_single,
            bpn=gen_xyz.new_zeros(()),
            objective=L_com,
            forward=L_com,
            hard=actual_bit_percent_t,
            surrogate=surrogate_bit_percent,
            aux=aux_loss,
            sparsepcgc=sparse_terms["loss"],
            backend=backend_label,
        )

        pred_percent = surrogate_bit_percent.detach().reshape(())
        target_percent = pred_percent.new_tensor(float(target_percent_value)).detach().reshape(())
        surrogate_abs_bit_error = (pred_percent - target_percent).abs()
        rate_proxy_before_value = float(cached_gt["bit"])
        rate_proxy_after_value = rate_proxy_before_value * (1.0 + self._scalar(pred_percent) / 100.0)
        gt_octree_node_value = float(cached_gt.get("octree_node", cached_gt.get("node", 0.0)))
        gt_octree_single_value = float(cached_gt.get("octree_single", cached_gt.get("single", 0.0)))
        if target_entry is not None and not teacher_refreshed:
            gen_octree_node_value = float(target_entry.get("gen_octree_node", 0.0))
            gen_octree_single_value = float(target_entry.get("gen_octree_single", 0.0))
            gen_octree_depth_value = int(target_entry.get("gen_octree_depth", 0))
        elif stats_gen is not None:
            gen_octree_node_value = float(stats_gen.get("octree_node", 0.0))
            gen_octree_single_value = float(stats_gen.get("octree_single", 0.0))
            gen_octree_depth_value = int(stats_gen.get("octree_depth", 0))
        else:
            gen_octree_node_value = 0.0
            gen_octree_single_value = 0.0
            gen_octree_depth_value = 0
        self.last_compression_debug = {
            "metric": "actual_total_bit_percent",
            "teacher_codec": teacher_codec,
            "total_bit": self._scalar(surrogate_bit_percent),
            "compression_objective": self._scalar(L_com),
            "compression_main_loss": self._scalar(main_loss),
            "compression_aux_loss": self._scalar(aux_loss),
            "sparsepcgc_aux_loss": self._scalar(sparse_terms["loss"].detach()),
            "sparsepcgc_active_coord_loss": self._scalar(sparse_terms["active"].detach()),
            "sparsepcgc_isolated_proxy_loss": self._scalar(sparse_terms["single"].detach()),
            "sparsepcgc_entropy_proxy_loss": self._scalar(sparse_terms["entropy"].detach()),
            "sparsepcgc_density_proxy_loss": self._scalar(sparse_terms["density"].detach()),
            "bpp": float(actual_bpp_percent),
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": gen_points,
            "gt_actual_bit": float(cached_gt["bit"]),
            "gen_actual_bit": gen_actual_bit,
            "actual_total_bit_percent": self._scalar(actual_bit_percent_t),
            "rate_proxy_before": rate_proxy_before_value,
            "rate_proxy_after": rate_proxy_after_value,
            "rate_proxy_delta": self._scalar(surrogate_bit_percent.detach()),
            "actual_single_percent": float(actual_single_percent),
            "actual_node_percent": float(actual_node_percent),
            "soft_single_percent": self._scalar(loss_single.detach()),
            "soft_node_percent": self._scalar(loss_nodes.detach()),
            "node_delta": gen_octree_node_value - gt_octree_node_value,
            "single_delta": gen_octree_single_value - gt_octree_single_value,
            "surrogate_pred_bit": self._scalar(pred_percent),
            "surrogate_objective_bit": self._scalar(surrogate_bit_percent.detach()),
            "surrogate_target_bit": self._scalar(target_percent),
            "surrogate_abs_bit_error": self._scalar(surrogate_abs_bit_error),
            "surrogate_abs_mean_error": self._scalar(surrogate_abs_bit_error),
            "surrogate_train_loss": self._scalar(L_sur),
            "surrogate_replay_size": int(len(getattr(self, "surrogate_replay", []))),
            "surrogate_forward_mode": forward_mode,
            "teacher_refresh": bool(teacher_refreshed),
            "teacher_cache_hit": target_cache_hit,
            "teacher_refresh_policy": self._surrogate_refresh_policy_label(args),
            "surrogate_backward_enabled": bool(inputs_finite),
            "inputs_finite": bool(inputs_finite),
            "gt_octree_node": gt_octree_node_value,
            "gen_octree_node": gen_octree_node_value,
            "gt_octree_single": gt_octree_single_value,
            "gen_octree_single": gen_octree_single_value,
            "gt_octree_depth": int(cached_gt.get("octree_depth", target_entry.get("gt_octree_depth", 0) if target_entry else 0)),
            "gen_octree_depth": gen_octree_depth_value,
            "timing": {key: round(float(value), 6) for key, value in timing.items()},
        }
        debug_gen_xyz = gen_xyz if actual_gen_xyz is None else actual_gen_xyz
        self._maybe_update_sparsepcgc_debug(
            args,
            self.last_compression_debug,
            gen_xyz=debug_gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            codec_name=teacher_codec,
        )

        if self._should_verbose_step(args):
            msg = (
                f"L_com(actual_total_bit_percent):{self._scalar(actual_bit_percent_t):.6f}, "
                f"surrogate_bit_percent:{self._scalar(pred_percent):.6f}, "
                f"surrogate_abs_bit_error:{self._scalar(surrogate_abs_bit_error):.6f}, "
                f"L_sur:{self._scalar(L_sur):.6f}"
            )
            if teacher_refreshed and stats_gen is not None:
                msg += (
                    f", actual_bit:{float(cached_gt['bit']):.6f}->{float(stats_gen['bit']):.6f} "
                    f"({float(actual_bit_percent):.4f}%)"
                )
            elif target_entry is not None:
                msg += f", actual_bit:cached({target_cache_hit})"
            else:
                msg += ", actual_bit:skipped(non-finite)"
            self.writer.write(msg)

        self._log_compression_grad_probe(args, backend_label, L_com, gen_xyz)
        return (
            L_com,
            surrogate_bit_percent.detach(),
            loss_single.detach(),
            loss_nodes.detach(),
            cached_gt,
            {
                "bit": float(cached_gt["bit"]),
                "bpp": float(cached_gt["bpp"]),
                "bpn": float(cached_gt["bpn"]),
                "single": float(cached_gt["single"]),
                "node": float(cached_gt["node"]),
            },
        )
