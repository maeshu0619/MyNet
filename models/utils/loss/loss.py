import os
import sys
from collections import OrderedDict
from contextlib import nullcontext

import torch

from .compression import CompressionLossMixin
from .geometry import GeometryLossMixin
from .proxy import ProxyCompressionLossMixin
from .surrogate import _CompressionSurrogateNet, SurrogateCompressionLossMixin, resolve_surrogate_pred_clip


device = torch.device("cpu")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.utils.compression.proxy_octree import ProxyOctreeConfig, SoftOctreeRateProxy


class Loss(
    CompressionLossMixin,
    SurrogateCompressionLossMixin,
    ProxyCompressionLossMixin,
    GeometryLossMixin,
):
    def __init__(self, args, file_date, writer):
        """セットアップ"""
        self.args = args
        self.com_bit = args.com_bit # 圧縮後ビットの重み係数
        self.com_sin = args.com_sin # 単一子ノードの重み係数
        self.com_node = args.com_node # ノード数の重み係数
        self.lambda_p = args.lambda_p # 幾何損失などで使う点群品質側の重み係数
        self.compress = args.compress # 圧縮損失の手法
        self.file_date = file_date # ログ用情報
        self.writer = writer
        self.bptt = args.bptt # bpp計算などに使う点数・ビット正規化用の設定
        self.ncl = None # 値が未定の内部変数の初期化
        self.actual_encoder = None # 実圧縮用Encoderの初期化
        self.actual_encoder_codec_key = None # 現在使っている圧縮の識別子を初期化

        """Proxy圧縮設定"""
        self.octree_cfgs = ProxyOctreeConfig( # Soft Octree Rate Proxy用のオブジェクト作成
            max_depth=args.proxy_max_depth,
            qs=args.qs,
            bptt=int(args.bptt),
            lambda_entropy=args.proxy_lambda_entropy,
            lambda_node_count=args.proxy_lambda_node_count,
            lambda_single_child=args.proxy_lambda_single_child,
            round_tau=float(getattr(args, "proxy_round_tau", 0.12)),
            mass_to_occ_gain=float(getattr(args, "proxy_mass_to_occ_gain", 1.0)),
            teacher_device=str(getattr(args, "octattention_teacher_device", "auto")),
        )
        self.rate_proxy = SoftOctreeRateProxy(self.octree_cfgs).to(device) # Soft Octreeによる微分可能な圧縮率推定器

        """Surrogate圧縮設定"""
        self.surrogate_levels = self._parse_surrogate_levels(args) # Surrogateが参照するOctree階層レベルを設定から解析
        self.surrogate_feature_dim = 22 + 5 * len(self.surrogate_levels) # Surrogateへ入力する特徴量次元の設定
        self.compression_surrogate = _CompressionSurrogateNet( # 実圧縮結果を近似するSurrogateNetworkの作成
            in_dim=self.surrogate_feature_dim,
            hidden_dim=int(getattr(args, "compression_surrogate_hidden_dim", 128)),
            pred_clip=resolve_surrogate_pred_clip(args),
        ).to(device)
        self.surrogate_optimizer = torch.optim.Adam( # Surrogat Network専用のAdam Optimizerを作成
            self.compression_surrogate.parameters(),
            lr=max(float(getattr(args, "compression_surrogate_lr", 1e-3)), float(getattr(args, "min_surrogate_lr", 1e-6))),
            weight_decay=float(getattr(args, "compression_surrogate_weight_decay", 1e-5)),
        )
        for param in self.compression_surrogate.parameters(): # Surrogat Networkの各Parameterを順番に取り出す
            param.requires_grad_(False)

        """キャッシュ設定"""
        self.gt_cache_enabled = bool(getattr(args, "cache_gt_loss", True)) # GT点群側の圧縮損失キャッシュを使うか否かを設定
        self.gt_cache_max_entries = max(int(getattr(args, "cache_max_entries", 64)), 0) # GTキャッシュの最大保存数
        self.gt_cache = OrderedDict() # GT点群のProxy圧縮結果を保存するLRU的なキャッシュ
        self.actual_gt_cache = OrderedDict() # 実CodecによるGT圧縮結果を保存
        self.surrogate_target_cache = OrderedDict() # Surrogateの教師値を保存するキャッシュ
        self.surrogate_target_cache_max_entries = max(int(getattr(args, "compression_surrogate_target_cache_entries", getattr(args, "cache_max_entries", 64))), 0)# Surrogate教師値キャッシュの最大保存数
        self.last_surrogate_target_entry = None # 直近のSurrogate教師データを初期化
        self.surrogate_replay = [] # Surrgateの再学習用Replay
        self.surrogate_replay_max_entries = max(int(getattr(args, "compression_surrogate_replay_entries", 512)), 0) # Replay Bufferの最大保存数を設定
        self.surrogate_replay_next = 0 # Replay Bufferの次に書き込む位置を初期化
        self._compression_grad_probe_count = 0 # 圧縮損失の勾配確認回数を0に初期化
        self._surrogate_step = 0 # Surrogateの更新Step数を0に初期化
        self._surrogate_call_count = 0 # Surrogateが呼ばれた回数を0に初期化
        self.last_geometry_debug = {} # 直近の幾何損失デバッグ情報を初期化
        self.last_compression_debug = {} # 直近の損失損失デバッグ情報を初期化
        self.last_compression_terms = {} # 直近の圧縮損失の内訳情報を初期化

    """基本判定"""
    @staticmethod
    def _scalar(x): # TensorをFloatに変換
        if torch.is_tensor(x):
            return float(x.detach())
        return float(x)

    @staticmethod
    def _discrete_loss_mode(args): # 離散操作に対する損失モードを取得
        return str(getattr(args, "discrete_loss_mode", "hard")).strip().lower()

    @staticmethod
    def _should_verbose_step(args): # 現在Stepで詳細ログを各べきか否かの判定
        return bool(
            getattr(args, "verbose_step_logs", False)
            and getattr(args, "_log_this_step", True)
            and not getattr(args, "compact_step_text_log", False)
        )

    def _surrogate_weight(self, args): # Hard LossにSurrogate勾配を混ぜる重みを取得
        return float(getattr(args, "discrete_surrogate_weight", 1.0))

    """損失関連の関数"""
    def _compose_discrete_loss(self, hard_loss, surrogate_loss, args): # Hardな損失値を使いつつ、BackWardではSurrogateの勾配を借りるための関数
        """Use the hard loss value while borrowing a surrogate backward pass."""
        weight = self._surrogate_weight(args)
        if surrogate_loss is None or weight == 0.0:
            return hard_loss
        return hard_loss + weight * (surrogate_loss - surrogate_loss.detach())

    def _get_cached_gt(self, cache_key, device): # GT点群に対する圧縮Proxy結果をキャッシュから取得する
        if not self.gt_cache_enabled or not cache_key:
            return None
        cache_entry = self.gt_cache.get(cache_key)
        if cache_entry is None:
            return None
        self.gt_cache.move_to_end(cache_key)
        out = dict(cache_entry)
        if out.get("gt_inlier") is not None:
            out["gt_inlier"] = out["gt_inlier"].to(device=device, non_blocking=True)
        return out

    def _store_cached_gt(self, cache_key, cache_entry): # GT点群に対する圧縮Proxy結果をキャッシュへ保存
        if not self.gt_cache_enabled or not cache_key or self.gt_cache_max_entries <= 0:
            return
        stored = dict(cache_entry)
        if stored.get("gt_inlier") is not None:
            stored["gt_inlier"] = stored["gt_inlier"].detach().to(device="cpu")
        self.gt_cache[cache_key] = stored
        self.gt_cache.move_to_end(cache_key)
        while len(self.gt_cache) > self.gt_cache_max_entries:
            self.gt_cache.popitem(last=False)

    def _ensure_rate_proxy_device(self, device): # Rate Prixyが入力点群と同じデバイスにあるか否かの判定
        if next(self.rate_proxy.buffers()).device != device:
            self.rate_proxy = self.rate_proxy.to(device)

    def _compression_autocast_ctx(self, device): # 圧縮Proxy計算時のAMP/AutoCastを制御する文脈を返す
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def warmup_gt_cache(self, gt_xyz, cache_key=None, subtree_tree=None, full_octree_context=None, octree_input_mode="auto"): # GT点群の圧縮Proxy結果を事前に計算してキャッシュする
        if not self.gt_cache_enabled or not cache_key:
            return
        device = gt_xyz.device
        self._ensure_rate_proxy_device(device)
        cached_gt = self._get_cached_gt(cache_key, device)
        if cached_gt is not None:
            return
        with self._compression_autocast_ctx(device):
            out_gt, bit_gt, stats_gt = self.rate_proxy.forward_hard_only(
                gen_xyz=gt_xyz.to(torch.float32),
                subtree_tree=subtree_tree,
                full_octree_context=full_octree_context,
                octree_input_mode=octree_input_mode,
            )
        cache_entry = {
            "rate_gt": self._scalar(out_gt["rate_total"]),
            "single_gt": self._scalar(out_gt["soft_single_child_count"]),
            "nodes_gt": self._scalar(out_gt["soft_node_count"]),
            "bit_gt": self._scalar(bit_gt),
            "point_count_gt": int(gt_xyz.shape[-1]),
            "stats_gt": {k: self._scalar(v) for k, v in stats_gt.items()},
        }
        self._store_cached_gt(cache_key, cache_entry)

    """全体損失の計算"""
    def get_loss(self, args, gen_pts, gt_pts, final_w, out_label, cache_key=None, subtree_tree=None, full_octree_context=None, octree_input_mode="auto"): # GT/Mine点群から、幾何/圧縮損失を計算する
        gt_xyz = gt_pts[:, :3, :]
        gen_xyz = gen_pts[:, :3, :]
        L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt = self.get_compression_loss( # 圧縮損失計算
            args,
            gen_xyz=gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            cache_key=cache_key,
            subtree_tree=subtree_tree,
            full_octree_context=full_octree_context,
            octree_input_mode=octree_input_mode,
        )
        L_geom = self.get_geometry_loss( # 幾何圧縮計算
            args,
            gen_pts=gen_pts,
            gt_pts=gt_pts,
            final_w=final_w,
            out_label=out_label,
        )

        if self._should_verbose_step(args): # ログを書くか否かの判定
            comp_debug = getattr(self, "last_compression_debug", {})
            metric = comp_debug.get("metric", self._compression_rate_metric(args))
            self.writer.write(
                f"L_com   :{self._scalar(L_com):.4f}->"
                f"L_rate({metric}):{self._scalar(loss_bit):.4f}, "
                f"L_total_bits:{float(comp_debug.get('total_bit', self._scalar(loss_bit))):.4f}, "
                f"L_bpp:{float(comp_debug.get('bpp', self._scalar(loss_bit))):.4f}, "
                f"L_single:{self._scalar(loss_single):.4f}, "
                f"L_nodes:{self._scalar(loss_nodes):.4f}, "
                f"points:{comp_debug.get('gt_points', gt_xyz.shape[-1])}->{comp_debug.get('gen_points', gen_xyz.shape[-1])}"
            )

        return L_geom, L_com, loss_bit, loss_single, loss_nodes
