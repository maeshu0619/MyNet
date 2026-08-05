import hashlib

import torch

from models.utils.training.utils import uses_actual_total_bit_objective


def resolve_compression_fixed_stage(args):
    # 圧縮損失を常に有効にするため、Episodeでdiagnosisへ切り替える二段階Stageを使わずjointに固定する。
    stage = str(getattr(args, "compression_fixed_stage", "joint")).strip().lower()
    # 空文字設定でもstage_loss_factorsがjoint扱いになるように既定値へ戻す。
    return stage or "joint"


def sparsepcgc_stage_guard_factors(args, stage, stage_factors):
    factors = dict(stage_factors or {})
    debug = {
        "stage_switch_guard_used": False,
        "stage_original": str(stage),
        "stage_effective": str(stage),
        "compression_loss_factor_original": float(factors.get("com", 1.0)),
        "compression_loss_factor_effective": float(factors.get("com", 1.0)),
        "policy_loss_factor_original": float(factors.get("policy", 1.0)),
        "policy_loss_factor_effective": float(factors.get("policy", 1.0)),
    }
    codec = str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "")
    if codec != "sparsepcgc" or not bool(getattr(args, "sparsepcgc_stage_switch_guard", True)):
        return factors, debug

    min_com = max(float(getattr(args, "sparsepcgc_min_compression_loss_factor", 1.0)), 0.0)
    max_policy = max(float(getattr(args, "sparsepcgc_max_policy_loss_factor_in_compression", 0.25)), 0.0)
    original_com = float(factors.get("com", 1.0))
    original_policy = float(factors.get("policy", 1.0))
    factors["com"] = max(original_com, min_com)
    factors["policy"] = min(original_policy, max_policy)
    debug.update(
        {
            "stage_switch_guard_used": bool(
                abs(factors["com"] - original_com) > 1e-12
                or abs(factors["policy"] - original_policy) > 1e-12
            ),
            "compression_loss_factor_effective": float(factors["com"]),
            "policy_loss_factor_effective": float(factors["policy"]),
            "stage_effective": str(stage),
        }
    )
    return factors, debug


def prepare_full_cloud_input_pcd(pts, use_cuda):
    # DataLoader出力を全点群の[B, N, C]形式にそろえる。
    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
    # 想定外の次元は点群として扱えないため、早い段階で明示的に止める。
    if input_pcd.dim() != 3:
        raise ValueError(f"Expected point cloud tensor [B, N, C], got {tuple(input_pcd.shape)}")
    # 点数を間引かず、必要な場合だけ全点をGPUへ転送する。
    if use_cuda:
        input_pcd = input_pcd.cuda(non_blocking=True)
    # モデル入力が期待する[B, C, N]形式へ全点を保ったまま変換する。
    return input_pcd.permute(0, 2, 1).contiguous()








def weighted_compression_terms(args, terms, L_com, La_fit, include_weight=True):
    # L_comと同じdevice/dtypeのゼロを作り、存在しない内訳を安全に0扱いする。
    zero = L_com.new_zeros(())
    # bit差分の圧縮内訳を取得する。
    bit_term = terms.get("bit", zero)
    # single-child差分の圧縮内訳を取得する。
    single_term = terms.get("single", zero)
    # node数差分の圧縮内訳を取得する。
    node_term = terms.get("node", zero)
    # bits-per-node差分の圧縮内訳を取得する。
    bpn_term = terms.get("bpn", zero)
    # SparsePCGC補助Proxyの圧縮内訳を取得する。
    sparsepcgc_term = terms.get("sparsepcgc", zero)
    # low-probability補助項はLa_fitがTensorのときだけ合成する。
    lowprob_term = La_fit if torch.is_tensor(La_fit) else zero
    # 既存のcom_*重みで内訳を1つの圧縮目的に合成する。
    raw_objective = (
        float(getattr(args, "com_bit", 0.0)) * bit_term
        + float(getattr(args, "com_sin", 0.0)) * single_term
        + float(getattr(args, "com_node", 0.0)) * node_term
        + float(getattr(args, "com_bpn", 0.0)) * bpn_term
        + float(getattr(args, "com_sparsepcgc", 0.0)) * sparsepcgc_term
        + float(getattr(args, "com_lowprob", 0.0)) * lowprob_term
    )
    # 呼び出し側が最終Lossを作る場合だけw_comを掛ける。
    if include_weight:
        return float(getattr(args, "w_com", 1.0)) * raw_objective
    # compression_primary側では後段でw_comを掛けるため、生の主目的を返す。
    return raw_objective


def actual_backend_compression_terms(args, terms, L_com, La_fit):
    # actual/surrogate backendではproxy用のcom_bit=1000を使わず、bit差分をpercent単位のまま主目的にする。
    zero = L_com.new_zeros(())
    # actual総bit差があればそれを優先し、なければSurrogateのbit予測percentを使う。
    actual_bit_term = terms.get("actual_total_bit_percent", None)
    bit_term = terms.get("bit", L_com)
    # actual bitがちょうど0のstepでは、proxy bitを主目的へ戻して無信号化を避ける。
    proxy_bit_term = terms.get("proxy_bit", None)
    if torch.is_tensor(actual_bit_term):
        try:
            if float(actual_bit_term.detach().abs().item()) <= 1e-9 and torch.is_tensor(proxy_bit_term):
                bit_term = proxy_bit_term
        except Exception:
            pass
    # SparsePCGCのsoft補助項を取得し、未計算なら0として扱う。
    sparsepcgc_term = terms.get("sparsepcgc", zero)
    # node補助は明示weightがある場合だけ足し、既定ではbit差分を膨らませない。
    node_term = terms.get("node", zero)
    # single補助も同じく明示weightがある場合だけ使う。
    single_term = terms.get("single", zero)
    # lowprob項はTensorとして存在する場合だけ補助目的に含める。
    lowprob_term = La_fit if torch.is_tensor(La_fit) else zero
    # actual系は各項をpercentスケールで合成し、proxy用の巨大係数を混ぜない。
    return (
        bit_term
        + float(getattr(args, "com_sparsepcgc", 0.0)) * sparsepcgc_term
        + float(getattr(args, "compression_surrogate_aux_node_weight", 0.0)) * node_term
        + float(getattr(args, "compression_surrogate_aux_single_weight", 0.0)) * single_term
        + float(getattr(args, "com_lowprob", 0.0)) * lowprob_term
    )


def _actual_total_bit_objective_mix_state(args, terms, L_com):
    # actual bit がちょうど0のstepでは、actual単独だとmainが無信号になる。
    # その場合だけproxy側を半分まで許して、mainが0のまま止まらないようにする。
    mix = min(max(float(getattr(args, "actual_total_bit_objective_mix", 0.5)), 0.0), 1.0)
    zero_fallback_used = False
    if uses_actual_total_bit_objective(args):
        actual_value = None
        if isinstance(terms, dict):
            actual_term = terms.get("actual_total_bit_percent", None)
            if torch.is_tensor(actual_term):
                try:
                    actual_value = float(actual_term.detach().abs().item())
                except Exception:
                    actual_value = None
            if actual_value is None:
                actual_term = terms.get("actual_total_bit_percent_fresh", None)
                if torch.is_tensor(actual_term):
                    try:
                        actual_value = float(actual_term.detach().abs().item())
                    except Exception:
                        actual_value = None
        if actual_value is None and isinstance(terms, dict):
            bit_term = terms.get("bit", None)
            if torch.is_tensor(bit_term):
                try:
                    actual_value = float(bit_term.detach().abs().item())
                except Exception:
                    actual_value = None
        if actual_value is None and torch.is_tensor(L_com):
            try:
                actual_value = float(L_com.detach().abs().item())
            except Exception:
                actual_value = None
        if actual_value is not None and actual_value <= 1e-9:
            mix = min(mix, 0.5)
            zero_fallback_used = True
    return mix, zero_fallback_used


def compose_train_compression_main(args, terms, L_com, La_fit):
    if bool(getattr(args, "minimal_loss_objective", True)):
        return L_com
    # compression_primary用に、w_comを掛ける前の圧縮主目的を作る。
    term_main = actual_backend_compression_terms(args, terms, L_com, La_fit) if uses_actual_total_bit_objective(args) else weighted_compression_terms(args, terms, L_com, La_fit, include_weight=False)
    # half/half比率を既定にしつつ、actual=0のときだけproxy側を少し戻す。
    mix, _ = _actual_total_bit_objective_mix_state(args, terms, L_com)
    # actual/surrogate系だけL_com直結と内訳目的を混ぜる。
    if uses_actual_total_bit_objective(args):
        return mix * L_com + (1.0 - mix) * term_main
    # Proxy系は従来通り内訳目的を使う。
    return term_main


def compose_train_compression_objective(args, terms, L_com, La_fit):
    if bool(getattr(args, "minimal_loss_objective", True)):
        return L_com
    # 実Codec/Surrogate系ではL_com直結と内訳合成を半々にし、片方だけに寄りすぎないようにする。
    direct_objective = float(getattr(args, "w_com", 1.0)) * L_com
    # actual/surrogateではproxy用の巨大係数を避け、proxyでは従来の重み付き内訳を使う。
    term_objective = (
        float(getattr(args, "w_com", 1.0)) * actual_backend_compression_terms(args, terms, L_com, La_fit)
        if uses_actual_total_bit_objective(args)
        else weighted_compression_terms(args, terms, L_com, La_fit, include_weight=True)
    )
    # half/half比率を既定にしつつ、actual=0のときだけproxy側を少し戻す。
    mix, _ = _actual_total_bit_objective_mix_state(args, terms, L_com)
    # actual/surrogate系だけL_com直結と内訳目的を混ぜる。
    if uses_actual_total_bit_objective(args):
        return mix * direct_objective + (1.0 - mix) * term_objective
    # Proxy系は従来通り内訳目的を使う。
    return term_objective


def should_collect_sparsepcgc_hard_debug(args, *, log_this_step, profile_this_step, global_step):
    # SparsePCGCのhard統計はactive座標集合を実生成するため、毎Stepではなく診断間隔だけ収集する。
    interval = max(int(getattr(args, "sparsepcgc_hard_debug_interval", getattr(args, "profile_interval", 100))), 0)
    # global stepは0始まりなので、ログ表示に合わせた1始まりのStep番号へ直す。
    step_idx = int(global_step) + 1
    # 0以下ならhard統計収集を完全に止め、soft補助lossと実Codec教師だけで学習する。
    if interval <= 0:
        return False
    # 明示設定時だけ通常ログStepにもhard統計を出し、既定ではprint_rate=1の巻き添えを避ける。
    if bool(getattr(args, "sparsepcgc_hard_debug_on_log", False)) and bool(log_this_step):
        return True
    # profile対象Stepでも先頭/末尾ログの巻き添えは避け、intervalに一致するStepだけ収集する。
    if bool(profile_this_step) and step_idx % interval == 0:
        return True
    # intervalに一致するStepだけhard統計を収集する。
    return step_idx % interval == 0


def apply_epoch_file_window(dataset, args, window_index):
    # Dataset生成時に保持した全ファイル一覧を取り出し、なければ現在のfilesを全体として扱う。
    all_files = list(getattr(dataset, "all_files", getattr(dataset, "files", [])))
    # 推論・評価用フレームを訓練へ混入させず、各系列の先頭150件（既定値）だけを訓練窓の母集団にする。
    train_limit = int(getattr(args, "train_frames_per_sequence", 150))
    if train_limit > 0:
        all_files = all_files[:train_limit]
    # max_filesを「1 epochで読むファイル数」として使う。
    window_size = int(getattr(args, "max_files", 0))
    # window指定が無効、または全件がwindow内に収まるなら全ファイルを使う。
    if window_size <= 0 or len(all_files) <= window_size:
        dataset.files = list(all_files)
        return dataset
    # 同じ系列を次に使うEpisodeでは、訓練領域内を次のwindow_size件へ進める。
    start = (int(window_index) * window_size) % len(all_files)
    # 訓練領域の末尾を超えた分だけ、その訓練領域の先頭へ巻き戻す。
    dataset.files = [all_files[(start + offset) % len(all_files)] for offset in range(window_size)]
    # DataLoaderが更新後のfilesだけを読む同じDatasetを返す。
    return dataset
