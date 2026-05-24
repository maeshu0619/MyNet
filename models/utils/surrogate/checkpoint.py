import math
import os
import re
import glob
import torch


def _safe_component(value, fallback):
    text = str(value if value not in (None, "") else fallback).strip() # ファイル/ディレクトリ名に使う文字列を取り出す
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text) # パス区切りや空白を安全な文字へ置き換える
    return text or str(fallback) # 空文字になった場合はfallback名へ戻す


def _tensor_tree_to_cpu(value):
    if torch.is_tensor(value): # 保存時にGPUテンソルを保持しないようTensorをCPUへ移す
        return value.detach().to(device="cpu") # 勾配グラフを切り離してCPU保存用Tensorへ変換する
    if isinstance(value, dict): # Optimizer stateなどの辞書を再帰的に処理する
        return {key: _tensor_tree_to_cpu(item) for key, item in value.items()} # 辞書内TensorもCPU化する
    if isinstance(value, list): # state内のlist要素を再帰的に処理する
        return [_tensor_tree_to_cpu(item) for item in value] # list内TensorもCPU化する
    if isinstance(value, tuple): # state内のtuple要素を再帰的に処理する
        return tuple(_tensor_tree_to_cpu(item) for item in value) # tuple内TensorもCPU化する
    return value # 数値や文字列などTensor以外はそのまま保存する


def _surrogate_method_dir(args):
    if not bool(getattr(args, "surrogate_registry_enabled", True)): # 共有Surrogate保存が無効なら保存先を作らない
        return None # 呼び出し側で保存/読込をskipできるようNoneを返す
    root = os.path.abspath(os.path.expanduser(str(getattr(args, "surrogate_pretrained_root", "")))) # 共有Surrogate rootを絶対パスへ変換する
    method_name = _safe_component(getattr(args, "compress", "codec"), "codec") # method_com相当の圧縮器名を保存ディレクトリ名にする
    return os.path.join(root, method_name) # 圧縮手法ごとのSurrogate保存ディレクトリを返す


def _surrogate_save_filename(args):
    date_name = _safe_component(getattr(args, "date", ""), "date") # 今回runの日付を保存ファイル名に使う
    time_name = _safe_component(getattr(args, "time", ""), "time") # 今回runの時刻を保存ファイル名に使う
    return f"{date_name}_{time_name}.pth" # 提案手法モデル保存判定時のSurrogate保存名を返す


def _surrogate_load_filename(args):
    date_name = _safe_component(getattr(args, "surrogate_date", ""), getattr(args, "date", "date")) # 読込対象Surrogateの日付を決める
    time_name = _safe_component(getattr(args, "surrogate_time", ""), getattr(args, "time", "time")) # 読込対象Surrogateの時刻を決める
    return f"{date_name}_{time_name}.pth" # 事前学習開始前に探すSurrogateファイル名を返す


def surrogate_registry_path(args):
    method_dir = _surrogate_method_dir(args) # 今回runのSurrogate保存ディレクトリを取得する
    if method_dir is None:
        return None # 保存機能が無効なら保存先なしにする
    return os.path.join(method_dir, _surrogate_save_filename(args)) # 今回runの日付_時刻形式の保存先パスを返す


def surrogate_pretrain_load_path(args):
    method_dir = _surrogate_method_dir(args) # 読込対象Surrogateのディレクトリを取得する
    if method_dir is None:
        return None # 読込機能が無効なら候補なしにする
    return os.path.join(method_dir, _surrogate_load_filename(args)) # argsのsurrogate_date_surrogate_time形式の読込パスを返す


def latest_surrogate_registry_path(args):
    exact_path = surrogate_pretrain_load_path(args) # 指定された読込対象Surrogateパスを取得する
    if exact_path is None:
        return None # 保存機能が無効なら候補なしにする
    candidates = glob.glob(os.path.join(os.path.dirname(exact_path), "*.pth")) # 同じmethod内の保存済みSurrogateを列挙する
    candidates = [path for path in candidates if os.path.isfile(path)] # 実ファイルだけを候補に残す
    if not candidates:
        return exact_path # 候補がない場合は完全一致pathを返す
    return max(candidates, key=lambda path: os.path.getmtime(path)) # 最終更新時刻が最も新しいSurrogateを返す


def surrogate_state_payload(args, loss, *, metric_name=None, metric_value=None, source="unknown", extra=None):
    surrogate = getattr(loss, "compression_surrogate", None) # Loss内のSurrogateモデルを取得する
    if surrogate is None:
        return None # Surrogateが無いbackendでは保存しない
    optimizer = getattr(loss, "surrogate_optimizer", None) # Surrogate用Optimizer状態も一緒に保存する
    payload = {
        "compression_surrogate_state_dict": _tensor_tree_to_cpu(surrogate.state_dict()), # Surrogate本体の重みをCPU化して保存する
        "surrogate_optimizer_state_dict": None if optimizer is None else _tensor_tree_to_cpu(optimizer.state_dict()), # 継続学習用Optimizer状態をCPU化して保存する
        "surrogate_step": int(getattr(loss, "_surrogate_step", 0)), # Surrogate更新回数を保存する
        "surrogate_feature_dim": int(getattr(loss, "surrogate_feature_dim", 0) or 0), # 特徴次元不一致読込を防ぐため保存する
        "surrogate_levels": list(getattr(loss, "surrogate_levels", []) or []), # Octree特徴level不一致読込を防ぐため保存する
        "compression_loss_backend": getattr(args, "compression_loss_backend", None), # backend条件を後で確認できるよう保存する
        "compress": getattr(args, "compress", None), # 圧縮器名を保存する
        "dataname": getattr(args, "dataname", None), # 学習データ種別を保存する
        "dataset_name": getattr(args, "dataset_name", None), # 学習シーケンス名を保存する
        "surrogate_data": getattr(args, "surrogate_data", None), # 指定されたSurrogateデータ名を保存する
        "surrogate_date": getattr(args, "surrogate_date", None), # 指定されたSurrogate読込日付を保存する
        "surrogate_time": getattr(args, "surrogate_time", None), # 指定されたSurrogate時刻名を保存する
        "metric_name": metric_name, # ベスト判定に使ったmetric名を保存する
        "metric_value": None if metric_value is None else float(metric_value), # ベスト判定に使ったmetric値を保存する
        "source": str(source), # pretrain完了/episode checkpointなど保存元を保存する
    }
    if isinstance(extra, dict):
        payload.update(extra) # 呼び出し側の追加メタデータを保存する
    return payload # torch.saveへ渡すpayloadを返す


def save_surrogate_registry_state(args, loss, writer=None, *, metric_name=None, metric_value=None, source="unknown", extra=None):
    path = surrogate_registry_path(args) # 指定形式の共有Surrogate保存先を取得する
    if path is None:
        return None # 保存機能が無効なら何もしない
    payload = surrogate_state_payload(args, loss, metric_name=metric_name, metric_value=metric_value, source=source, extra=extra) # 保存payloadを組み立てる
    if payload is None:
        return None # Surrogateが存在しない場合は保存しない
    os.makedirs(os.path.dirname(path), exist_ok=True) # 保存先ディレクトリを作成する
    torch.save(payload, path) # Surrogate重みとOptimizer状態を指定パスへ保存する
    if writer is not None and hasattr(writer, "write"):
        writer.write(f"SurrogateRegistry saved: path={path}, source={source}, metric={metric_name}, value={metric_value}") # 保存結果をログに残す
    return path # 保存したパスを返す


def load_surrogate_registry_state(args, loss, writer=None):
    if not bool(getattr(args, "surrogate_pretrain_resume", True)): # resume無効時は共有Surrogateを読まない
        path = surrogate_pretrain_load_path(args) # resume無効時でもログ用に候補パスを作る
        if writer is not None and hasattr(writer, "write"):
            writer.write(f"Surrogate Model is NOT found!! path={path}, reason=resume_disabled") # 読込しない理由を端的にログへ残す
        return False, path # 呼び出し側に候補パスだけ返す
    if bool(getattr(args, "surrogate_pretrain_force_retrain", False)): # 強制再学習時は共有Surrogateを読まない
        path = surrogate_pretrain_load_path(args) # force retrain時でもログ用に候補パスを作る
        if writer is not None and hasattr(writer, "write"):
            writer.write(f"Surrogate Model is NOT found!! path={path}, reason=force_retrain") # 強制再学習で読まないことをログへ残す
        return False, path # 呼び出し側に候補パスだけ返す
    path = surrogate_pretrain_load_path(args) # argsのsurrogate_date_surrogate_timeで指定された共有Surrogateパスを取得する
    if path is None or not os.path.exists(path):
        if bool(getattr(args, "surrogate_registry_load_latest_if_missing", False)):
            latest_path = latest_surrogate_registry_path(args) # 完全一致が無い場合は同一dataの最新Surrogateを探す
            if latest_path is not None and latest_path != path and os.path.exists(latest_path):
                path = latest_path # 最新Surrogateを読込対象に切り替える
        if path is None or not os.path.exists(path):
            if writer is not None and hasattr(writer, "write"):
                writer.write(f"Surrogate Model is NOT found!! path={path}") # 指定Surrogateが無く1から模倣学習することをログへ残す
            return False, path # ファイルがなければ未読込として返す
    try:
        payload = torch.load(path, map_location="cpu") # GPUメモリを使わずCPUへ読み込む
    except Exception as exc:
        if writer is not None and hasattr(writer, "write"):
            writer.write(f"SurrogateRegistry load failed: path={path}, error={type(exc).__name__}: {exc}") # 読込失敗をログに残す
            writer.write(f"Surrogate Model is NOT found!! path={path}, reason=load_failed") # 読込失敗時も1から開始することをログへ残す
        return False, path # 壊れたファイルでは学習を止めず未読込扱いにする
    feature_dim = int(payload.get("surrogate_feature_dim", -1)) # 保存時の特徴次元を読む
    levels = list(payload.get("surrogate_levels", []) or []) # 保存時のOctree特徴levelを読む
    expected_dim = int(getattr(loss, "surrogate_feature_dim", 0) or 0) # 現在の特徴次元を読む
    expected_levels = list(getattr(loss, "surrogate_levels", []) or []) # 現在のOctree特徴levelを読む
    if feature_dim != expected_dim or levels != expected_levels:
        if writer is not None and hasattr(writer, "write"):
            writer.write(f"SurrogateRegistry skipped incompatible state: path={path}, feature_dim={feature_dim}->{expected_dim}, levels={levels}->{expected_levels}") # 不一致理由をログに残す
            writer.write(f"Surrogate Model is NOT found!! path={path}, reason=incompatible_state") # 互換性不一致で1から開始することをログへ残す
        return False, path # 構造が違う重みは読まない
    surrogate = getattr(loss, "compression_surrogate", None) # Loss内のSurrogateモデルを取得する
    state = payload.get("compression_surrogate_state_dict") # 保存済みstate_dictを取得する
    if surrogate is None or not isinstance(state, dict):
        if writer is not None and hasattr(writer, "write"):
            writer.write(f"Surrogate Model is NOT found!! path={path}, reason=missing_state_dict") # state_dict欠落で読めないことをログへ残す
        return False, path # 読み込めるSurrogate stateがなければ未読込扱いにする
    surrogate.load_state_dict(state, strict=False) # Surrogate本体へ重みを読み込む
    optimizer = getattr(loss, "surrogate_optimizer", None) # Surrogate用Optimizerを取得する
    opt_state = payload.get("surrogate_optimizer_state_dict") # 保存済みOptimizer状態を取得する
    if optimizer is not None and isinstance(opt_state, dict):
        optimizer.load_state_dict(opt_state) # 継続学習できるようOptimizer状態を復元する
    loss._surrogate_step = int(payload.get("surrogate_step", getattr(loss, "_surrogate_step", 0)) or 0) # Surrogate更新回数を復元する
    if writer is not None and hasattr(writer, "write"):
        writer.write(f"Surrogate Model is found!! path={path}") # 指定Surrogateを見つけて読み込んだことを端的にログへ残す
        writer.write(f"SurrogateRegistry loaded: path={path}, surrogate_step={int(getattr(loss, '_surrogate_step', 0))}, metric={payload.get('metric_name')}, value={payload.get('metric_value')}") # 読込結果をログに残す
    return True, path # 読込成功を返す


def maybe_save_best_surrogate_registry(args, loss, best_trackers, writer=None, *, metric_name, metric_value, source):
    try:
        value = float(metric_value) # 比較用にmetric値をfloat化する
    except (TypeError, ValueError, OverflowError):
        return None # 非数値metricでは保存判定しない
    if not math.isfinite(value):
        return None # NaN/Inf metricでは保存判定しない
    best_trackers.setdefault("surrogate_best_metric", float("inf")) # 初回用に最良metricを初期化する
    if value >= float(best_trackers["surrogate_best_metric"]):
        return None # 改善していなければ保存しない
    best_trackers["surrogate_best_metric"] = value # 改善したmetric値を記録する
    best_trackers["surrogate_best_metric_name"] = str(metric_name) # 改善判定に使ったmetric名を記録する
    return save_surrogate_registry_state(args, loss, writer, metric_name=metric_name, metric_value=value, source=source) # ベストSurrogateを指定パスへ保存する
