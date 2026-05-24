def optimizer_lrs_safe(optimizer):
    if optimizer is None:  # Optimizerが無い場合は空のLR一覧を返す
        return []  # ログ側でNA扱いにできるよう空リストにする
    return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]  # 各param groupの現在LRを取得する


def _lr_floor_for_label(args, label):
    if str(label).strip().lower() == "surrogate":  # Surrogate optimizer用のfloorか判定する
        return max(float(getattr(args, "min_surrogate_lr", 1e-6)), 0.0)  # Surrogate LRの下限値を返す
    return max(float(getattr(args, "min_main_lr", 1e-5)), 0.0)  # main optimizer LRの下限値を返す


def apply_optimizer_lr_floor(optimizer, args, *, label="main", writer=None, global_step=None, reason=""):
    before = optimizer_lrs_safe(optimizer)  # floor適用前のLR一覧を記録する
    floor = _lr_floor_for_label(args, label)  # 対象optimizerに対応するLR floorを取得する
    applied = False  # floorが実際に適用されたかを保存する
    if optimizer is not None and floor > 0.0:  # Optimizerが存在し、floorが有効な場合だけ処理する
        for group in optimizer.param_groups:  # 各param groupのLRを個別に確認する
            lr = float(group.get("lr", 0.0))  # 現在のparam group LRを取得する
            if lr < floor:  # LRがfloorを下回ったか判定する
                group["lr"] = floor  # LRが潰れないようfloor値へ戻す
                applied = True  # floor適用済みとして記録する
    after = optimizer_lrs_safe(optimizer)  # floor適用後のLR一覧を記録する
    if applied and writer is not None and hasattr(writer, "write"):  # floor適用時だけwarningログを出す
        writer.write(
            "LRFloorWarning: "
            f"label={label}, global_step={global_step}, floor={floor:.6g}, "
            f"before={before}, after={after}, reason={reason}"
        )  # LRがfloorへ張り付いた事実と理由を記録する
    return {
        "lr_floor_applied": bool(applied),  # floor適用有無を呼び出し側へ返す
        "lr_floor": float(floor),  # 適用対象のfloor値を返す
        "lr_before_floor": before,  # floor適用前LRを返す
        "lr_after_floor": after,  # floor適用後LRを返す
    }


def step_scheduler_with_floor(scheduler, optimizer, args, *, writer=None, global_epoch=None, global_step=None):
    enabled = bool(getattr(args, "lr_scheduler_enabled", False))  # StepLRを使う設定か確認する
    before = optimizer_lrs_safe(optimizer)  # scheduler前のLR一覧を記録する
    event = {
        "scheduler_stepped": False,  # scheduler.step()が実行されたかを保存する
        "scheduler_lr_before": before,  # scheduler前LRを保存する
        "scheduler_lr_after": before,  # scheduler後LRを初期値として保存する
        "lr_floor_applied": False,  # scheduler後floor適用の有無を初期化する
        "scheduler_disabled": not enabled,  # schedulerが設定で止まっているか保存する
    }
    if enabled and scheduler is not None:  # scheduler有効かつ実体がある場合だけstepする
        scheduler.step()  # Epoch単位のStepLRを1回進める
        event["scheduler_stepped"] = True  # scheduler実行済みとして記録する
    after_scheduler = optimizer_lrs_safe(optimizer)  # scheduler直後のLR一覧を取得する
    floor_event = apply_optimizer_lr_floor(
        optimizer,
        args,
        label="main",
        writer=writer,
        global_step=global_step,
        reason="scheduler_step" if event["scheduler_stepped"] else "scheduler_disabled",
    )  # scheduler後にmain LR floorを必ず適用する
    event.update(
        {
            "scheduler_lr_after": after_scheduler,  # scheduler自体が出したLRを保存する
            "scheduler_lr_after_floor": floor_event["lr_after_floor"],  # floor反映後LRを保存する
            "lr_floor_applied": bool(floor_event["lr_floor_applied"]),  # floor適用有無を保存する
            "lr_floor": float(floor_event["lr_floor"]),  # floor値を保存する
            "global_epoch": global_epoch,  # scheduler発火epochを保存する
            "global_step": global_step,  # scheduler発火時点のglobal stepを保存する
        }
    )
    return event
