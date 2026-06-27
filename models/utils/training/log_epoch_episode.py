# models/utils/training/log_epoch_episode.py


def _extract_point_edit_values(edit_info):
    """
    点操作統計の plot_values から added/deleted/adjusted の3値だけを安全に取り出す。
    plot_values が4個以上ある場合は、既存ログに必要な先頭3個だけを使う。
    plot_values が3個未満の場合は None で補完する。
    """

    values = edit_info.get("plot_values", None)

    if values is None:
        return None, None, None

    # dict形式で渡された場合にも対応する
    if isinstance(values, dict):
        added = values.get("added_ratio", values.get("added", None))
        deleted = values.get("deleted_ratio", values.get("deleted", values.get("pruned", None)))
        adjusted = values.get("adjusted_ratio", values.get("adjusted", None))
        return added, deleted, adjusted

    # list / tuple 形式を想定する
    if isinstance(values, (list, tuple)):
        values = list(values)

        # 3個未満なら None で埋める
        if len(values) < 3:
            values = values + [None] * (3 - len(values))

        # 4個以上なら先頭3個だけ使う
        return values[0], values[1], values[2]

    # 想定外の型ならログだけ安全に0扱いにする
    return None, None, None


def log_epoch_point_edit_average(writer, epoch_edit_info, global_epoch):
    if epoch_edit_info.get("skipped", False):
        return

    added, deleted, adjusted = _extract_point_edit_values(epoch_edit_info)

    writer.write(
        "EpochPointEditAverage: "
        f"epoch={global_epoch + 1}, "
        f"added_ratio={0.0 if added is None else float(added):.4f}%, "
        f"deleted_ratio={0.0 if deleted is None else float(deleted):.4f}%, "
        f"adjusted_ratio={0.0 if adjusted is None else float(adjusted):.4f}%, "
        f"steps={int(epoch_edit_info.get('accepted_steps', 0))}"
    )


def log_episode_point_edit_average(writer, episode_edit_info, episode):
    if episode_edit_info.get("skipped", False):
        return

    added, deleted, adjusted = _extract_point_edit_values(episode_edit_info)

    writer.write(
        "EpisodePointEditAverage: "
        f"episode={episode + 1}, "
        f"added_ratio={0.0 if added is None else float(added):.4f}%, "
        f"deleted_ratio={0.0 if deleted is None else float(deleted):.4f}%, "
        f"adjusted_ratio={0.0 if adjusted is None else float(adjusted):.4f}%, "
        f"steps={int(episode_edit_info.get('accepted_steps', 0))}"
    )


def log_plot_skip_epoch(writer, plot_epoch_info, global_epoch):
    if not plot_epoch_info.get("skipped", False):
        return

    writer.write(
        "PlotSkipEpoch: "
        f"epoch={global_epoch + 1}, "
        f"reason={plot_epoch_info.get('reason', 'unknown')}"
    )


def log_plot_skip_episode(writer, plot_episode_info, episode):
    if not plot_episode_info.get("skipped", False):
        return

    writer.write(
        "PlotSkipEpisode: "
        f"episode={episode + 1}, "
        f"reason={plot_episode_info.get('reason', 'unknown')}"
    )