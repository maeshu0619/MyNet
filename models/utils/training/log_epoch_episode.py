# models/utils/training/log_epoch_episode.py


def log_epoch_point_edit_average(writer, epoch_edit_info, global_epoch):
    if epoch_edit_info.get("skipped", False):
        return

    added, deleted, adjusted = epoch_edit_info.get("plot_values", [None, None, None])

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

    added, deleted, adjusted = episode_edit_info.get("plot_values", [None, None, None])

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