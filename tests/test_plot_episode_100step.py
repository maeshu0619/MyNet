import os
import tempfile
import unittest
from types import SimpleNamespace

from models.utils.training.metric_csv import init_metric_csvs
from record.plot import PlotMaker


class _Writer:
    def write(self, _message):
        return None


def _args(root, window=100):
    return SimpleNamespace(
        log_root=root,
        date="20260810",
        compress="SparsePCGC",
        time="unit",
        plot_step_average_window=window,
        save_step_metric_csv=False,
        save_compression_metric_csv=True,
        save_operation_metric_csv=True,
        save_checkpoint_metric_csv=True,
        surrogate_pretrain_steps=0,
        loss_grad_probe_enabled=False,
        sparsepcgc_algorithmic_proposal_selector=True,
        sparsepcgc_training_mode="subtree_selector",
    )


class EpisodeAndHundredStepPlotTest(unittest.TestCase):
    def test_episode_dataset_weight_keeps_100step_chronological(self):
        with tempfile.TemporaryDirectory() as root:
            plot = PlotMaker(_args(root, window=2))
            for step, value, weight in ((1, 1.0, 3.0), (2, 3.0, 1.0)):
                metrics = [value] * plot.num_loss
                plot.record_metrics(
                    "step", step, metrics, episode_weight=weight
                )
                plot.record_point_edits(
                    "step",
                    step,
                    {
                        "added_ratio_percent": value,
                        "deleted_ratio_percent": value,
                        "adjusted_ratio_percent": value,
                    },
                    episode_weight=weight,
                )
            metric_info = plot.record_metrics(
                "epi", 1, [0.0] * plot.num_loss
            )
            edit_info = plot.record_point_edits("epi", 1)
            self.assertAlmostEqual(metric_info["plot_values"][0], 1.5)
            self.assertAlmostEqual(edit_info["plot_values"][0], 1.5)
            self.assertAlmostEqual(plot.step100_loss_his[0][0], 2.0)
            self.assertAlmostEqual(plot.step100_edit_his[0][0], 2.0)

    def test_step_values_are_kept_only_as_episode_and_block_average(self):
        with tempfile.TemporaryDirectory() as root:
            plot = PlotMaker(_args(root))
            for step in range(1, 101):
                values = [float(step)] + [1.0] * (plot.num_loss - 1)
                plot.record_metrics("step", step, values)
                plot.record_point_edits(
                    "step",
                    step,
                    {
                        "added_ratio_percent": float(step),
                        "deleted_ratio_percent": 2.0,
                        "adjusted_ratio_percent": 3.0,
                    },
                )

            self.assertEqual(plot.step_x_his, [])
            self.assertEqual(plot.epo_x_his, [])
            self.assertEqual(plot.step100_x_his, [1])
            self.assertAlmostEqual(plot.step100_loss_his[0][0], 50.5)
            self.assertAlmostEqual(plot.step100_edit_his[0][0], 50.5)
            self.assertEqual(
                plot.group_title,
                ["compression", "other", "repair", "repair_policy"],
            )
            self.assertEqual(plot.title_group[0], [2, 3, 9, 8])
            self.assertEqual(plot.title[2], "Surrogate Compression Delta")
            self.assertEqual(plot.title[3], "Actual Compression Delta")

    def test_default_metric_csvs_are_episode_only(self):
        with tempfile.TemporaryDirectory() as root:
            args = _args(root)
            plot = PlotMaker(args)
            paths = init_metric_csvs(args, plot, _Writer())
            self.assertIsNone(paths["compression_step"])
            self.assertIsNone(paths["operation_step"])
            self.assertIsNone(paths["proposal_candidate_step"])
            self.assertTrue(os.path.exists(paths["compression_episode"]))
            self.assertTrue(os.path.exists(paths["operation_episode"]))
            self.assertTrue(os.path.exists(paths["checkpoint_episode"]))

    def test_generated_files_match_requested_groups(self):
        with tempfile.TemporaryDirectory() as root:
            plot = PlotMaker(_args(root, window=2))
            for step in (1, 2):
                values = [float(step)] * plot.num_loss
                plot.record_metrics("step", step, values)
                plot.record_point_edits(
                    "step",
                    step,
                    {
                        "added_ratio_percent": 0.1,
                        "deleted_ratio_percent": 0.1,
                        "adjusted_ratio_percent": 0.05,
                    },
                )
            plot.record_metrics("epi", 1, [1.5] * plot.num_loss)
            plot.record_point_edits("epi", 1)
            plot.plot_loss_curve("epi")
            plot.plot_point_edit_curve("epi")
            plot.plot_loss_curve("step100")
            plot.plot_point_edit_curve("step100")

            names = set(os.listdir(plot.save_dir))
            for mode in ("epi", "100step"):
                for group in ("compression", "other", "repair", "repair_policy"):
                    self.assertIn(f"unit_{mode}_{group}.png", names)
                self.assertIn(f"unit_{mode}_point_edits.png", names)
            self.assertIn("unit_epi_metrics.csv", names)
            self.assertIn("unit_epi_point_edits.csv", names)
            self.assertNotIn("unit_100step_metrics.csv", names)
            self.assertFalse(any("occupancy" in name for name in names))
            self.assertFalse(any("voxel_collision" in name for name in names))
            self.assertFalse(any("_step_metrics.csv" in name for name in names))
            self.assertFalse(any("_epo_" in name for name in names))


if __name__ == "__main__":
    unittest.main()
