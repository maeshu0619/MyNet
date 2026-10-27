import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch

from models.utils.training.episode_frames import configure_fixed_episode_frames
from models.utils.training.train_flow import apply_epoch_file_window
from models.utils.training.sparsepcgc_actual_credit import matched_sparsepcgc_dataset


class FixedEpisodeFramesTest(unittest.TestCase):
    def _datasets(self, sizes):
        return [(str(index), SimpleNamespace(
            all_files=[f"/sequence{index}/frame{j}.ply" for j in range(1, count + 1)],
            files=[f"/sequence{index}/frame1.ply"],
        )) for index, count in enumerate(sizes)]

    def _args(self):
        return SimpleNamespace(train_fixed_episode_frames=100, train_frames_per_sequence=100,
                               max_files=10, compress="SparsePCGC", trainORtest="train",
                               heuristic_guidance_mode="ana_den6_online",
                               sparsepcgc_matched_frame_repeats=2)

    def test_four_sequences_first_25_frames_constant_for_all_episodes(self):
        args = self._args()
        original = self._datasets([160] * 4)
        original_all = [list(dataset.all_files) for _, dataset in original]
        rng = torch.random.get_rng_state().clone()
        datasets, manifest = configure_fixed_episode_frames(original, args)
        self.assertEqual([row["frames"] for row in manifest["sequences"]], [25] * 4)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for episode in (0, 1, 20, 95, 381):
            for index, (_, dataset) in enumerate(datasets):
                apply_epoch_file_window(dataset, args, episode)
                self.assertEqual(dataset.files, original_all[index][:25])
                self.assertEqual(dataset.all_files, original_all[index])
            self.assertEqual(sum(len(ds.files) for _, ds in datasets), 100)

    def test_three_sequences_distribute_remainder(self):
        datasets, manifest = configure_fixed_episode_frames(self._datasets([150] * 3), self._args())
        self.assertEqual([row["frames"] for row in manifest["sequences"]], [34, 33, 33])
        # Even a legacy repeat=2 configuration must yield exactly 100 steps.
        self.assertEqual(sum(len(matched_sparsepcgc_dataset(ds, self._args()).files)
                             for _, ds in datasets), 100)

    def test_multiple_datasets_each_receive_100_steps(self):
        datasets = self._datasets([150] * 9)
        for index, (_, dataset) in enumerate(datasets):
            dataset.training_dataset_name = "8i" if index < 4 else "MVUB"
        selected, manifest = configure_fixed_episode_frames(datasets, self._args())
        self.assertEqual(manifest["unique_frames_per_episode"], 200)
        self.assertEqual([row["sequence_steps"] for row in manifest["datasets"]],
                         [[25] * 4, [20] * 5])
        counts = {}
        for _, dataset in selected:
            active = matched_sparsepcgc_dataset(dataset, self._args())
            self.assertIs(active, dataset)
            name = dataset.training_dataset_name
            counts[name] = counts.get(name, 0) + len(active.files)
        self.assertEqual(counts, {"8i": 100, "MVUB": 100})

    def test_a_short_dataset_cannot_borrow_another_dataset_frames(self):
        datasets = self._datasets([150, 99])
        datasets[0][1].training_dataset_name = "8i"
        datasets[1][1].training_dataset_name = "MVUB"
        with self.assertRaisesRegex(ValueError, "MVUB needs 100.*only 99"):
            configure_fixed_episode_frames(datasets, self._args())

    def test_pending_actual_comparison_can_resume_next_episode(self):
        from models.modules.structure_actuator import StructureRepairActuator
        for revisit_step in (100, 300):
            args = self._args()
            args._global_train_step = revisit_step
            args._current_input_file = "/sequence/frame.ply"
            args.heuristic_guidance_online_simultaneous_operation_exploration = True
            key = "/sequence/frame.ply|0|0|8"
            owner = SimpleNamespace(
                args=args, training=True, _den6_factorized_group_counts={},
                _den6_pending_matched_groups=OrderedDict([(key, {
                    "created_step": 0, "group": "Amount", "attempts": 0,
                })]),
            )
            chosen = StructureRepairActuator._select_den6_factorized_exploration_group(owner, {})
            self.assertEqual(chosen, "Amount")
            self.assertEqual(owner._den6_pending_matched_groups[key]["attempts"], 1)

    def test_capacity_redistribution_and_no_validation_leak(self):
        datasets, manifest = configure_fixed_episode_frames(self._datasets([2, 160, 160]), self._args())
        self.assertEqual([row["frames"] for row in manifest["sequences"]], [2, 49, 49])
        for _, ds in datasets:
            self.assertFalse(set(ds.files) & set(ds.all_files[100:]))

    def test_insufficient_capacity_fails_without_repetition(self):
        with self.assertRaisesRegex(ValueError, "only 99 available"):
            configure_fixed_episode_frames(self._datasets([33] * 3), self._args())

    def test_duplicates_rejected_and_legacy_mode_available(self):
        datasets = self._datasets([100])
        datasets[0][1].all_files.append(datasets[0][1].all_files[0])
        args = self._args()
        args.train_frames_per_sequence = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            configure_fixed_episode_frames(datasets, args)
        args.train_fixed_episode_frames = 0
        self.assertIsNone(configure_fixed_episode_frames(datasets, args)[1])


if __name__ == "__main__":
    unittest.main()
