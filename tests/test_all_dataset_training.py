import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.utils.data.dataset import (
    activate_training_dataset_context,
    canonical_training_dataset_name,
    collect_seq_dirs2,
)
from models.utils.loss.compression import CompressionLossMixin
from models.utils.pointcloud.ana_den6_online import _identity
from models.utils.training.train_runtime import (
    _compression_primary_remaining_support_balance,
    _balance_actual_operation_head_gradients,
)


def _args():
    return SimpleNamespace(
        train_all_datasets=True,
        dataname="8i",
        sparsepcgc_native_bit_depth=10,
        sparsepcgc_scale_m=8,
        sparsepcgc_scale_ae=0,
        sparsepcgc_scale_sr=2,
        sparsepcgc_psnr_resolution=1023,
        sparsepcgc_dense_scale_sr_list=[2],
        sparsepcgc_voxel_size=1.0,
        sparsepcgc_pos_quantscale=1,
        sparsepcgc_mode="dense_lossy",
    )


class _Encoder:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _CompressionHarness(CompressionLossMixin):
    def __init__(self):
        self.actual_encoder = None
        self.actual_encoder_codec_key = None
        self.writer = None
        self.actual_gt_cache = {}
        self.surrogate_target_cache = {}
        self.last_surrogate_target_entry = None


class AllDatasetTrainingTest(unittest.TestCase):
    def test_private_loss_helper_is_explicitly_imported(self):
        self.assertTrue(callable(_compression_primary_remaining_support_balance))

    def test_dataset_context_keeps_shared_codec_setting(self):
        args = _args()
        activate_training_dataset_context(args, "UVG")
        self.assertEqual(args.dataname, "UVG")
        self.assertEqual(args.sparsepcgc_native_bit_depth, 10)
        self.assertEqual(args.sparsepcgc_scale_ae, 0)
        self.assertEqual(args.sparsepcgc_scale_sr, 2)
        self.assertEqual(args.sparsepcgc_psnr_resolution, 1023)
        self.assertEqual(args.sparsepcgc_dense_scale_sr_list, [2])
        identity = _identity(args, Path("frame.ply"), "abc")
        self.assertEqual(identity["dataset"], "UVG")
        self.assertEqual(identity["setting_id"], "vs1_pq1_ae0_sr2_m8")

        activate_training_dataset_context(args, "MVUB")
        self.assertEqual(args.dataname, "MVUB")
        self.assertEqual(args.sparsepcgc_scale_sr, 2)
        self.assertEqual(args.sparsepcgc_psnr_resolution, 1023)

    def test_codec_setting_is_never_overwritten(self):
        args = _args()
        args.sparsepcgc_native_bit_depth = 10
        args.sparsepcgc_scale_ae = 1
        args.sparsepcgc_scale_sr = 1
        args.sparsepcgc_psnr_resolution = 2047
        activate_training_dataset_context(args, "UVG")
        self.assertEqual(args.sparsepcgc_native_bit_depth, 10)
        self.assertEqual(args.sparsepcgc_scale_ae, 1)
        self.assertEqual(args.sparsepcgc_scale_sr, 1)
        self.assertEqual(args.sparsepcgc_psnr_resolution, 2047)

    def test_dataset_collection_order_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            for dataset, sequence in (("8i", "longdress"), ("MVUB", "andrew"), ("UVG", "Gymnast")):
                sequence_dir = Path(root) / dataset / sequence
                sequence_dir.mkdir(parents=True)
            ordered = []
            for dataset in ("8i", "MVUB", "UVG"):
                ordered.extend(collect_seq_dirs2(root, dataset_name=dataset))
            self.assertEqual(
                [Path(path).parent.name for path in ordered],
                ["8i", "MVUB", "UVG"],
            )

    def test_actual_encoder_is_reused_when_only_dataset_changes(self):
        args = _args()
        args.compress = "SparsePCGC"
        args.compression_loss_backend = "sparsepcgc_surrogate"
        harness = _CompressionHarness()
        created = []

        def _build(_args, writer=None):
            del writer
            encoder = _Encoder()
            created.append(encoder)
            return encoder

        with patch("models.utils.loss.compression.build_actual_encoder", side_effect=_build):
            first = harness._get_actual_encoder(args)
            activate_training_dataset_context(args, "UVG")
            second = harness._get_actual_encoder(args)
        self.assertEqual(len(created), 1)
        self.assertIs(first, second)
        self.assertFalse(first.closed)

    def test_supported_dataset_aliases(self):
        self.assertEqual(canonical_training_dataset_name("mvub"), "MVUB")
        self.assertEqual(canonical_training_dataset_name("uvg"), "UVG")

    def test_online_gradient_balance_never_amplifies_small_gradient(self):
        """online方策では収束を示す微小勾配をnorm targetまで増幅しない。"""
        import torch

        class _Actuator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.drop_head = torch.nn.Linear(1, 1, bias=False)
                self.add_head = torch.nn.Linear(1, 1, bias=False)
                self.add_voxel_head = torch.nn.Linear(1, 1, bias=False)
                self.move_voxel_head = torch.nn.Linear(1, 1, bias=False)
                self.drop_amount_head = torch.nn.Linear(1, 1, bias=False)
                self.add_amount_head = torch.nn.Linear(1, 1, bias=False)
                self.move_amount_head = torch.nn.Linear(1, 1, bias=False)
                self.algorithmic_amount_selector_head = torch.nn.Linear(1, 1, bias=False)
                self.algorithmic_amount_residual_head = torch.nn.Linear(1, 1, bias=False)
                self.operation_gate_head = torch.nn.Linear(1, 1, bias=False)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actuator = _Actuator()
                self.policy_module = torch.nn.Linear(1, 1, bias=False)

        model = _Model()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.01)
        args = SimpleNamespace(
            heuristic_guidance_mode="ana_den6_online",
            repair_balance_operation_head_grads=True,
            repair_operation_head_grad_target=1.0,
            minimal_loss_objective=True,
            repair_operation_head_grad_min_scale=1e-4,
            repair_operation_head_grad_max_scale=1e5,
        )
        before = model.actuator.drop_amount_head.weight.grad.clone()
        debug = _balance_actual_operation_head_gradients(args, model, {})
        self.assertTrue(torch.equal(before, model.actuator.drop_amount_head.weight.grad))
        self.assertEqual(debug["prune_amount_grad_balance_scale"], 1.0)
        self.assertGreater(
            debug["den6_online_amount_grad_norm_before_balance"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
