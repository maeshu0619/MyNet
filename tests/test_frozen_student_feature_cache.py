from pathlib import Path
import tempfile
import unittest

import torch

from models.utils.cache.frozen_student_feature_cache import (
    FrozenStudentFeatureCache,
    feature_fingerprint,
)


class FrozenStudentFeatureCacheTest(unittest.TestCase):
    def test_checkpoint_change_invalidates_key_and_hit_is_exact(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "encoder.pth"
            source = Path(root) / "encoder.py"
            checkpoint.write_bytes(b"checkpoint-a")
            source.write_text("x=1\n", encoding="utf-8")
            first = feature_fingerprint(
                layer_a_key="a", encoder_checkpoint=str(checkpoint),
                encoder_source=str(source), dtype="float32", receptive_field=2,
                tile_partition={"size": 16},
            )
            cache = FrozenStudentFeatureCache(str(Path(root) / "cache"))
            value = torch.randn(2, 3)
            cache.write(first, {"local": value})
            self.assertTrue(torch.equal(cache.load(first)["local"], value))
            checkpoint.write_bytes(b"checkpoint-b")
            second = feature_fingerprint(
                layer_a_key="a", encoder_checkpoint=str(checkpoint),
                encoder_source=str(source), dtype="float32", receptive_field=2,
                tile_partition={"size": 16},
            )
            self.assertNotEqual(first["fingerprint_sha256"], second["fingerprint_sha256"])
            with self.assertRaises(FileNotFoundError):
                cache.load(second)


if __name__ == "__main__":
    unittest.main()

