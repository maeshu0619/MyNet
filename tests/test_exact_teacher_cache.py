import json
from pathlib import Path
import tempfile
import unittest

from models.utils.cache.exact_teacher_cache import ExactTeacherCache, build_fingerprint


class ExactTeacherCacheTest(unittest.TestCase):
    def test_atomic_hit_and_corrupt_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            input_path = Path(root) / "input.ply"
            source_path = Path(root) / "teacher.py"
            input_path.write_bytes(b"ply")
            source_path.write_text("x=1\n", encoding="utf-8")
            fingerprint = build_fingerprint(
                input_path=str(input_path), codec={"m": 8},
                source_files=[str(source_path)], geometry={"d1": True},
            )
            cache = ExactTeacherCache(str(Path(root) / "cache"))
            path = cache.write(fingerprint, {"actual_plans": [{"gain": 1.0}]})
            self.assertEqual(cache.load(fingerprint)["content"]["actual_plans"][0]["gain"], 1.0)
            # immutable hitで既存内容を上書きしない。
            cache.write(fingerprint, {"actual_plans": []})
            self.assertEqual(cache.load(fingerprint)["content"]["actual_plans"][0]["gain"], 1.0)
            path.write_bytes(b"broken")
            with self.assertRaises(RuntimeError):
                cache.load(fingerprint)


if __name__ == "__main__":
    unittest.main()

