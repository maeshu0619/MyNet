"""ana_den6 online workerの同値高速化テスト。"""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tools.ana_den6_online_worker import _baseline_codec_rate_only


class _FakeBase:
    @staticmethod
    def _safe_name(path):
        return Path(path).stem

    @staticmethod
    def _ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    @staticmethod
    def _normalize_coder_result(raw):
        return {
            "decoder_complete_bits": raw["file_size"],
            "bpp": raw["bpp"],
        }


class _FakeCoder:
    def __init__(self):
        self.test_psnr = None

    def test(self, input_file, bin_file, decoded_file, **kwargs):
        del input_file
        self.test_psnr = kwargs["test_psnr"]
        Path(bin_file).write_bytes(b"0123456789")
        Path(decoded_file).write_text("ply\n", encoding="utf-8")
        return {"file_size": 104, "bpp": 1.0}


class AnaDen6OnlineWorkerTest(unittest.TestCase):
    def test_rate_only_baseline_keeps_exact_bits_without_psnr(self):
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "input.ply"
            input_file.write_text("ply\n", encoding="utf-8")
            coder = _FakeCoder()
            result = _baseline_codec_rate_only(
                _FakeBase,
                coder,
                input_file,
                SimpleNamespace(
                    setting_id="native",
                    scale_ae=0,
                    scale_sr=2,
                    scale_m=8,
                    psnr_resolution=1023,
                ),
                SimpleNamespace(),
                Path(directory) / "out",
            )

        self.assertFalse(coder.test_psnr)
        self.assertEqual(result["decoder_complete_bits"], 104)
        self.assertEqual(result["main_bin_bits"], 80.0)
        self.assertEqual(result["side_information_bits"], 24.0)
        self.assertEqual(
            result["formal_metric_status"],
            "skipped_for_online_plan_generation",
        )


if __name__ == "__main__":
    unittest.main()
