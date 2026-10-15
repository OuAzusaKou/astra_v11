import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/evaluate_imagewam_mano_server.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_imagewam_mano_server", MODULE_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


class ImageWAMManoServerEvaluationTest(unittest.TestCase):
    def test_split_metrics_reports_macro_and_micro_mse(self):
        pred = np.zeros((2, 1, 326), dtype=np.float32)
        gt_physical = np.zeros_like(pred)
        gt_normalized = np.zeros_like(pred)
        valid = np.zeros_like(pred, dtype=bool)
        valid[0, 0, 0:9] = True
        valid[1, 0, 0:9] = True
        valid[0, 0, 10:100] = True
        valid[1, 0, 10:100] = True
        pred[0, 0, valid[0, 0]] = 1.0
        pred[1, 0, valid[1, 0]] = 3.0
        scale = np.ones(326, dtype=np.float32)
        offset = np.zeros(326, dtype=np.float32)
        latency = np.asarray([1.0, 3.0])

        metrics = EVALUATOR._split_metrics(
            pred,
            gt_physical,
            gt_normalized,
            valid,
            scale,
            offset,
            latency,
            latency,
            latency,
        )

        self.assertAlmostEqual(
            metrics["loss"]["normalized_unweighted_mse_macro"], 5.0
        )
        self.assertAlmostEqual(
            metrics["loss"]["normalized_unweighted_mse_micro"], 5.0
        )
        expected_weighted = 5.0 * (9.0 + 90.0 * 0.5) / 99.0
        self.assertAlmostEqual(
            metrics["loss"]["normalized_weighted_mse_ee1_mano05_macro"],
            expected_weighted,
        )
        self.assertAlmostEqual(
            metrics["loss"]["physical_mixed_unit_mse_micro"], 5.0
        )

    def test_load_samples_rejects_missing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            np.savez(path, split=np.asarray(["train", "eval"]))
            with self.assertRaisesRegex(ValueError, "missing keys"):
                EVALUATOR._load_samples(path, expected_per_split=1)

    def test_load_samples_accepts_horizon_shaped_normalizer(self):
        count = 2
        action_shape = (count, 32, 326)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            np.savez_compressed(
                path,
                split=np.asarray(["train", "eval"]),
                dataset_name=np.asarray("dex_bench_2"),
                instruction=np.asarray(["a", "b"]),
                images_bgr=np.zeros((count, 3, 480, 640, 3), dtype=np.uint8),
                left_ee_pose=np.zeros((count, 6), dtype=np.float32),
                right_ee_pose=np.zeros((count, 6), dtype=np.float32),
                left_hand_joint=np.zeros((count, 20), dtype=np.float32),
                right_hand_joint=np.zeros((count, 20), dtype=np.float32),
                gt_action_normalized=np.zeros(action_shape, dtype=np.float32),
                gt_action_physical=np.zeros(action_shape, dtype=np.float32),
                action_valid_mask=np.ones(action_shape, dtype=bool),
                action_scale=np.ones((32, 326), dtype=np.float32),
                action_offset=np.zeros((32, 326), dtype=np.float32),
                episode_index=np.arange(count),
                frame_index=np.arange(count),
                target_frame_index=np.arange(count),
            )
            loaded = EVALUATOR._load_samples(path, expected_per_split=1)
            self.assertEqual(loaded["action_scale"].shape, (32, 326))


if __name__ == "__main__":
    unittest.main()
