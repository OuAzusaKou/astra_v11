"""Real encoder/decoder checks using synthetic frames, without ROS or robot control."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from progress_vlm.recording import WindowRecorder


class RecordingTests(unittest.TestCase):
    def test_playback_timing_segments_and_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "video"
            recorder = WindowRecorder(output, fps=10, segment_seconds=1)
            a = np.full((64, 96, 3), 30, dtype=np.uint8)
            b = np.full_like(a, 220)
            recorder.write(a, now=100.)
            recorder.write(a, now=100.03)  # No extra frame for fast redraws.
            recorder.write(b, now=101.2)  # Fill the gap at wall-clock speed.
            recorder.close()
            recorder.close()
            counts, means = [], []
            for path in sorted(output.glob("*.mp4")):
                capture = cv2.VideoCapture(str(path))
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10)
                count = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    count += 1
                    means.append(frame.mean())
                capture.release()
                counts.append(count)
            self.assertEqual(counts, [10, 3])
            self.assertLess(max(means[:-1]), 40)
            self.assertGreater(means[-1], 200)
            events = [json.loads(line) for line in (output / "segments.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["opened", "closed", "opened", "closed"])


if __name__ == "__main__":
    unittest.main()
