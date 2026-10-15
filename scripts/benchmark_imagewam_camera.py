"""Compare camera preprocessing alone, with exact pixel and mosaic checks."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from utils.imagewam_camera import (
    model_input, prepare_model_input, preprocess_cameras, preprocess_rgb,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-npz', required=True)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with np.load(args.observation_npz) as observation:
        frames = [observation[key] for key in ('image', 'image_wrist_left', 'image_wrist_right')]

    for crop in (100, 90):
        for frame in frames:
            original = preprocess_rgb(np.ascontiguousarray(frame[..., ::-1]), crop)
            optimized = preprocess_rgb(frame, crop, bgr=True)
            np.testing.assert_array_equal(original, optimized)

    def original():
        rgb = [np.ascontiguousarray(frame[..., ::-1]) for frame in frames]
        return model_input(preprocess_cameras(rgb))

    reference = original()
    rows = []
    for index in range(args.warmup + args.requests):
        mode = ('original', 'optimized', 'optimized', 'original')[index % 4]
        started = time.perf_counter()
        image = original() if mode == 'original' else prepare_model_input(frames)
        elapsed = (time.perf_counter() - started) * 1000
        np.testing.assert_array_equal(image, reference)
        if index >= args.warmup:
            rows.append({'mode': mode, 'ms': elapsed})
    summary = {}
    for mode in ('original', 'optimized'):
        values = [row['ms'] for row in rows if row['mode'] == mode]
        summary[mode] = {'n': len(values), 'mean': float(np.mean(values)),
                         'p50': float(np.median(values)), 'p95': float(np.percentile(values, 95))}
    report = {'observation': args.observation_npz, 'pixels_equal': True,
              'mosaic_equal': True, 'summary': summary, 'rows': rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
