"""Read-only ROI optical-depth statistics. Never deprojects XYZ or controls a robot."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def query(metadata_path, rois):
    path = Path(metadata_path)
    info = json.loads(path.read_text())
    if info.get("status") != "available":
        raise ValueError("depth observation unavailable")
    file = Path(info["raw"]["path"])
    if hashlib.sha256(file.read_bytes()).hexdigest() != info["raw"]["sha256"]:
        raise ValueError("raw depth checksum mismatch")
    with Image.open(file) as image:
        raw = np.asarray(image)
    if raw.ndim != 2 or raw.shape != (info["height"], info["width"]) or raw.dtype.kind not in "ui":
        raise ValueError("raw depth geometry/type mismatch")
    if np.any(raw < 0) or np.any(raw > 65535):
        raise ValueError("raw depth outside Z16 range")
    scale = float(info["depth_scale_m"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid depth scale")
    result = {"metadata_path": str(path.resolve()), "camera_serial": info["camera_serial"],
              "frame_number": info["frame_number"], "host_received_at": info["host_received_at"],
              "units": "meters_along_camera_optical_axis", "view": "realsense_rgb", "regions": [],
              "interpretation": "Smaller depth is nearer this camera. Differences are not XYZ, Euclidean clearance, robot motion, or contact evidence. Each ROI must isolate the intended visible surface."}
    for roi in rois:
        x0, y0, x1, y1 = roi
        if not 0 <= x0 < x1 <= info["width"] or not 0 <= y0 < y1 <= info["height"]:
            raise ValueError("ROI outside aligned RealSense RGB image")
        crop = raw[y0:y1, x0:x1]
        values = crop[crop > 0].astype(np.float64) * scale
        fraction = values.size / crop.size
        row = {"roi": list(roi), "valid_pixels": int(values.size), "total_pixels": int(crop.size),
               "valid_fraction": fraction, "usable": bool(values.size >= 5 and fraction >= .5)}
        if values.size:
            low, median, high = np.percentile(values, [10, 50, 90])
            row.update(p10_m=float(low), median_m=float(median), p90_m=float(high),
                       mixed_depth_warning=bool(high - low > .04))
        else:
            row["reason"] = "no valid depth; occlusion/holes are unknown, not zero distance"
        result["regions"].append(row)
    if len(result["regions"]) == 2:
        a, b = result["regions"]
        comparison = {"ordering": "uncertain"}
        if a["usable"] and b["usable"]:
            comparison["median_a_minus_b_m"] = a["median_m"] - b["median_m"]
            if not a["mixed_depth_warning"] and not b["mixed_depth_warning"]:
                if a["p90_m"] + .01 < b["p10_m"]:
                    comparison["ordering"] = "region_a_nearer_camera"
                elif b["p90_m"] + .01 < a["p10_m"]:
                    comparison["ordering"] = "region_b_nearer_camera"
        comparison["note"] = "Heuristic ordering with 1 cm separation margin, not a calibrated accuracy guarantee. Unknown ordering does not require pausing all task motion."
        result["comparison"] = comparison
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--roi", nargs=4, type=int, action="append", required=True,
                        metavar=("X0", "Y0", "X1", "Y1"), help="RealSense RGB pixel box, end exclusive; repeat to compare")
    args = parser.parse_args()
    try:
        print(json.dumps(query(args.metadata, args.roi), ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
