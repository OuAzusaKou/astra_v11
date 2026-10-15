# D435i depth judgment for drawer tests

Add `--depth-camera auto` to the leo3 runner command for a new V2 session. With multiple D435 cameras, replace `auto` with the serial returned by `rs-enumerate-devices -s`. The normal Mac bridge command is unchanged. Close other programs streaming from this camera first. The runner prints `[DEPTH READY]` with the selected SDK serial.

The runner owns one continuous 640 by 480, 30 fps RGB/depth stream. Only Astra review snapshots gain the extra RealSense RGB and fixed-scale depth preview. Existing WAM observations and model input remain unchanged. Depth acquisition stops when the runner exits.

Startup waits for a paired RGB/depth frame. The SDK can initially deliver an unmatched pair; frames exceeding the 50 ms pair limit are skipped rather than announcing readiness or replacing a usable frame. If valid pairs stop arriving, the existing frame-age check marks the camera unavailable.

Raw depth is aligned to the RealSense color image and stored as lossless 16-bit PNG, alongside a valid-pixel mask and metadata. Depth pixels do not correspond to any of the original three RGB camera views. No intrinsics-based XYZ calculation, camera-to-robot extrinsic calibration or target conversion is performed.

Black preview pixels and zero raw values are unknown; no hole filling is applied. The preview always maps 0.15 m to red and 2.0 m to blue; out-of-range values are clipped only in the preview. Quantitative judgments must use raw-depth ROI queries rather than colors.

The metadata contains sensor timestamps/domain, host reception time, frame number, depth scale, camera serial and approximate robot-observation time difference. Frames older than 0.5 s, robot observation offsets above 0.5 s, or paired RGB/depth offsets above 50 ms are reported unavailable rather than silently reused. Host reception is not hardware synchronization or an exact exposure timestamp.

Depth is selected and validated before any of the three original RGB images are encoded. Encoding delay must not advance the selected camera frame relative to the robot observation. Metadata separates selection time, depth save completion and total snapshot processing duration; rejected timing checks retain their measured age/skew. Each review prints `[DEPTH] available` with frame/age/skew or `[DEPTH] unavailable` with the reason, and records a `depth_snapshot` event. `[DEPTH READY]` at startup alone does not confirm that a review received usable depth.

The Mac transport checks raw depth, mask and image SHA256 and writes local metadata with local file paths. Each Astra request includes an executable read-only query command:

```bash
python3 -B astra/depth_query.py --metadata /absolute/observation/depth_metadata.json \
  --roi 100 100 110 110 --roi 150 100 160 110
```

Coordinates are integer pixels in the attached RealSense RGB image, origin top-left, upper bounds exclusive. Select small patches on the intended visible surfaces. Output includes median, 10th/90th percentiles, valid fraction and mixed-depth warnings; holes and mixed foreground/background must not be treated as contact geometry. Ordering requires at least five valid pixels and 50 percent coverage per ROI, non-mixed patches and non-overlapping percentile ranges with a 1 cm margin. These are conservative heuristics, not calibrated accuracy guarantees or blanket motion/pause rules.

Depth is distance along the camera optical axis, not Euclidean separation, insertion distance, robot-axis displacement, contact force or proof of hooking. Cross-frame comparisons require the same camera, unchanged viewpoint/background and correspondence of the same visible surface. Moving the camera invalidates that comparison. Lack of camera-to-robot calibration alone must not trigger a pause in this mode.

The operator still uses Enter/T/P/R/B as before. Keep planning around meaningful operation-stage endpoints using the existing measured-state/control mapping; use depth to resolve a specific near/far question and check visible progress. Never derive the 52-dimensional robot target directly from depth values. A local invalid-depth region does not invalidate otherwise supported RGB/state-based actions.
