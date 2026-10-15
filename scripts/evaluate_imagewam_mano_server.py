#!/usr/bin/env python3
"""Evaluate a teleop_mano ImageWAM through the deployed client/server wire path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np

from policy_server.msgpack_numpy import packb, unpackb
from utils.liber_teleop_v22_client import (
    add_client_timing,
    build_request,
    decode_action,
    prepare_hand_action_decoder,
)


EE_SLICES = (slice(0, 9), slice(125, 134))
MANO_SLICES = (slice(10, 100), slice(135, 225))
WIRE_IMAGE_KEYS = (
    "observation.images.image_ego_0",
    "observation.images.image_wrist_left",
    "observation.images.image_wrist_right",
)
REQUIRED_SAMPLE_KEYS = {
    "split",
    "dataset_name",
    "instruction",
    "images_bgr",
    "left_ee_pose",
    "right_ee_pose",
    "left_hand_joint",
    "right_hand_joint",
    "gt_action_normalized",
    "gt_action_physical",
    "action_valid_mask",
    "action_scale",
    "action_offset",
    "episode_index",
    "frame_index",
    "target_frame_index",
}


class PersistentPolicyClient:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.sock: socket.socket | None = None

    def __enter__(self):
        self.sock = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        self.sock.settimeout(self.timeout)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _recv_exact(self, size: int) -> bytes:
        if self.sock is None:
            raise RuntimeError("client is not connected")
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(size - received)
            if not chunk:
                raise ConnectionError("server closed the persistent connection")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.sock is None:
            raise RuntimeError("client is not connected")
        started = time.perf_counter()
        payload = packb(request)
        self.sock.sendall(len(payload).to_bytes(4, "big"))
        self.sock.sendall(payload)
        response_size = int.from_bytes(self._recv_exact(4), "big")
        response = unpackb(self._recv_exact(response_size))
        if not isinstance(response, dict):
            raise TypeError(f"server response must be a dict, got {type(response)}")
        add_client_timing(response, (time.perf_counter() - started) * 1000.0)
        return response


def _percentiles(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "p50": float(np.quantile(value, 0.50)),
        "p95": float(np.quantile(value, 0.95)),
        "max": float(value.max()),
    }


def _row6_to_matrix(row6: np.ndarray) -> np.ndarray:
    value = np.asarray(row6, dtype=np.float64)
    row0 = value[..., :3]
    row1_seed = value[..., 3:6]
    row0 = row0 / np.maximum(np.linalg.norm(row0, axis=-1, keepdims=True), 1e-12)
    row1 = row1_seed - np.sum(row0 * row1_seed, axis=-1, keepdims=True) * row0
    row1 = row1 / np.maximum(np.linalg.norm(row1, axis=-1, keepdims=True), 1e-12)
    row2 = np.cross(row0, row1, axis=-1)
    return np.stack((row0, row1, row2), axis=-2)


def _rotation_error_deg(pred_row6: np.ndarray, gt_row6: np.ndarray) -> np.ndarray:
    pred_rotation = _row6_to_matrix(pred_row6)
    gt_rotation = _row6_to_matrix(gt_row6)
    relative = pred_rotation @ np.swapaxes(gt_rotation, -1, -2)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _masked_values(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = np.asarray(value)[np.asarray(valid, dtype=bool)]
    if not selected.size:
        raise ValueError("metric selection contains no valid values")
    return selected.astype(np.float64, copy=False)


def _load_samples(path: Path, expected_per_split: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(REQUIRED_SAMPLE_KEYS.difference(payload.files))
        if missing:
            raise ValueError(f"sample payload is missing keys: {missing}")
        data = {key: payload[key] for key in payload.files}

    count = len(data["split"])
    if count != expected_per_split * 2:
        raise ValueError(
            f"expected {expected_per_split * 2} total samples, got {count}"
        )
    for split in ("train", "eval"):
        selected = np.asarray(data["split"]) == split
        if int(selected.sum()) != expected_per_split:
            raise ValueError(
                f"expected exactly {expected_per_split} {split} samples, "
                f"got {selected.sum()}"
            )
    expected_action_shape = (count, 32, 326)
    for key in (
        "gt_action_normalized",
        "gt_action_physical",
        "action_valid_mask",
    ):
        if data[key].shape != expected_action_shape:
            raise ValueError(
                f"{key} must have shape {expected_action_shape}, got {data[key].shape}"
            )
    if data["images_bgr"].shape != (count, 3, 480, 640, 3):
        raise ValueError(
            "images_bgr must have shape "
            f"{(count, 3, 480, 640, 3)}, got {data['images_bgr'].shape}"
        )
    normalizer_shape = expected_action_shape[1:]
    for key in ("action_scale", "action_offset"):
        if data[key].shape != normalizer_shape or not np.isfinite(data[key]).all():
            raise ValueError(
                f"{key} must be finite {normalizer_shape}, got {data[key].shape}"
            )
    return data


def _split_metrics(
    pred_physical: np.ndarray,
    gt_physical: np.ndarray,
    gt_normalized: np.ndarray,
    valid: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    inference_ms: np.ndarray,
    server_ms: np.ndarray,
    round_trip_ms: np.ndarray,
) -> dict[str, Any]:
    pred_normalized = pred_physical * scale[None, ...] + offset[None, ...]
    normalized_error = pred_normalized - gt_normalized
    physical_error = pred_physical - gt_physical

    weights = np.ones(pred_physical.shape[-1], dtype=np.float64)
    for mano_slice in MANO_SLICES:
        weights[mano_slice] = 0.5
    valid_counts = valid.sum(axis=(1, 2)).astype(np.float64)
    if np.any(valid_counts <= 0):
        raise ValueError("every sample must contain at least one valid action value")
    weighted_mse_per_sample = (
        np.square(normalized_error) * weights[None, None, :] * valid
    ).sum(axis=(1, 2)) / valid_counts
    mse_per_sample = (
        np.square(normalized_error) * valid
    ).sum(axis=(1, 2)) / valid_counts
    mae_per_sample = (
        np.abs(normalized_error) * valid
    ).sum(axis=(1, 2)) / valid_counts

    ee_mask = np.zeros_like(valid)
    mano_mask = np.zeros_like(valid)
    for section in EE_SLICES:
        ee_mask[..., section] = valid[..., section]
    for section in MANO_SLICES:
        mano_mask[..., section] = valid[..., section]
    ee_norm_error = _masked_values(normalized_error, ee_mask)
    mano_norm_error = _masked_values(normalized_error, mano_mask)
    mano_physical_error = _masked_values(physical_error, mano_mask)

    translation_l2_cm = []
    ee_rotation_error_deg = []
    for section in EE_SLICES:
        pose_valid = valid[..., section].all(axis=-1)
        pred_pose = pred_physical[..., section]
        gt_pose = gt_physical[..., section]
        translation = np.linalg.norm(pred_pose[..., :3] - gt_pose[..., :3], axis=-1)
        translation_l2_cm.append(translation[pose_valid] * 100.0)
        ee_rotation_error_deg.append(
            _rotation_error_deg(pred_pose[..., 3:9], gt_pose[..., 3:9])[pose_valid]
        )
    translation_l2_cm = np.concatenate(translation_l2_cm)
    ee_rotation_error_deg = np.concatenate(ee_rotation_error_deg)

    mano_rotation_error_deg = []
    for section in MANO_SLICES:
        component_valid = valid[..., section].reshape(*valid.shape[:2], 15, 6)
        joint_valid = component_valid.all(axis=-1)
        pred_mano = pred_physical[..., section].reshape(
            *pred_physical.shape[:2], 15, 6
        )
        gt_mano = gt_physical[..., section].reshape(
            *gt_physical.shape[:2], 15, 6
        )
        mano_rotation_error_deg.append(
            _rotation_error_deg(pred_mano, gt_mano)[joint_valid]
        )
    mano_rotation_error_deg = np.concatenate(mano_rotation_error_deg)

    normalized_values = _masked_values(normalized_error, valid)
    physical_values = _masked_values(physical_error, valid)
    physical_mse_per_sample = (
        np.square(physical_error) * valid
    ).sum(axis=(1, 2)) / valid_counts
    return {
        "sample_count": int(pred_physical.shape[0]),
        "valid_action_value_count": int(valid.sum()),
        "loss": {
            "normalized_weighted_mse_ee1_mano05_macro": float(
                weighted_mse_per_sample.mean()
            ),
            "normalized_unweighted_mse_macro": float(mse_per_sample.mean()),
            "normalized_unweighted_mse_micro": float(
                np.mean(np.square(normalized_values))
            ),
            "normalized_unweighted_rmse_micro": float(
                np.sqrt(np.mean(np.square(normalized_values)))
            ),
            "normalized_unweighted_mae_macro": float(mae_per_sample.mean()),
            "physical_mixed_unit_mse_micro": float(
                np.mean(np.square(physical_values))
            ),
            "physical_mixed_unit_mse_macro": float(
                physical_mse_per_sample.mean()
            ),
        },
        "groups": {
            "ee": {
                "normalized_mse": float(np.mean(np.square(ee_norm_error))),
                "normalized_mae": float(np.mean(np.abs(ee_norm_error))),
                "translation_l2_cm_mean": float(translation_l2_cm.mean()),
                "translation_l2_cm_p95": float(
                    np.quantile(translation_l2_cm, 0.95)
                ),
                "rotation_error_deg_mean": float(ee_rotation_error_deg.mean()),
                "rotation_error_deg_p95": float(
                    np.quantile(ee_rotation_error_deg, 0.95)
                ),
            },
            "mano90": {
                "normalized_mse": float(np.mean(np.square(mano_norm_error))),
                "normalized_mae": float(np.mean(np.abs(mano_norm_error))),
                "physical_row6_mae": float(np.mean(np.abs(mano_physical_error))),
                "physical_row6_rmse": float(
                    np.sqrt(np.mean(np.square(mano_physical_error)))
                ),
                "joint_rotation_error_deg_mean": float(
                    mano_rotation_error_deg.mean()
                ),
                "joint_rotation_error_deg_p95": float(
                    np.quantile(mano_rotation_error_deg, 0.95)
                ),
            },
        },
        "latency_ms": {
            "model_inference": _percentiles(inference_ms),
            "server_processing": _percentiles(server_ms),
            "client_round_trip": _percentiles(round_trip_ms),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples-per-split", type=int, default=100)
    args = parser.parse_args()
    if args.expected_samples_per_split <= 0:
        raise ValueError("expected-samples-per-split must be positive")

    samples_path = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_samples(samples_path, args.expected_samples_per_split)
    count = len(data["split"])
    dataset_name = str(data["dataset_name"].item())
    predictions = []
    decoded_actions = []
    inference_ms = []
    server_ms = []
    round_trip_ms = []
    server_info = None

    # Exercise the exact robot-side MANO -> q20 path before sending requests.
    prepare_hand_action_decoder("mano")
    with PersistentPolicyClient(args.host, args.port, args.timeout) as client:
        print(f"[client] connected to {args.host}:{args.port}", flush=True)
        for index in range(count):
            # build_request owns the live q20/EE -> MANO conversion. Restore the
            # exact decoded dataset RGB frames after its 16:9 robot-image check.
            robot_images = data["images_bgr"][index, :, 60:420]
            observation = {
                "image": robot_images[0],
                "image_wrist_left": robot_images[1],
                "image_wrist_right": robot_images[2],
                "left_ee_pose": data["left_ee_pose"][index],
                "right_ee_pose": data["right_ee_pose"][index],
                "left_hand_joint": data["left_hand_joint"][index],
                "right_hand_joint": data["right_hand_joint"][index],
            }
            request, context = build_request(
                observation,
                str(data["instruction"][index]),
                client_id="mano198-split-eval",
            )
            for camera_index, key in enumerate(WIRE_IMAGE_KEYS):
                request["observation"][key] = np.ascontiguousarray(
                    data["images_bgr"][index, camera_index, ..., ::-1]
                )
            request["dataset_name"] = dataset_name
            response = client.send(request)
            if not response.get("success"):
                raise RuntimeError(
                    f"request {index} failed: {response.get('error', response)}"
                )
            info = dict(response.get("server_info") or {})
            if info.get("adapter_name") != "teleop_mano":
                raise RuntimeError(
                    f"request {index} reached the wrong adapter: "
                    f"{info.get('adapter_name')}"
                )
            prediction = np.asarray(response["action"], dtype=np.float32)
            if prediction.shape != data["gt_action_physical"][index].shape:
                raise ValueError(
                    f"request {index} action shape {prediction.shape} does not match GT "
                    f"{data['gt_action_physical'][index].shape}"
                )
            decoded = decode_action(
                prediction,
                context,
                action_type="relative",
                hand_action_source="mano",
                adapter_name="teleop_mano",
            )
            if decoded.shape != (prediction.shape[0], 52) or not np.isfinite(
                decoded
            ).all():
                raise ValueError(
                    f"request {index} produced invalid client action {decoded.shape}"
                )
            predictions.append(prediction)
            decoded_actions.append(decoded)
            timing = dict(response.get("timing") or {})
            inference_ms.append(float(timing["inference_ms"]))
            server_ms.append(float(timing["server_processing_ms"]))
            round_trip_ms.append(float(timing["round_trip_ms"]))
            server_info = info
            if (index + 1) % 10 == 0 or index + 1 == count:
                print(f"[client] completed {index + 1}/{count}", flush=True)

    pred_physical = np.stack(predictions)
    decoded = np.stack(decoded_actions)
    inference_ms_array = np.asarray(inference_ms, dtype=np.float64)
    server_ms_array = np.asarray(server_ms, dtype=np.float64)
    round_trip_ms_array = np.asarray(round_trip_ms, dtype=np.float64)
    split_metrics = {}
    for split in ("train", "eval"):
        selected = data["split"] == split
        if int(selected.sum()) != args.expected_samples_per_split:
            raise ValueError(
                f"expected exactly {args.expected_samples_per_split} {split} samples, "
                f"got {selected.sum()}"
            )
        split_metrics[split] = _split_metrics(
            pred_physical[selected],
            data["gt_action_physical"][selected],
            data["gt_action_normalized"][selected],
            data["action_valid_mask"][selected],
            data["action_scale"],
            data["action_offset"],
            inference_ms_array[selected],
            server_ms_array[selected],
            round_trip_ms_array[selected],
        )

    metrics = {
        "samples": str(samples_path),
        "dataset_name": dataset_name,
        "protocol": "liber_teleop_v22",
        "client": "Liberai_Deploy utils.liber_teleop_v22_client",
        "connection": "one persistent length-prefixed msgpack-numpy TCP connection",
        "wire_images": "exact decoded dataset RGB frames at 480x640",
        "action_decode": "relative EE + relative MANO90 restored to absolute and decoded to q20",
        "loss_definition": (
            "Final denoised server action versus canonical GT. The headline loss is "
            "normalized squared error weighted EE=1.0/MANO90=0.5, summed over "
            "valid values, divided by the raw valid-value count per sample, then "
            f"averaged over {args.expected_samples_per_split} samples. It is not the "
            "diffusion flow-matching "
            "training objective."
        ),
        "server_info": server_info,
        "splits": split_metrics,
        "all_predictions_finite": bool(np.isfinite(pred_physical).all()),
        "all_client_decodes_finite": bool(np.isfinite(decoded).all()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        prediction_physical=pred_physical,
        client_action_52d=decoded,
        split=data["split"],
        episode_index=data["episode_index"],
        frame_index=data["frame_index"],
        target_frame_index=data["target_frame_index"],
        inference_ms=inference_ms_array,
        server_processing_ms=server_ms_array,
        round_trip_ms=round_trip_ms_array,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
