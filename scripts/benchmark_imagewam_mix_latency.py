"""Measure the mix client with live observations; never reset or publish actions."""

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

from scripts.evaluate_imagewam_mano_server import PersistentPolicyClient
from utils.client_config import load_client_config
from utils.mix_policy_client import MixPolicyClient
from policy_server.msgpack_numpy import packb, unpackb


class ProfilePolicyClient(PersistentPolicyClient):
    def send(self, request):
        started = time.perf_counter()
        payload = packb(request)
        packed = time.perf_counter()
        self.sock.sendall(len(payload).to_bytes(4, "big"))
        self.sock.sendall(payload)
        sent = time.perf_counter()
        response_size = int.from_bytes(self._recv_exact(4), "big")
        header_received = time.perf_counter()
        response_payload = self._recv_exact(response_size)
        received = time.perf_counter()
        response = unpackb(response_payload)
        unpacked = time.perf_counter()
        self.last_timing = {
            "pack_ms": (packed - started) * 1000,
            "send_ms": (sent - packed) * 1000,
            "wait_response_header_ms": (header_received - sent) * 1000,
            "receive_body_ms": (received - header_received) * 1000,
            "unpack_ms": (unpacked - received) * 1000,
            "request_bytes": len(payload), "response_bytes": response_size,
        }
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-npz", help="Replay one saved observation instead of subscribing to ROS.")
    parser.add_argument("--profile", action="store_true", help="Save client wire and server stage timings.")
    parser.add_argument("--compare-image-preprocess", action="store_true", help="Alternate server/client image processing over one connection (server,client,client,server).")
    parser.add_argument("--compare-camera-implementation", action="store_true", help="Compare original and fused leo1 preprocessing over one connection; use a client-preprocessing config.")
    args = parser.parse_args()
    config = load_client_config(args.config)
    runtime = config["runtime"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.compare_camera_implementation:
        from utils import imagewam_camera
        optimized_camera = imagewam_camera.prepare_model_input
        def original_camera(frames, crop):
            rgb = [np.ascontiguousarray(frame[..., ::-1]) for frame in frames]
            return imagewam_camera.model_input(imagewam_camera.preprocess_cameras(rgb, crop))
    node = executor = None
    if args.observation_npz:
        with np.load(args.observation_npz) as data:
            observation = dict(data)
        get_observation = lambda: observation
    else:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from envs.leo_wuji_env_infer import LeoWujiInferEnv

        rclpy.init()
        node = Node("imagewam_latency_observer")
        env = LeoWujiInferEnv(
            node, fps=runtime["fps"], camera_latency=runtime["camera_latency"],
            wrist_camera_latency=runtime["wrist_camera_latency"],
        )
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()
        print("Waiting for camera and robot state (observation only)...", flush=True)
        while not env.is_ready():
            time.sleep(.1)
        get_observation = lambda: env.get_obs(runtime["history_offsets"])
    try:
        client_type = ProfilePolicyClient if args.profile else PersistentPolicyClient
        with client_type(config["policy"]["host"], config["policy"]["port"], 180.) as wire:
            client = MixPolicyClient(wire, config["policy"], hand_source=config["action"]["hand_source"],
                                     crop=runtime["crop_percent"])
            for index in range(args.warmup + args.requests):
                if args.compare_image_preprocess:
                    client.config["image_preprocess"] = ("server", "client", "client", "server")[index % 4]
                if args.compare_camera_implementation:
                    camera_mode = ("original", "optimized", "optimized", "original")[index % 4]
                    imagewam_camera.prepare_model_input = {"original": original_camera, "optimized": optimized_camera}[camera_mode]
                started = time.perf_counter()
                observation = get_observation()
                observed = time.perf_counter()
                if index == 0:
                    np.savez_compressed(output.with_suffix(".observation.npz"), **observation)
                prepared = time.perf_counter()
                request, context = client.build_request(observation, config["instruction"])
                sent = time.perf_counter()
                response = wire.send(request)
                received = time.perf_counter()
                action = client.decode_response(response, context)
                decoded = time.perf_counter()
                row = {
                    "index": index, "warmup": index < args.warmup,
                    "image_preprocess": client.config.get("image_preprocess", "server"),
                    "observation_wait_ms": (observed - started) * 1000,
                    "request_build_ms": (sent - prepared) * 1000,
                    "round_trip_ms": (received - sent) * 1000,
                    "client_decode_ms": (decoded - received) * 1000,
                    "client_pipeline_ms": (decoded - prepared) * 1000,
                    "action_count": len(action),
                }
                if args.profile:
                    row.update(request_id=request["request_id"], client_wire=wire.last_timing,
                               server_profile=response.get("profile"), server_timing=response.get("timing"))
                if args.compare_camera_implementation:
                    row["camera_implementation"] = camera_mode
                if node is not None:
                    stamp = node.get_clock().now().to_msg()
                    age = stamp.sec + stamp.nanosec * 1e-9 - observation["timestamp"]
                    offsets = (np.arange(min(len(action), runtime["chunk_size"])) + client.action_row_offset) / runtime["fps"]
                    row.update(observation_age_ms=float(age * 1000), future_actions=int((offsets > age).sum()))
                rows.append(row)
                print(json.dumps(row), flush=True)
                # Preserve completed requests even if a later request is interrupted.
                output.write_text(json.dumps({"samples": rows, "server_info": response["server_info"]}, indent=2))
        measured = [row for row in rows if not row["warmup"]]
        summary = {}
        for key in measured[0]:
            if key.endswith("_ms"):
                values = [row[key] for row in measured]
                summary[key] = {"mean": float(np.mean(values)), "p50": float(np.median(values)),
                                "p95": float(np.percentile(values, 95)), "min": float(np.min(values)), "max": float(np.max(values))}
        report = {"host": config["policy"]["host"], "port": config["policy"]["port"],
                  "observation_source": args.observation_npz or "live_ros",
                  "warmup_requests": args.warmup, "measured_requests": args.requests,
                  "summary_ms": summary, "samples": rows, "server_info": response["server_info"]}
        for enabled, key, modes in ((args.compare_image_preprocess, "image_preprocess", ("server", "client")),
                                    (args.compare_camera_implementation, "camera_implementation", ("original", "optimized"))):
            if not enabled:
                continue
            grouped = report[f"summary_by_{key}"] = {}
            for mode in modes:
                selected = [row for row in measured if row[key] == mode]
                grouped[mode] = {
                    key: {"mean": float(np.mean([row[key] for row in selected])),
                          "p50": float(np.median([row[key] for row in selected])),
                          "p95": float(np.percentile([row[key] for row in selected], 95))}
                    for key in summary
                }
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
            env.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
