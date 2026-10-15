"""ROS camera subscriber + progress display. No robot publishers or control clients."""
import argparse
import json
import signal
from pathlib import Path
import threading
import time
import uuid

from .frames import Frame, Frames, Trend
from .protocol import Client, ROLES, request

TOPICS = {"main": "/cam/chest/image_raw/compressed",
          "left": "/cam/hand_left/image_raw/compressed",
          "right": "/cam/hand_right/image_raw/compressed"}


class Observer:
    def __init__(self, args, node):
        self.args, self.node = args, node
        self.frames = Frames(max_age=args.max_camera_age)
        self.stop, self.lock = threading.Event(), threading.Lock()
        self.client = Client(args.host, args.port)
        self.follower = None
        if args.follow_session:
            from .follow import SessionFollower
            self.follower = SessionFollower(args.follow_session)
        self.follow_key, self.follow_active, self.follow_context = None, False, None
        self.follow_poll_lock = threading.Lock()
        self.session = uuid.uuid4().hex
        self.tasks = [args.l2, *args.next_l2]
        self.index, self.epoch, self.start_frame = 0, 0, None
        self.latest, self.display_images = None, None
        self.status = "Waiting for runner session" if self.follower else "Press Enter to capture the L2 start image"
        self.trend = Trend(seconds=args.plateau_seconds, max_gap=max(3., 2.5 / args.hz))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps({**vars(args), "output": str(args.output),
            "follow_session": str(args.follow_session) if args.follow_session else None,
            "session_id": self.session, "topics": TOPICS, "mode": "observe_only"}, indent=2))
        self.log = (args.output / "predictions.jsonl").open("a", buffering=1)

    def camera(self, name, msg):
        latency = self.args.main_latency if name == "main" else self.args.wrist_latency
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9 - latency
        self.frames.add(name, Frame(stamp, time.monotonic(), bytes(msg.data)))

    def start_subtask(self, advance=False):
        if self.follower:
            return  # Task changes belong exclusively to the runner's P/Enter/B flow.
        frame = self.frames.latest()
        now = time.monotonic()
        ros_now = self.node.get_clock().now().nanoseconds / 1e9
        if frame is None or now - frame.received > self.args.max_camera_age or not -.1 <= ros_now-frame.stamp <= self.args.max_camera_age:
            with self.lock:
                self.status = "No fresh main image; start rejected"
            return
        with self.lock:
            if advance:
                if self.index + 1 >= len(self.tasks):
                    self.status = "No next L2 configured"
                    return
                self.index += 1
            self.epoch += 1
            self.start_frame = frame
            self.latest, self.display_images = None, None
            self.trend.reset()
            self.status = "Observing " + self.tasks[self.index]
            self.log.write(json.dumps({"event": "subtask_start", "time": time.time(), "epoch": self.epoch,
                                       "l2": self.tasks[self.index], "stamp": frame.stamp}) + "\n")

    def sync_follow(self):
        with self.follow_poll_lock:
            self._sync_follow()

    def _sync_follow(self):
        if self.follower is None:
            return
        try:
            context = self.follower.poll()
        except (OSError, ValueError, KeyError, TypeError) as error:
            context = {"key": self.follow_key, "active": False,
                       "reason": "Waiting for runner: " + str(error)[:100]}
        with self.lock:
            changed = context["key"] != self.follow_key
            activation = context["active"] != self.follow_active
            if changed or activation:
                self.epoch += 1  # Invalidate in-flight responses on P, B, pause or exit.
                self.latest, self.display_images = None, None
                self.trend.reset()
                self.log.write(json.dumps({"event": "runner_context", "time": time.time(),
                    **{k: v for k, v in context.items() if k != "anchor"}}) + "\n")
            self.follow_key, self.follow_active = context["key"], context["active"]
            self.follow_context = {k: v for k, v in context.items() if k != "anchor"}
            self.start_frame = context.get("anchor") if context["active"] else None
            if "l3" in context:
                self.args.l3 = context["l3"]
                self.tasks, self.index = [context["l2"]], 0
            if not context["active"] or changed or activation:
                self.status = context["reason"]

    def follow_loop(self):
        while not self.stop.is_set():
            self.sync_follow()
            self.stop.wait(.05)

    def worker(self):
        import cv2
        import numpy as np
        last_key = None
        while not self.stop.is_set():
            began = time.monotonic()
            with self.lock:
                epoch, start, l2, l3 = self.epoch, self.start_frame, self.tasks[self.index], self.args.l3
                runner_context = self.follow_context
            if start is None:
                self.stop.wait(.1)
                continue
            try:
                selected = self.frames.assemble(start, began, self.node.get_clock().now().nanoseconds / 1e9)
                current = selected[2]
                key = (epoch, current.stamp)
                if key == last_key:
                    self.stop.wait(.05)
                    continue
                last_key = key
                encoded, decoded = [], []
                for i, frame in enumerate(selected):
                    image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("camera JPEG decoding failed")
                    # Same wrist orientation as LeoWujiInferEnv; head is unchanged.
                    if i >= 3:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise ValueError("camera JPEG encoding failed")
                    decoded.append(image)
                    encoded.append(jpg.tobytes())
                msg = request(encoded, l3, l2, self.session, str(epoch),
                              image_timestamps=[f.stamp for f in selected])
                reply = self.client.infer(msg)
                age = time.monotonic() - current.received + self.args.main_latency
                # Re-read status at acceptance, even if the background poll has not run yet.
                if self.follower:
                    self.sync_follow()
                with self.lock:
                    if epoch != self.epoch:
                        self.log.write(json.dumps({"event": "discarded_old_subtask", "request_id": msg["request_id"]}) + "\n")
                        continue
                    if age > self.args.max_result_age or not reply.get("valid"):
                        self.trend.reset()
                        self.latest = None
                        self.status = "Result discarded: stale or invalid output"
                        self.log.write(json.dumps({"event": "discarded_result", "age": age, **reply}) + "\n")
                        continue
                    trend = self.trend.add(current.stamp, reply["progress"], reply["anomalous_replay"])
                    image_dir = self.args.output / "images" / msg["request_id"]
                    image_dir.mkdir(parents=True)
                    for role, data in zip(ROLES, encoded):
                        (image_dir / (role + ".jpg")).write_bytes(data)
                    result = {"event": "prediction", "time": time.time(), "l3": l3, "l2": l2,
                              "runner_context": runner_context,
                              "capture_stamp": current.stamp, "image_timestamps": msg["image_timestamps"],
                              "image_dir": str(image_dir), "age_on_arrival": age, **reply, **trend,
                              "diagnostic_only": not (runner_context and runner_context.get("version") in (4, 5, 6, 7))}
                    self.log.write(json.dumps(result) + "\n")
                    self.latest = (result, current.received)
                    self.display_images = decoded
                    self.status = "Observing; runner uses VLM for next L2" if runner_context and runner_context.get("version") in (4, 5, 6, 7) else "Observing; diagnostic only"
                print(f"[VLM] L2={l2} progress={reply['progress']}% replay={reply['anomalous_replay']} "
                      f"plateau={trend['plateau_candidate']} RTT={reply['round_trip_seconds']:.3f}s", flush=True)
            except Exception as error:
                with self.lock:
                    if epoch == self.epoch:
                        self.trend.reset()
                        self.latest = None
                        status = str(error)[:160]
                        if self.status != status:
                            self.log.write(json.dumps({"event": "observation_error", "time": time.time(),
                                                       "epoch": epoch, "error": status}) + "\n")
                            print("[VLM WAIT] " + status, flush=True)
                        self.status = status
                self.stop.wait(.25)
            self.stop.wait(max(0., 1. / self.args.hz - (time.monotonic() - began)))

    def window(self):
        import cv2
        import numpy as np
        from .recording import WindowRecorder
        recorder = None
        recording_error = None
        try:
            recorder = WindowRecorder(self.args.output / "window_video")
            print(f"[VLM REC] 10 fps, 60-second MP4 segments: {recorder.directory}", flush=True)
        except Exception as error:
            recording_error = str(error)
            print(f"[VLM REC ERROR] {error}", flush=True)
        try:
            self._window_loop(cv2, np, recorder, recording_error)
        finally:
            self.stop.set()
            if recorder:
                recorder.close()
            cv2.destroyAllWindows()

    def _window_loop(self, cv2, np, recorder, recording_error):
        while not self.stop.is_set():
            live = self.frames.latest()
            with self.lock:
                status, latest, images = self.status, self.latest, self.display_images
                l2 = self.tasks[self.index]
                context = self.follow_context or {}
            canvas = np.zeros((660, 1200, 3), dtype=np.uint8)
            if live:
                image = cv2.imdecode(np.frombuffer(live.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    canvas[40:400, :480] = cv2.resize(image, (480, 360))
            title = f"V{context.get('version')} / VLM AUTO NEXT L2 / RUNNER CONTROLS ROBOT" if context.get("version") in (4, 5, 6, 7) else "VLM OBSERVER / NO ROBOT CONTROL"
            lines = [title, "L3: " + self.args.l3, "L2: " + l2, status,
                     "P/Enter/B: use ASTRA main terminal" if self.follower else "Enter/R: new L2 start image",
                     "Task follows runner automatically | Q: exit viewer" if self.follower else "P: start next configured L2 | Q: exit"]
            if latest:
                result, received = latest
                age = time.monotonic() - received + self.args.main_latency
                if age <= self.args.max_result_age:
                    lines += [f"Progress: {result['progress']}% | replay: {result['anomalous_replay']}",
                              f"Plateau candidate: {result['plateau_candidate']} ({result['window_seconds']:.1f}s window)",
                              f"RTT: {result['round_trip_seconds']:.3f}s | image age: {age:.2f}s"]
                else:
                    lines.append(f"STALE RESULT ({age:.1f}s); waiting for fresh prediction")
            if context.get("vlm_auto_l2"):
                gate = context["vlm_auto_l2"]
                lines.append(f"Auto L2 {gate['consecutive']}/{gate['required_count']} >= {gate['threshold']}%: {gate['reason']}")
            if context.get("vlm_auto_takeover"):
                lines.append("Auto Astra: " + context["vlm_auto_takeover"]["reason"])
            for i, line in enumerate(lines):
                cv2.putText(canvas, line[:86], (490, 45 + 32*i), cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 240, 220), 1)
            cv2.putText(canvas, "LIVE HEAD", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
            if images:
                for i, image in enumerate(images):
                    canvas[460:640, i*240:(i+1)*240] = cv2.resize(image, (240, 180))
                    cv2.putText(canvas, ROLES[i], (i*240+4, 450), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 220, 220), 1)
            cv2.putText(canvas, "Bottom row: exact five views of last accepted prediction", (10, 425), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1)
            label = "REC 10 fps | " + time.strftime("%Y-%m-%d %H:%M:%S")
            if recording_error:
                label = "RECORDING FAILED - see VLM-progress log"
            cv2.putText(canvas, label, (490, 385), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (60, 60, 255) if recording_error else (80, 220, 80), 1)
            if recorder and not recording_error:
                try:
                    recorder.write(canvas)
                except Exception as error:
                    recording_error = str(error)
                    recorder.close()
                    print(f"[VLM REC ERROR] {error}", flush=True)
                    self.log.write(json.dumps({"event": "recording_error", "time": time.time(),
                                               "error": recording_error}) + "\n")
            cv2.imshow("Progress VLM observer", canvas)
            key = cv2.waitKey(30) & 0xff
            if key in (ord("q"), 27) or cv2.getWindowProperty("Progress VLM observer", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13, ord("r")):
                self.start_subtask()
            if key == ord("p"):
                self.start_subtask(advance=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="10.128.40.203")
    p.add_argument("--port", type=int, default=29573)
    p.add_argument("--follow-session", type=Path, help="read L3/L2 and the first observation from this Astra runner")
    p.add_argument("--l3", default="")
    p.add_argument("--l2", default="")
    p.add_argument("--next-l2", action="append", default=[])
    p.add_argument("--output", type=Path, required=True, help="new directory, never reused")
    p.add_argument("--hz", type=float, default=1.)
    p.add_argument("--main-latency", type=float, default=.100)
    p.add_argument("--wrist-latency", type=float, default=.040)
    p.add_argument("--max-camera-age", type=float, default=1.)
    p.add_argument("--max-result-age", type=float, default=2.)
    p.add_argument("--plateau-seconds", type=float, default=8.)
    p.add_argument("--probe-seconds", type=float, default=0., help="headless receive-only topic check; no inference")
    args = p.parse_args()
    if args.follow_session and (args.l3 or args.l2 or args.next_l2):
        p.error("--follow-session owns task selection; do not also supply L3/L2")
    if not args.follow_session and any(not text.strip() or len(text) > 2000 for text in [args.l3, args.l2, *args.next_l2]):
        p.error("task descriptions must contain 1–2000 characters")
    if not 0 < args.hz <= 3 or args.max_result_age <= 0 or args.max_camera_age <= 0 or args.plateau_seconds <= 0:
        p.error("invalid rate or age/window threshold")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    rclpy.init()
    node = Node("progress_vlm_observer")
    observer = Observer(args, node)
    subscriptions = [node.create_subscription(CompressedImage, topic,
                     lambda msg, name=name: observer.camera(name, msg), qos_profile_sensor_data)
                     for name, topic in TOPICS.items()]
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    worker = follower = None
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {sig: signal.signal(sig, lambda *_: observer.stop.set()) for sig in signals}
    try:
        spinner.start()
        if args.probe_seconds:
            observer.stop.wait(args.probe_seconds)
            with observer.frames.lock:
                counts = {k: len(v) for k, v in observer.frames.cameras.items()}
            (args.output / "camera_probe.json").write_text(json.dumps(counts, indent=2))
            print(json.dumps({"camera_frames": counts, "inference_started": False, "robot_control": False}), flush=True)
        else:
            if observer.follower:
                follower = threading.Thread(target=observer.follow_loop, daemon=True)
                follower.start()
            worker = threading.Thread(target=observer.worker, daemon=True)
            worker.start()
            observer.window()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        observer.stop.set()
        if follower:
            follower.join(timeout=5)
        if worker:
            worker.join(timeout=12)
        observer.client.close()
        observer.log.close()
        rclpy.shutdown()
        spinner.join(timeout=2)
        node.destroy_node()


if __name__ == "__main__":
    main()
