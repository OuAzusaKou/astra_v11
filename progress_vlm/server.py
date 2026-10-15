"""Independent progress-only GPU server. Never returns or publishes robot actions."""
import argparse
import io
import json
from pathlib import Path
import socketserver
import sys
import threading
import time

from .protocol import PROTOCOL, parse_prediction, receive, send, validate


class Predictor:
    def __init__(self, root):
        sys.path[:0] = [str(root / "code"), str(root / "scripts")]
        import torch
        from progress_runtime import ProgressPreprocessor, Qwen3VLForConditionalGeneration, generate_prediction
        self.torch, self.generate = torch, generate_prediction
        info = json.loads((root / "model/EXPORT.json").read_text())
        if info["source_step"] != 40000 or not info["strict_load"]:
            raise ValueError("expected verified step_40000 model export")
        self.identity = {"step": 40000, "path": str(root / "model"),
                         "checkpoint_sha256": json.loads((root / "checkpoint/verification.json").read_text())["sha256"]}
        torch.set_num_threads(4)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            root / "model", torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="flash_attention_2", local_files_only=True).eval()
        self.model.config.use_cache = False
        self.model.generation_config.use_cache = False
        self.prep = ProgressPreprocessor(root / "model")

    def predict(self, encoded, l3, l2):
        from PIL import Image
        images = []
        for data in encoded:
            image = Image.open(io.BytesIO(data))
            if max(image.size) > 4096 or min(image.size) < 16:
                raise ValueError("image dimensions outside supported range")
            images.append(image.convert("RGB"))
        started = time.monotonic()
        text = self.generate(self.model, self.prep, self.prep.vlm_processor, images,
                             f"L3: {l3}\nL2: {l2}", "The subtask is 10% complete.",
                             self.torch.device("cuda:0"))
        self.torch.cuda.synchronize()
        return {**parse_prediction(text), "inference_seconds": time.monotonic() - started}


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address, predictor, log):
        self.predictor, self.log = predictor, log
        self.gpu_lock = threading.Lock()
        self.log_lock = threading.Lock()
        # Bound both socket handlers and GPU admission; no stale inference queue.
        self.connections = threading.BoundedSemaphore(8)
        super().__init__(address, Handler)

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connections.release()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(15)
        while True:
            try:
                message = receive(self.request)
            except (ConnectionError, TimeoutError, OSError, ValueError):
                return
            response = {"protocol": PROTOCOL, "success": False,
                        **{k: message.get(k) for k in ("request_id", "session_id", "subtask_id")},
                        "model": self.server.predictor.identity}
            try:
                validate(message)
                if message["cmd"] == "health":
                    response.update(success=True, ready=True)
                elif not self.server.gpu_lock.acquire(blocking=False):
                    response["error"] = "server_busy; retry with fresh images"
                else:
                    try:
                        result = self.server.predictor.predict(message["images"], message["l3"], message["l2"])
                        response.update(success=True, **result)
                    finally:
                        self.server.gpu_lock.release()
            except Exception as error:
                response["error"] = type(error).__name__ + ": " + str(error)[:1000]
            try:
                send(self.request, response)
            except OSError:
                return
            with self.server.log_lock:
                self.server.log.write(json.dumps({"time": time.time(), **response}) + "\n")
                self.server.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="10.128.40.203")
    parser.add_argument("--port", type=int, default=29573)
    parser.add_argument("--root", type=Path, default=Path("/data/trans/progress_40k_20260930"))
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    predictor = Predictor(args.root)
    sample = json.loads((args.root / "samples/samples.json").read_text())[0]
    l3, l2 = sample["task"].removeprefix("L3: ").split("\nL2: ", 1)
    warm = predictor.predict([(args.root / "samples" / name).read_bytes() for name in sample["images"]], l3, l2)
    if not warm["valid"]:
        raise RuntimeError("warmup returned invalid progress text")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as log, Server((args.host, args.port), predictor, log) as server:
        print(json.dumps({"event": "READY", "address": [args.host, args.port], "model": predictor.identity}), flush=True)
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
