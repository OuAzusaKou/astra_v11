"""Same TCP framing and MessagePack encoding as the WAM connection."""
import re
import socket
import time
import uuid

from policy_server.msgpack_numpy import packb, unpackb

PROTOCOL = "qwen_progress_v1"
CHECKPOINT_SHA256 = "1cb7a4c5688286786a68380161c44822e54f5225819793ca382aad34abfad85d"
MAX_MESSAGE = 16 * 1024 * 1024
ROLES = ("history_main", "subtask_start_main", "current_main", "current_left", "current_right")


def receive(sock):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            block = sock.recv(size - len(data))
            if not block:
                raise ConnectionError("peer disconnected")
            data.extend(block)
        return bytes(data)
    size = int.from_bytes(exact(4), "big")
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("invalid packet size")
    message = unpackb(exact(size))
    if not isinstance(message, dict):
        raise ValueError("expected a dictionary")
    return message


def send(sock, message):
    data = packb(message)
    if not 0 < len(data) <= MAX_MESSAGE:
        raise ValueError("packet too large")
    sock.sendall(len(data).to_bytes(4, "big") + data)


def request(images, l3, l2, session_id, subtask_id, **metadata):
    return {"protocol": PROTOCOL, "cmd": "infer_progress", "request_id": uuid.uuid4().hex,
            "session_id": session_id, "subtask_id": subtask_id, "l3": l3, "l2": l2,
            "image_roles": list(ROLES), "images": images, **metadata}


def validate(message):
    if message.get("protocol") != PROTOCOL or message.get("cmd") not in ("health", "infer_progress"):
        raise ValueError("unsupported protocol or command")
    for key in ("request_id", "session_id", "subtask_id"):
        if not isinstance(message.get(key), str) or not 0 < len(message[key]) <= 128:
            raise ValueError("invalid " + key)
    if message["cmd"] == "health":
        return
    for key in ("l3", "l2"):
        if not isinstance(message.get(key), str) or not message[key].strip() or len(message[key]) > 2000:
            raise ValueError("missing/invalid " + key)
    if message.get("image_roles") != list(ROLES):
        raise ValueError("five-image order mismatch")
    images = message.get("images")
    if not isinstance(images, list) or len(images) != 5:
        raise ValueError("exactly five encoded RGB images required")
    if any(not isinstance(x, bytes) or not 0 < len(x) <= 3 * 1024 * 1024 for x in images):
        raise ValueError("invalid image bytes")


def parse_prediction(text):
    match = re.fullmatch(r"The subtask is (10|20|30|40|50|60|70|80|90|100)% complete\."
                         r"(?:\s+(This is an anomalous replay\.))?", text.strip())
    return {"valid": bool(match), "progress": int(match[1]) if match else None,
            "anomalous_replay": bool(match[2]) if match else None, "text": text}


class Client:
    def __init__(self, host="10.128.40.203", port=29573, timeout=10):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock = None

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def infer(self, message):
        started = time.monotonic()
        try:
            if self.sock is None:
                self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.sock.settimeout(self.timeout)
            send(self.sock, message)
            reply = receive(self.sock)
            if reply.get("protocol") != PROTOCOL:
                raise ValueError("server protocol mismatch")
            for key in ("request_id", "session_id", "subtask_id"):
                if reply.get(key) != message.get(key):
                    raise ValueError("response identity mismatch: " + key)
            if reply.get("success") is not True:
                raise RuntimeError(reply.get("error", "server error"))
            if reply.get("model", {}).get("step") != 40000:
                raise ValueError("unexpected checkpoint step")
            if reply.get("model", {}).get("checkpoint_sha256") != CHECKPOINT_SHA256:
                raise ValueError("unexpected checkpoint identity")
            reply["round_trip_seconds"] = time.monotonic() - started
            return reply
        except BaseException:
            self.close()
            raise
