"""Mode-2 adapter for the actual remote server: infer_action_chunk only."""
import select
import socket
import time

import numpy as np

from .control import actions52

MODE2_CHECKPOINT = "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114"


class PolicyTransportError(ConnectionError):
    """Inference transport failed; no robot publication is performed here."""


class Connection:
    def __init__(self, host, port, stop, timeout=60.):
        self.host, self.port, self.stop, self.timeout = host, port, stop, timeout
        self.sock = None
        self.last_response_at = None

    def close(self):
        self.last_response_at = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def send(self, request):
        from policy_server.msgpack_numpy import packb, unpackb
        try:
            if self.stop.is_set():
                raise InterruptedError("runner stopped")
            # Astra may own the robot for minutes. Do not reuse an idle TCP
            # connection which the inference server may already have closed.
            if self.last_response_at is not None and time.monotonic() - self.last_response_at >= 30.:
                self.close()
            if self.sock is None:
                self.sock = socket.create_connection((self.host, self.port), timeout=5.)
            self.sock.settimeout(self.timeout)
            payload = packb(request)
            self.sock.sendall(len(payload).to_bytes(4, "big") + payload)
            deadline = time.monotonic() + self.timeout
            size = int.from_bytes(self._read(4, deadline), "big")
            if not 0 < size <= 128 * 1024 * 1024:
                raise ValueError("invalid model response length")
            response = unpackb(self._read(size, deadline))
            if not isinstance(response, dict) or response.get("request_id") != request["request_id"]:
                raise ValueError("model response request_id mismatch")
            self.last_response_at = time.monotonic()
            return response
        except BaseException as error:
            self.close()  # Never reuse a socket containing a late/partial response.
            if isinstance(error, OSError) and not isinstance(error, InterruptedError):
                raise PolicyTransportError(str(error)) from error
            raise

    def _read(self, size, deadline):
        parts = []
        while size:
            if self.stop.is_set():
                raise InterruptedError("runner stopped")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model response timeout")
            if not select.select([self.sock], [], [], min(.1, remaining))[0]:
                continue
            block = self.sock.recv(size)
            if not block:
                raise ConnectionError("model server disconnected")
            parts.append(block)
            size -= len(block)
        return b"".join(parts)


class Mode2Policy:
    def __init__(self, connection, client_id, *, chunk_size=32, dataset_name=None, codec=None, expected_checkpoint=MODE2_CHECKPOINT):
        if codec is None:
            from utils import liber_teleop_v22_client as codec
        self.codec, self.connection, self.client_id = codec, connection, client_id
        if not isinstance(expected_checkpoint, str) or not expected_checkpoint.strip():
            raise ValueError("expected_checkpoint must be explicit")
        self.expected_checkpoint = expected_checkpoint
        self.chunk_size, self.dataset_name = chunk_size, dataset_name

    def infer(self, observation, instruction):
        request, context = self.codec.build_request(
            observation, instruction, client_id=self.client_id, dataset_name=self.dataset_name)
        response = self.connection.send(request)
        if response.get("request_id") != request["request_id"] or response.get("success") is not True:
            raise RuntimeError(f"WAM inference failed: {response.get('error', 'invalid response')}")
        info = response.get("server_info") or {}
        if (self.expected_checkpoint not in str(info.get("checkpoint", ""))
                or info.get("adapter_name") != "teleop_mano"
                or info.get("protocol") != "liber_teleop_v22"
                or info.get("action_dim") != 326):
            raise RuntimeError("server profile is not the expected mode-2 teleop_mano checkpoint")
        if response.get("action_type") != "lerobot_canonical":
            raise ValueError("expected canonical 326D action response")
        raw = np.asarray(response.get("action"))
        if (raw.ndim != 2 or raw.shape[1] != 326 or not len(raw)
                or not np.isfinite(raw).all() or info.get("action_horizon") != len(raw)):
            raise ValueError("invalid canonical action shape, values or horizon")
        decoded = self.codec.decode_action(raw, context, action_type="relative",
                                           hand_action_source="mano", adapter_name="teleop_mano")
        result = actions52(decoded)
        if len(result) != len(raw):
            raise ValueError("decoder changed the action horizon")
        return result[:self.chunk_size], {
            "request_id": request["request_id"], "instruction_sent": instruction,
            "server_info": info, "timing": response.get("timing", {}),
            "raw_action_shape": list(raw.shape), "conditioning_receipt_available": False,
        }
