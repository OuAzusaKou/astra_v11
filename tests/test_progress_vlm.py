import io
import socket
import threading
import unittest

from progress_vlm.frames import Frame, Frames, Trend
from progress_vlm.protocol import Client, CHECKPOINT_SHA256, PROTOCOL, parse_prediction, receive, request, send, validate
from progress_vlm.server import Server


class FrameTests(unittest.TestCase):
    def frames(self):
        f = Frames()
        for stamp in (9., 9.5, 10.):
            f.add("main", Frame(stamp, 20.+stamp-10., b"main"))
        f.add("left", Frame(9.98, 20., b"left"))
        f.add("right", Frame(10.02, 20., b"right"))
        return f

    def test_timestamp_history_and_old_l2_start(self):
        start = Frame(1., 1., b"start")
        images = self.frames().assemble(start, 20.1, 10.2)
        self.assertEqual([f.data for f in images], [b"main", b"start", b"main", b"left", b"right"])
        self.assertEqual(images[0].stamp, 9.)

    def test_missing_one_second_history(self):
        f = self.frames()
        f.cameras["main"].popleft()
        with self.assertRaisesRegex(ValueError, "one second"):
            f.assemble(Frame(1., 1., b"start"), 20.1, 10.2)

    def test_stale_and_unsynchronized_cameras(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            self.frames().assemble(Frame(1., 1., b"start"), 25., 10.2)
        f = self.frames()
        f.cameras["left"].clear()
        f.add("left", Frame(9.7, 20., b"left"))
        with self.assertRaisesRegex(ValueError, "synchronization"):
            f.assemble(Frame(1., 1., b"start"), 20.1, 10.2)

    def test_out_of_order_camera_ignored(self):
        f = self.frames()
        f.add("main", Frame(9.8, 21., b"old"))
        self.assertEqual(f.latest().stamp, 10.)


class TrendTests(unittest.TestCase):
    def test_no_instant_plateau(self):
        t = Trend(seconds=4., min_samples=3)
        for stamp in range(4):
            self.assertFalse(t.add(stamp, 30, False)["plateau_candidate"])
        self.assertTrue(t.add(4, 30, False)["plateau_candidate"])

    def test_progress_and_gap_clear_plateau(self):
        t = Trend(seconds=4., min_samples=3)
        for stamp in range(5):
            result = t.add(stamp, 30 if stamp < 4 else 40, False)
        self.assertFalse(result["plateau_candidate"])
        self.assertEqual(t.add(20, 40, True)["samples"], 1)
        t.reset()
        self.assertEqual(t.add(21, 10, False)["replay_streak"], 0)

    def test_duplicate_not_counted_as_new_round(self):
        t = Trend()
        t.add(1., 30, True)
        with self.assertRaises(ValueError):
            t.add(1., 30, True)
        self.assertEqual(t.replay_streak, 1)


class ProtocolTests(unittest.TestCase):
    def test_generated_text_contract(self):
        self.assertEqual(parse_prediction("The subtask is 70% complete.")["progress"], 70)
        self.assertTrue(parse_prediction("The subtask is 10% complete. This is an anomalous replay.")["anomalous_replay"])
        for bad in ("", "70%", "The subtask is 0% complete.", "The subtask is 110% complete."):
            self.assertFalse(parse_prediction(bad)["valid"])
            self.assertIsNone(parse_prediction(bad)["progress"])

    def test_five_image_order_required(self):
        msg = request([b"x"] * 5, "task", "subtask", "session", "1")
        validate(msg)
        msg["image_roles"][0] = "current_main"
        with self.assertRaises(ValueError):
            validate(msg)

    def test_packet_size_bound(self):
        a, b = socket.socketpair()
        try:
            a.sendall((2**31).to_bytes(4, "big"))
            with self.assertRaises(ValueError):
                receive(b)
        finally:
            a.close(); b.close()

    def test_identity_mismatch_drops_socket(self):
        a, b = socket.socketpair()
        def wrong_reply():
            msg = receive(b)
            send(b, {"protocol": PROTOCOL, "request_id": "old", "session_id": msg["session_id"], "subtask_id": msg["subtask_id"]})
            b.close()
        worker = threading.Thread(target=wrong_reply)
        worker.start()
        client = Client(); client.sock = a
        with self.assertRaisesRegex(ValueError, "identity"):
            client.infer(request([b"x"]*5, "task", "subtask", "session", "1"))
        self.assertIsNone(client.sock)
        worker.join()

    def test_actual_tcp_and_busy_has_no_inference_queue(self):
        class Fake:
            identity = {"step": 40000, "checkpoint_sha256": CHECKPOINT_SHA256}
            calls = 0
            def predict(self, images, l3, l2):
                self.calls += 1
                return parse_prediction("The subtask is 30% complete.")
        predictor = Fake()
        with Server(("127.0.0.1", 0), predictor, io.StringIO()) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = Client(*server.server_address)
            try:
                result = client.infer(request([b"x"]*5, "task", "subtask", "session", "1"))
                self.assertEqual(result["progress"], 30)
                server.gpu_lock.acquire()
                try:
                    with self.assertRaisesRegex(RuntimeError, "server_busy"):
                        client.infer(request([b"x"]*5, "task", "subtask", "session", "1"))
                finally:
                    server.gpu_lock.release()
                self.assertEqual(predictor.calls, 1)
            finally:
                client.close()
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
