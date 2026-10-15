"""A latched terminal toggle; the runner alone applies control transitions."""
import os
import select
import sys
import termios
import threading
import time
import tty

from .terminal_input import TerminalInput


class ToggleKeyboard:
    def __init__(self, key="t", stream=None, enable_toggle=True):
        self.key = key.lower()
        if self.key in ("b", "p", "r"):
            raise ValueError("B/P/R are reserved for reset/subtask/retry")
        self.retry_pending = threading.Event()
        self.reset_pending = threading.Event()
        self.subtask_pending = threading.Event()
        self.enable_toggle = enable_toggle
        self.stream = stream if stream is not None else sys.stdin
        self.pending = threading.Event()
        self.ready = threading.Event()
        self.start_requested = threading.Event()
        self.input_lock = threading.Lock()
        self.closed = threading.Event()
        self.thread = None
        self.attrs = None
        self.fd = None
        self.decoder = TerminalInput()

    def start(self):
        if not self.stream.isatty():
            raise ValueError("runner requires an interactive terminal/tmux pane for Enter start")
        self.fd = self.stream.fileno()
        self.attrs = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)
            # Ask the terminal to delimit pasted text so it cannot become
            # a sequence of robot hotkeys (including Enter and B).
            os.write(self.fd, b"\x1b[?2004h")
            self.thread = threading.Thread(target=self._read, daemon=True, name="astra-toggle")
            self.thread.start()
        except BaseException:
            self.close()
            raise
        print("[CONTROL] P selects the next configured subtask; Enter confirms", flush=True)
        print("[CONTROL] B returns to the configured task initial pose; Enter is required afterwards", flush=True)
        if self.enable_toggle:
            print("[CONTROL] R retries paused Astra with fresh observations; no WAM hand-back", flush=True)
            print(f"[CONTROL] {self.key.upper()} toggles WAM / ASTRA at a batch boundary; ROS pedal remains ESTOP", flush=True)

    def arm_start(self):
        with self.input_lock:
            termios.tcflush(self.fd, termios.TCIFLUSH)
            self.decoder = TerminalInput()
            self.start_requested.clear()
            self.pending.clear()
            self.reset_pending.clear()
            self.subtask_pending.clear()
            self.retry_pending.clear()
            print("[READY] Press Enter to START the session. No model actions will be sent before Enter.", flush=True)
            self.ready.set()

    def _read(self):
        try:
            while not self.closed.is_set():
                if select.select([self.fd], [], [], .1)[0]:
                    with self.input_lock:
                        # arm_start may have flushed input since the outer select.
                        if not select.select([self.fd], [], [], 0)[0]:
                            continue
                        key = os.read(self.fd, 1)
                        if not key:
                            break
                        key = self.decoder.feed(key)
                        if key is None:
                            continue
                        if self.ready.is_set() and key.lower() == b"b":
                            if not self.reset_pending.is_set():
                                self.reset_pending.set()
                                print(f"[INPUT {time.time():.6f}] standalone reset byte=0x{key.hex()} tty={os.ttyname(self.fd)}", flush=True)
                                print("[CONTROL] B RESET requested; waiting for current batch boundary", flush=True)
                            continue
                        if not self.start_requested.is_set():
                            if self.ready.is_set() and key in (b"\r", b"\n"):
                                self.start_requested.set()
                                print("[CONTROL] START requested by Enter", flush=True)
                            continue
                        if self.enable_toggle and key.lower() == b"r":
                            self.retry_pending.set()
                            print("[CONTROL] R ASTRA RETRY requested", flush=True)
                            continue
                        if key.lower() == b"p":
                            self.subtask_pending.set()
                            print("[CONTROL] P NEXT SUBTASK requested; waiting for batch boundary", flush=True)
                            continue
                        if self.enable_toggle and key.decode("utf-8", "ignore").lower() == self.key and not self.pending.is_set():
                            self.pending.set()
                            print("[CONTROL] SWITCH REQUESTED — waiting for current inference/batch boundary", flush=True)
        finally:
            self.closed.set()

    def close(self):
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.attrs is not None:
            os.write(self.fd, b"\x1b[?2004l")
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.attrs)
            self.attrs = None
