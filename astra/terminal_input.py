"""Recognize only standalone ASCII keys, never bytes inside terminal sequences."""


class TerminalInput:
    def __init__(self):
        self.state = "plain"
        self.sequence = bytearray()
        self.paste_tail = bytearray()
        self.mouse_remaining = 0

    def feed(self, key):
        """Consume one byte; return a standalone key or None.

        Sequence state persists across reads and delays. Timing out a partial
        sequence would let a delayed arrow's final B become a reset command.
        Bracketed paste contents are discarded, including newlines and hotkeys.
        """
        if len(key) != 1:
            raise ValueError("one byte required")
        value = key[0]
        if self.state == "paste":
            self.paste_tail += key
            self.paste_tail = self.paste_tail[-6:]
            if self.paste_tail == b"\x1b[201~":
                self.state = "plain"
                self.paste_tail.clear()
            return None
        if self.state == "mouse":
            self.mouse_remaining -= 1
            if not self.mouse_remaining:
                self.state = "plain"
            return None
        if self.state == "string":
            if key == b"\x07":
                self.state = "plain"
            elif key == b"\x1b":
                self.state = "string_escape"
            return None
        if self.state == "string_escape":
            self.state = "plain" if key == b"\\" else "string"
            return None
        if key == b"\x1b":
            self.state = "escape"
            self.sequence.clear()
            return None
        if self.state == "escape":
            if key == b"[":
                self.state = "csi"
            elif key == b"O":
                self.state = "ss3"
            elif key in (b"]", b"P", b"^", b"_", b"X"):
                self.state = "string"
            elif 0x20 <= value <= 0x2f:
                self.state = "escape_intermediate"
            else:
                self.state = "plain"  # Also ignore Alt+letter.
            return None
        if self.state == "escape_intermediate":
            if 0x30 <= value <= 0x7e:
                self.state = "plain"
            return None
        if self.state in ("csi", "ss3"):
            if len(self.sequence) < 64:
                self.sequence += key
            if 0x40 <= value <= 0x7e:
                if self.state == "csi" and self.sequence == b"200~":
                    self.state = "paste"
                elif self.state == "csi" and self.sequence == b"M":
                    self.state = "mouse"
                    self.mouse_remaining = 3
                else:
                    self.state = "plain"
                self.sequence.clear()
            return None
        return key if value < 0x80 else None
