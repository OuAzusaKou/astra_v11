"""Offline terminal decoding and PTY regressions; never import a robot."""
import os
import time
import unittest

from astra.terminal_input import TerminalInput


def decode(raw):
    decoder = TerminalInput()
    return b"".join(result for b in raw if (result := decoder.feed(bytes([b]))) is not None)


class TerminalInputTests(unittest.TestCase):
    def test_plain_controls_preserved(self):
        self.assertEqual(decode(b"bBtTpPrR\r\n"), b"bBtTpPrR\r\n")

    def test_arrows_function_keys_and_terminal_replies(self):
        for raw in (b"\x1b[B", b"\x1bOB", b"\x1b[1;2B", b"\x1bOP",
                    b"\x1b[15;20R", b"\x1b[4;600;800t", b"\x1b[?1;2c"):
            with self.subTest(raw=raw):
                self.assertEqual(decode(raw+b"t"), b"t")

    def test_fragmented_sequence_never_times_out_to_reset(self):
        decoder = TerminalInput()
        self.assertIsNone(decoder.feed(b"\x1b"))
        self.assertIsNone(decoder.feed(b"["))
        time.sleep(.12)
        self.assertIsNone(decoder.feed(b"B"))
        self.assertEqual(decoder.feed(b"b"), b"b")

    def test_paste_and_control_strings_ignored(self):
        for raw in (b"\x1b[200~bBtrp\n\x1b[B\x1b[201~",
                    b"\x1b]52;c;bbbb\x07", b"\x1bPbbtrp\x1b\\"):
            self.assertEqual(decode(raw+b"r"), b"r")

    def test_mouse_and_alt_not_controls(self):
        self.assertEqual(decode(b"\x1b[MbBr\x1bb\x1b[<0;10;10M"), b"")

    @unittest.skipIf(os.name == 'nt', 'POSIX runner PTY')
    def test_real_pty_arrow_does_not_reset_b_still_does(self):
        import pty
        from astra.keyboard import ToggleKeyboard
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        keyboard = ToggleKeyboard(stream=stream)
        try:
            keyboard.start()
            keyboard.arm_start()
            os.write(master, b'\x1b[B\x1bOB\x1b[200~b\ntrp\x1b[201~')
            time.sleep(.2)
            for event in (keyboard.reset_pending, keyboard.start_requested,
                          keyboard.pending, keyboard.retry_pending, keyboard.subtask_pending):
                self.assertFalse(event.is_set())
            os.write(master, b'\n')
            self.assertTrue(keyboard.start_requested.wait(1))
            os.write(master, b'\x1b[15;20R\x1bOP\x1b[B')
            time.sleep(.2)
            self.assertFalse(keyboard.reset_pending.is_set())
            self.assertFalse(keyboard.retry_pending.is_set())
            self.assertFalse(keyboard.subtask_pending.is_set())
            os.write(master, b'b')
            self.assertTrue(keyboard.reset_pending.wait(1))
        finally:
            keyboard.close()
            stream.close()
            os.close(master)


if __name__ == '__main__':
    unittest.main()
