"""Offline tests; no tmux, GUI, ROS or robot is started."""
import importlib.util
from pathlib import Path
import shlex
import unittest
from unittest.mock import patch, Mock
import subprocess
import tempfile
import os

spec = importlib.util.spec_from_file_location("launcher", Path(__file__).resolve().parents[1] / "wam_direct/launch_tmux.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_prompt_preserved_and_no_ssh(self):
        prompt = "Task: pick $(touch /tmp/nope) 'cup' Subtask: pick cup"
        create, attach = launcher.commands("/robot/my copy", "wam-task1", "wam", ["--mode", "2", "--prompt", prompt])
        inner = shlex.split(create[-1])[2]
        self.assertIn(shlex.join(["--prompt", prompt]), inner)
        self.assertEqual(create[0], "tmux")
        self.assertEqual(attach, ["tmux", "attach-session", "-t", "=wam-task1"])

    def test_dry_run_has_no_side_effects(self):
        with patch.object(launcher.subprocess, "run") as run, patch.object(launcher.subprocess, "Popen") as popen, patch("builtins.print"):
            self.assertEqual(launcher.main(["--name", "test", "--dry-run", "--", "run"]), 0)
            run.assert_not_called()
            popen.assert_not_called()

    def test_no_display_does_not_start_client(self):
        with patch.dict(launcher.os.environ, {}, clear=True), patch.object(launcher, "desktop_environment", return_value={}), patch.object(launcher.subprocess, "Popen"), patch.object(launcher.shutil, "which", return_value="/bin/tmux"), patch.object(launcher.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                launcher.main(["--name", "test", "--", "run"])
            run.assert_not_called()

    def test_existing_session_does_not_open_window(self):
        with patch.object(launcher.shutil, "which", return_value="/bin/tmux"), patch.object(launcher, "terminal_command", return_value=["xterm"]), patch.object(launcher.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "tmux")), patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.main(["--name", "test", "--", "run"])
            popen.assert_not_called()

    def test_window_failure_reports_existing_session(self):
        with patch.object(launcher.shutil, "which", return_value="/bin/tmux"), patch.object(launcher, "terminal_command", return_value=["xterm"]), patch.object(launcher.subprocess, "run"), patch.object(launcher.subprocess, "Popen", side_effect=OSError("failed")), patch("builtins.print") as output:
            self.assertEqual(launcher.main(["--name", "test", "--", "run"]), 1)
            self.assertTrue(any("remains running" in str(call) for call in output.call_args_list))

    def test_local_terminal_argv(self):
        with patch.dict(launcher.os.environ, {"DISPLAY": ":0"}), patch.object(launcher.shutil, "which", side_effect=lambda name: "/bin/gnome-terminal" if name == "gnome-terminal" else None):
            self.assertEqual(launcher.terminal_command(["tmux", "attach"]), ["/bin/gnome-terminal", "--window", "--", "tmux", "attach"])

    def test_gui_environment_recovered_without_importing_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Path(directory) / "123"
            process.mkdir()
            (process / "comm").write_text("gnome-terminal-\n")
            (process / "environ").write_bytes(b"DISPLAY=:0\0XDG_RUNTIME_DIR=/run/user/1000\0SECRET=hidden\0")
            with patch.dict(os.environ, {}, clear=True):
                env = launcher.desktop_environment(Path(directory))
            self.assertEqual(env["DISPLAY"], ":0")
            self.assertNotIn("SECRET", env)

    def test_attached_window_is_confirmed_by_session_listing(self):
        def result(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "test:1\n" if "list-sessions" in command else "", "")
        with patch.object(launcher.shutil, "which", return_value="/bin/tmux"), patch.object(launcher, "terminal_command", return_value=["xterm"]), patch.object(launcher.subprocess, "run", side_effect=result), patch.object(launcher.subprocess, "Popen", return_value=Mock(wait=Mock(return_value=0))), patch("builtins.print"):
            self.assertEqual(launcher.main(["--name", "test", "--", "run"]), 0)

    def test_invalid_session_rejected(self):
        with self.assertRaises(ValueError):
            launcher.commands("/copy", "a:b", "astra", ["run"])
