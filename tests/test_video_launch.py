"""Cross-platform launch argument checks; no processes or network."""
import contextlib
import io
import shlex
import sys
import unittest
from unittest.mock import patch

class VideoLaunchTests(unittest.TestCase):
    def test_cli_forwards_video_for_v4_and_v8_without_starting(self):
        from wam_direct.start_mac_vlm import main
        video = "/home/liberai/demos/示范 video's notebook.mp4"
        for version in (4, 8):
            output = io.StringIO()
            argv = ['start_mac_vlm.py', '--dry-run', '--control-version', str(version),
                    '--reference-video', video, '--reference-video-frames', '20']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
                main()
            command = next(line for line in output.getvalue().splitlines() if line.startswith('python3 '))
            args = shlex.split(command)
            self.assertEqual(args[args.index('--reference-video')+1], video)
            self.assertEqual(args[args.index('--reference-video-frames')+1], '20')
            self.assertIn('Reference video: '+video, output.getvalue())

    def test_invalid_video_options_fail_before_bridge_start(self):
        from wam_direct.start_mac_vlm import main, leo_command
        for video in ('', 'relative.mp4', 'C:/demo.mp4'):
            with self.assertRaisesRegex(ValueError, 'absolute path on Leo3'):
                leo_command('fixture', 'task', [], reference_video=video)
        for count in (0, 33):
            with self.assertRaises(ValueError):
                leo_command('fixture', 'task', [], reference_video_frames=count)
        with patch.object(sys, 'argv', ['start_mac_vlm.py', '--reference-video', 'C:/demo.mp4']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()

    def test_omitted_video_keeps_existing_behavior(self):
        from wam_direct.start_mac_vlm import leo_command
        for version in (4, 8):
            self.assertNotIn('--reference-video', leo_command('fixture', 'task', [], version=version))
