import struct,unittest
from unittest.mock import patch,Mock
import numpy as np
from astra.rviz_record import decode_xwd,find_window
from wam_direct.launch_tmux import rviz_record_commands
from wam_direct.start_mac_vlm import leo_command


class RecorderTests(unittest.TestCase):
    def test_pixel_masks_and_endian(self):
        for order,endian in ((0,'<'),(1,'>')):
            h=[100,7,2,24,2,1,0,order,32,order,32,32,8,4,0xff0000,0xff00,0xff,8,256,0,2,1,0,0,0]
            data=struct.pack('>25I',*h)+struct.pack(endian+'2I',0x1122ff,0xff3377)
            np.testing.assert_array_equal(decode_xwd(data),[[[255,34,17],[119,51,255]]])
        with self.assertRaises(ValueError):decode_xwd(data[:-1])

    def test_24bit_padded(self):
        h=[100,7,2,24,1,1,0,0,32,0,32,24,4,4,0xff0000,0xff00,0xff,8,256,0,1,1,0,0,0]
        np.testing.assert_array_equal(decode_xwd(struct.pack('>25I',*h)+bytes([3,2,1,0])),[[[3,2,1]]])

    def test_window_selection(self):
        lines='''0x1 "view_robot.rviz - RViz": ("rviz2" "rviz2") 1117x1164+0+0
0x2 "version6_pointcloud.rviz - RViz": ("mutter-x11-frames" "mutter-x11-frames") 1200x1000+0+0
0x3 "version6_pointcloud.rviz - RViz": ("rviz2" "rviz2") 1117x1164+0+0'''
        with patch('astra.rviz_record.subprocess.run',return_value=Mock(stdout=lines)):
            self.assertEqual(find_window({}),'0x3')
        with patch('astra.rviz_record.subprocess.run',return_value=Mock(stdout=lines+'\n'+lines)):
            with self.assertRaises(RuntimeError):find_window({})

    def test_launcher(self):
        cmd=rviz_record_commands('/tmp/repo','run',['run','--session','/tmp/run'],{'DISPLAY':':0'})
        self.assertIn('RViz-recording',cmd);self.assertIn('astra.rviz_record',cmd[-1])
        self.assertIn('/tmp/repo/outputs/rviz/run',cmd[-1]);self.assertIn('DISPLAY=:0',cmd[-1])
        self.assertIn('--rviz-record',leo_command('r','Task: a Subtask: b',[],version=6))
        self.assertNotIn('--rviz-record',leo_command('r','Task: a Subtask: b',[],version=5))
