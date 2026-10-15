"""Version-specific launch flags and candidate publication gates, offline only."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from astra.control import Controller,Settings
from astra.session import Session
from test_astra_mode2 import Robot,Policy
from wam_direct.start_mac_vlm import leo_command

class VersionRoutes(unittest.TestCase):
    def test_launch_matrix(self):
        for version in (4,5,6,7,8):
            with self.subTest(version=version):
                cmd=leo_command('fixture','Task: task Subtask: pick',[],version=version)
                self.assertIn(f'version{version}_',cmd)
                self.assertEqual('--dynamics-mode' in cmd,version in (5,8))
                self.assertEqual('--dynamics-view' in cmd,version in (5,8))
                self.assertEqual('--pointcloud-mode' in cmd,version in (6,7))
                self.assertEqual('--rviz-record' in cmd,version in (6,7))
        self.assertIn('version4_',leo_command('fixture','Task: task Subtask: pick',[]))

    def test_only_v7_accepts_long_plan_and_v4_does_not_imagine(self):
        for version in (4,5,6,7):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as tmp:
                robot=Robot();session=Session(Path(tmp)/'run','test',{})
                c=Controller(robot,Policy(),session,Settings(400,version=version),'task',initial_l2='pick')
                c.start();c.keyboard_toggle();c.imagination=Mock()
                identity={k:session.pending[k] for k in ('session_id','owner','request_id')}
                d=dict(identity,operation='takeover',reason='test',evidence='test',new_attempt=True,
                       change_reason='test',actions=np.zeros((60,52)).tolist(),execute_steps=60,
                       short_batch_reason='milestone',plan_stages=[dict(kind='transport',end_step=60,evidence='fixture')])
                if version!=7:
                    with self.assertRaises(ValueError):c.decide(d)
                    d.update(actions=np.zeros((30,52)).tolist(),execute_steps=30)
                c.decide(d)
                self.assertEqual(bool(robot.published),version==4)
                self.assertEqual(c.imagination.start.called,version in (5,6,7))
