import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from astra.control import Controller, Settings
from astra.session import Session
from test_astra_mode2 import Robot, Policy

class SubtaskTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.robot=Robot();self.policy=Policy()
        self.c=Controller(self.robot,self.policy,Session(Path(self.temp.name)/'r','test',{}),Settings(400,version=2),'put pepper in bowl','pick pepper',next_subtasks=['place pepper in bowl'])
        self.c.start()

    def test_switch_waits_and_discards_old_chunk(self):
        self.c.auto_step()
        self.assertEqual(len(self.c.queue),30)
        self.assertTrue(self.c.next_subtask())
        self.assertEqual(self.c.state,'WAITING_FOR_START')
        self.assertEqual(len(self.c.queue),0)
        self.c.auto_step()
        self.assertEqual(len(self.policy.calls),1)
        self.assertEqual(self.robot.published,[])
        self.c.start();self.c.auto_step()
        self.assertEqual(self.policy.calls[-1],'Task: put pepper in bowl Subtask: place pepper in bowl')
        self.assertFalse(self.c.next_subtask())
        self.assertEqual(self.c.subtask_index,1)

    def test_reset_restores_first_subtask(self):
        self.c.next_subtask()
        self.robot.reset_to_task_pose=Mock(return_value={})
        self.c.operator_reset()
        self.assertEqual(self.c.subtask_index,0)
        self.assertEqual(self.c.l2,'pick pepper')

    def test_p_does_not_steal_astra_control(self):
        self.c.keyboard_toggle()
        request=self.c.session.pending
        self.assertFalse(self.c.next_subtask())
        self.assertIs(self.c.session.pending,request)
        self.assertEqual(self.c.mode,'ASTRA')
