import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from astra.control import Controller, Settings
from astra.session import Session
from test_astra_mode2 import Robot, Policy

class WamContextTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.robot=Robot()
        self.c=Controller(self.robot,Policy(),Session(Path(self.temp.name)/'r','test',{}),Settings(1000,version=2),'task','pick',next_subtasks=['place'])
        self.c.start()

    def test_takeover_preserves_reference_without_execution_queue(self):
        self.c.auto_step();self.c.auto_step()
        self.c.keyboard_toggle()
        r=self.c.session.pending
        self.assertEqual(r['proposed_actions'],[])
        self.assertEqual(len(self.c.queue),0)
        record=r['wam_context']['trajectories'][0]
        self.assertEqual(record['row_status'],['published']*15+['not_published_discarded']*15)
        self.assertEqual(len(record['actions']),30)
        self.assertEqual(record['l2_instruction'],'pick')
        self.assertEqual(len(record['observation']['images']),3)
        self.assertNotEqual(record['observation']['state_path'],r['observation']['state_path'])
        self.assertFalse(r['wam_context']['physical_execution_confirmed'])
        self.assertEqual(len(self.robot.published),1)

    def test_partial_publish_not_reported_as_success(self):
        self.c.auto_step()
        self.robot.publish=Mock(side_effect=RuntimeError('partial write'))
        with self.assertRaises(RuntimeError):self.c.auto_step()
        self.c.pause('failed')
        record=self.c.wam_context()['trajectories'][0]
        self.assertEqual(record['row_status'][:15],['publication_attempted']*15)
        self.assertEqual(record['row_status'][15:],['not_published_discarded']*15)

    def test_history_bounded_snapshot_is_immutable_and_reset_clears(self):
        for _ in range(4):self.c.infer(review=False)
        self.assertEqual(len(self.c.wam_history),3)
        snapshot=self.c.wam_context()
        self.c.auto_step()
        self.assertEqual(snapshot['trajectories'][-1]['row_status'][0],'not_published')
        self.robot.reset_to_task_pose=Mock(return_value={})
        self.c.operator_reset()
        self.assertEqual(self.c.wam_history,[])

    def test_changed_subtask_marks_old_reference(self):
        self.c.auto_step();self.c.next_subtask();self.c.start()
        self.assertFalse(self.c.wam_context()['trajectories'][0]['matches_current_condition'])
