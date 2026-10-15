"""Exercise real runner state transitions with fake sensors; no ROS/actions/network."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from astra.control import Controller, Settings
from astra.session import Session, atomic_json
from progress_vlm.follow import SessionFollower
from progress_vlm.observe import Observer
from wam_direct.launch_tmux import progress_commands
from test_astra_mode2 import Robot, Policy, observation


class FollowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "runner"
        self.robot = Robot()
        self.session = Session(self.path, "test", {"instruction": "put pepper in bowl", "initial_l2_instruction": "pick pepper"})
        self.controller = Controller(self.robot, Policy(), self.session, Settings(400, version=3),
                                     "put pepper in bowl", "pick pepper", next_subtasks=["place pepper in bowl"])
        self.follow = SessionFollower(self.path)
        self.controller.start()

    def test_same_l2_retains_first_wam_image_until_p_enter(self):
        c = self.controller
        self.assertFalse(self.follow.poll()["active"])
        c.infer(review=False)
        first = self.follow.poll()
        self.assertTrue(first["active"])
        self.assertEqual(first["l2"], "pick pepper")
        self.assertEqual(first["anchor"].stamp, 1.)
        fresh = observation(); fresh["timestamp"] = 2.
        self.robot.observe = lambda: fresh
        c.infer(review=False)
        self.assertEqual(self.follow.poll()["anchor"], first["anchor"])
        self.assertTrue(c.next_subtask())
        pending = self.follow.poll()
        self.assertFalse(pending["active"])
        self.assertEqual(pending["l2"], "place pepper in bowl")
        self.assertNotEqual(pending["key"], first["key"])
        c.start()
        self.assertFalse(self.follow.poll()["active"])
        c.infer(review=False)
        second = self.follow.poll()
        self.assertTrue(second["active"])
        self.assertEqual(second["anchor"].stamp, 2.)
        self.assertEqual(second["l3"], first["l3"])
        self.assertEqual(self.robot.published, [])

    def test_t_does_not_reset_l2_anchor(self):
        c = self.controller
        c.infer(review=False)
        first = self.follow.poll()
        c.keyboard_toggle()
        astra = self.follow.poll()
        self.assertEqual(astra["control_mode"], "ASTRA")
        self.assertEqual(astra["anchor"], first["anchor"])
        c.keyboard_toggle()
        self.assertEqual(self.follow.poll()["key"], first["key"])

    def test_b_pause_exit_and_stale_disable_predictions(self):
        c = self.controller
        c.infer(review=False)
        first = self.follow.poll()
        c.pause("synthetic test")
        self.assertFalse(self.follow.poll()["active"])
        self.robot.reset_to_task_pose = Mock(return_value={})
        c.operator_reset()
        reset = self.follow.poll()
        self.assertFalse(reset["active"])
        self.assertNotEqual(reset["key"], first["key"])
        c.start(); c.infer(review=False)
        current = json.loads((self.path/"status.json").read_text())
        self.assertFalse(self.follow.poll(now=current["updated_at"]+16.)["active"])
        c.state="EXITED"; c.status()
        self.assertFalse(self.follow.poll()["active"])

    def test_late_attachment_uses_first_not_latest_frame(self):
        self.controller.infer(review=False)
        fresh=observation(); fresh["timestamp"]=3.
        self.robot.observe=lambda: fresh
        self.controller.infer(review=False)
        self.assertEqual(SessionFollower(self.path).poll()["anchor"].stamp, 1.)

    def test_observer_invalidates_old_epoch_and_ignores_its_own_p(self):
        args=SimpleNamespace(host="unused", port=29573, follow_session=self.path, l3="", l2="", next_l2=[],
                             hz=1., plateau_seconds=8., max_camera_age=1., output=Path(self.temp.name)/"observer")
        observer=Observer(args,None)
        self.addCleanup(observer.log.close)
        self.controller.infer(review=False)
        observer.sync_follow()
        epoch=observer.epoch
        observer.latest=({"progress":80},0.)
        observer.start_subtask(advance=True)
        self.assertEqual(observer.epoch,epoch)
        self.controller.next_subtask()
        observer.sync_follow()
        self.assertGreater(observer.epoch,epoch)
        self.assertIsNone(observer.latest)
        self.assertIsNone(observer.start_frame)
        self.assertEqual(observer.tasks,["place pepper in bowl"])
        self.controller.start(); self.controller.infer(review=False)
        observer.sync_follow()
        self.assertIsNotNone(observer.start_frame)

    def test_foreign_status_refused(self):
        self.controller.infer(review=False)
        status=json.loads((self.path/"status.json").read_text())
        status["session_id"]="other"
        atomic_json(self.path/"status.json",status)
        with self.assertRaises(ValueError): self.follow.poll()


class LauncherFollowTests(unittest.TestCase):
    def test_follower_bound_to_runner_path_and_separate_output(self):
        with tempfile.TemporaryDirectory() as d:
            cmd,out=progress_commands(d,"run1",["run","--session","outputs/astra/run1"],"pc",29573,{"DISPLAY":":0", "ROS_DOMAIN_ID":"103"})
            self.assertIn("VLM-progress",cmd)
            self.assertIn("--follow-session",cmd[-1])
            self.assertIn(str(Path(d)/"outputs/astra/run1"),cmd[-1])
            self.assertIn("DISPLAY=:0",cmd[-1])
            self.assertIn("ROS_DOMAIN_ID=103",cmd[-1])
            self.assertEqual(out,Path(d)/"outputs/progress_vlm/run1_follow")

    def test_missing_session_refused(self):
        with self.assertRaises(ValueError): progress_commands("/tmp","x",["run"],"pc",29573,{})


if __name__ == "__main__": unittest.main()
