"""V4 transition tests with fake sensors and publishers; no ROS/network/motion."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from astra.control import Controller, Settings
from astra.run import serve, load_control_config
from astra.session import Session
from astra.vlm_gate import ProgressGate
from progress_vlm.protocol import CHECKPOINT_SHA256, PROTOCOL
from test_astra_mode2 import Robot, Policy
from wam_direct.start_mac_vlm import leo_command


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'predictions.jsonl'
        self.robot, self.policy = Robot(), Policy()
        self.session = Session(self.root / 'runner', 'test', {})
        self.c = Controller(self.robot, self.policy, self.session, Settings(400, version=4),
                            'pick and place', initial_l2='pick', next_subtasks=['place', 'finish'])
        self.c.start()
        self.gate = self.c.vlm_gate = ProgressGate(self.path)
        self.gate.poll(self.c, now=100.)

    def row(self, stamp, progress=100, **changes):
        c = self.c
        row = {'event': 'prediction', 'time': stamp + .5, 'capture_stamp': stamp,
               'age_on_arrival': .5, 'request_id': str(stamp), 'valid': True, 'success': True,
               'protocol': PROTOCOL, 'model': {'step': 40000, 'checkpoint_sha256': CHECKPOINT_SHA256},
               'l3': c.l3, 'l2': c.l2, 'progress': progress, 'anomalous_replay': False,
               'runner_context': {'session_id': c.session.id, 'condition_version': c.condition_version,
                                  'phase': c.phase, 'active': True, 'control_mode': 'WAM', 'state': 'EXECUTING'}}
        row.update(changes)
        return row

    def append(self, *rows):
        with self.path.open('a') as f:
            for row in rows:
                f.write(json.dumps(row)+'\n')

    def highs(self):
        self.append(self.row(103), self.row(104))

    def test_two_fresh_predictions_switch_once_discard_queue_preserve_budget(self):
        self.c.queue = np.ones((17, 52)); self.c.steps = 120
        self.highs()
        with patch('astra.vlm_gate.time.time', return_value=105.):
            self.assertTrue(self.c.try_vlm_subtask())
            self.assertFalse(self.c.try_vlm_subtask())
        self.assertEqual((self.c.l2, self.c.condition_version, self.c.state), ('place', 2, 'AUTO'))
        self.assertEqual(self.c.steps, 120)
        self.assertEqual(len(self.c.queue), 0)
        self.assertEqual(self.robot.published, [])
        self.c.auto_step()
        self.assertEqual(self.policy.calls, ['Task: pick and place Subtask: place'])
        self.assertEqual(self.robot.published, [])

    def test_one_frame_or_duplicate_cannot_confirm(self):
        self.append(self.row(103), self.row(103))
        self.assertIsNone(self.gate.poll(self.c, now=104))
        self.append(self.row(104))
        self.assertIsNone(self.gate.poll(self.c, now=105))

    def test_low_anomalous_invalid_and_error_break_streak(self):
        for middle in (self.row(103.5, 70), self.row(103.5, anomalous_replay=True),
                       self.row(103.5, valid=False), {'event': 'observation_error'},
                       {'event': 'discarded_result'}):
            with self.subTest(middle=middle):
                self.path.write_text('')
                self.gate = ProgressGate(self.path)
                self.gate.poll(self.c, now=100)
                self.append(self.row(103), middle, self.row(104))
                self.assertIsNone(self.gate.poll(self.c, now=105))

    def test_wrong_session_condition_phase_and_checkpoint_rejected(self):
        for field, value in [('session_id', 'other'), ('condition_version', 99), ('phase', 99), ('control_mode', 'ASTRA')]:
            self.path.write_text('')
            gate = ProgressGate(self.path); gate.poll(self.c, now=100)
            a,b = self.row(103), self.row(104)
            a['runner_context'][field] = b['runner_context'][field] = value
            self.append(a,b)
            self.assertIsNone(gate.poll(self.c, now=105))
        self.path.write_text('')
        gate=ProgressGate(self.path);gate.poll(self.c, now=100)
        self.append(self.row(103, model={}), self.row(104, model={}))
        self.assertIsNone(gate.poll(self.c, now=105))

    def test_expired_future_gap_and_too_close_frames_rejected(self):
        for stamps, now in [((103,104),110), ((110,111),105), ((101,104),105), ((103,103.2),104)]:
            self.path.write_text('')
            gate=ProgressGate(self.path);gate.poll(self.c, now=100)
            self.append(*(self.row(t) for t in stamps))
            self.assertIsNone(gate.poll(self.c, now=now))

    def test_manual_controls_pause_takeover_and_final_l2_block(self):
        self.highs()
        for state,mode in [('WAITING_FOR_START','WAM'),('PAUSED_FOR_OPERATOR','WAM'),('AUTO','ASTRA'),('EXECUTING','WAM')]:
            self.c.state,self.c.mode=state,mode
            self.assertIsNone(self.gate.poll(self.c, now=105))
        self.c.state,self.c.mode='AUTO','WAM'
        self.c.switch_pending=lambda:True
        self.assertIsNone(self.gate.poll(self.c, now=105))
        self.c.switch_pending=lambda:False
        self.c.subtask_index=2
        self.assertIsNone(self.gate.poll(self.c, now=105))
        self.assertEqual(self.c.state,'AUTO')

    def test_no_auto_restart_after_manual_p_or_b(self):
        self.highs()
        self.c.next_subtask()
        with patch('astra.vlm_gate.time.time', return_value=105):
            self.assertFalse(self.c.try_vlm_subtask())
        self.assertEqual(self.c.state,'WAITING_FOR_START')
        self.robot.reset_to_task_pose=Mock(return_value={})
        self.c.operator_reset()
        with patch('astra.vlm_gate.time.time', return_value=106):
            self.assertFalse(self.c.try_vlm_subtask())
        self.assertEqual((self.c.l2,self.c.state),('pick','WAITING_FOR_START'))

    def test_hardware_fault_prevents_transition(self):
        self.highs();self.robot.fault=True
        with patch('astra.vlm_gate.time.time', return_value=105), self.assertRaisesRegex(RuntimeError,'ESTOP'):
            self.c.try_vlm_subtask()
        self.assertEqual(self.c.l2,'pick')
        self.assertEqual(self.robot.published,[])

    def test_partial_malformed_log_and_replacement(self):
        self.path.write_text('{')
        self.assertIsNone(self.gate.poll(self.c,now=103))
        self.path.write_text('bad json\n')
        self.assertIsNone(self.gate.poll(self.c,now=104))
        replacement=self.root/'replacement.jsonl'
        replacement.write_text(json.dumps(self.row(103))+'\n'+json.dumps(self.row(104))+'\n')
        replacement.replace(self.path)
        self.assertIsNotNone(self.gate.poll(self.c,now=105))

    def test_hold_minimum_l2_time(self):
        self.append(self.row(100.1),self.row(101.1))
        self.assertIsNone(self.gate.poll(self.c,now=102))

    def test_v3_remains_manual(self):
        self.highs();self.c.settings=Settings(400,version=3)
        self.assertIsNone(self.gate.poll(self.c,now=105))
        self.assertTrue(self.c.next_subtask())
        self.assertEqual(self.c.state,'WAITING_FOR_START')

    def test_serve_only_switches_after_drain_before_next_inference(self):
        stop=threading.Event();clock=[100.]
        events=[]
        keyboard=SimpleNamespace(pending=threading.Event(),reset_pending=threading.Event(),
            subtask_pending=threading.Event(),retry_pending=threading.Event(),closed=threading.Event(),
            start_requested=threading.Event(),arm_start=Mock())
        keyboard.start_requested.set()
        self.c.vlm_gate=ProgressGate(self.path,min_l2_seconds=0)
        original=self.policy.infer
        def infer(obs,prompt):
            events.append(('infer',self.c.l2))
            result=original(obs,prompt)
            if self.c.l2=='place':stop.set()
            return result
        def drain(cancelled):
            events.append(('drain_begin',self.c.l2))
            self.append(self.row(101),self.row(102))
            clock[0]=103.
            events.append(('drain_end',self.c.l2))
        self.policy.infer=infer;self.robot.drain=drain
        with patch('astra.vlm_gate.time.time',side_effect=lambda:clock[0]):
            serve(self.c,stop,keyboard)
        self.assertEqual(events,[('infer','pick'),('drain_begin','pick'),('drain_end','pick'),('infer','place')])
        self.assertEqual(len(self.robot.published),1)
        self.assertEqual(keyboard.arm_start.call_count,1)

    def test_pause_resume_discards_pre_pause_evidence(self):
        self.highs()
        with patch('astra.vlm_gate.time.time', return_value=105):
            self.c.pause('temporary interruption')
        self.c.state='AUTO'  # Model a separately authorized resume.
        self.assertIsNone(self.gate.poll(self.c, now=105.1))
        self.assertEqual(self.gate.highs, [])

    def test_v4_manual_handback_keeps_l2_and_requires_new_phase_evidence(self):
        self.highs()
        self.c.keyboard_toggle()
        self.assertIsNone(self.gate.poll(self.c, now=105))
        pending=self.session.pending
        decision={k:pending[k] for k in ('session_id','owner','request_id')}
        decision.update(operation='hand_back',reason='recovered',evidence='fake test feedback',
                        recovery_evidence='synthetic recovery')
        with self.assertRaisesRegex(ValueError,'disabled'):
            self.c.decide(decision)
        self.c.keyboard_toggle()
        self.assertEqual((self.c.mode,self.c.state,self.c.l2),('WAM','AUTO','pick'))
        self.assertIsNone(self.gate.poll(self.c,now=105.1))
        self.assertEqual(self.robot.published,[])

    def test_v4_command_binds_exact_log_and_v3_omits_it(self):
        command=leo_command('test_run','Task: pick and place Subtask: pick',['place'])
        self.assertIn('version4_vlm_subtasks.json',command)
        self.assertIn('/test_run_follow/predictions.jsonl',command)
        self.assertNotIn('--vlm-progress-log',leo_command('test_run','Task: x Subtask: y',['z'],version=3))
        config=Path(__file__).resolve().parents[1]/'config/astra/version4_vlm_subtasks.json'
        self.assertEqual(load_control_config(config)['version'],4)


if __name__=='__main__':unittest.main()
