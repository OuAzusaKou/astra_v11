"""Offline fake-robot checks for V4/V8 terminal-only, controller-owned fallback."""
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from astra.control import Controller, Settings
from astra.remote_supervisor import RemoteSupervisor
from astra.session import Session
from astra.supervisor import CodexSupervisor
from test_astra_mode2 import Robot, Policy


class CodeHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.session = Session(Path(self.temp.name)/'run', 'fixture', {})
        self.robot = Robot(); self.policy = Policy()
        self.c = Controller(self.robot, self.policy, self.session,
                            Settings(100, version=4, astra_pause_handback=True), 'open drawer', 'pull drawer')
        self.c.start(); self.c.keyboard_toggle()

    def decision(self, operation, **extra):
        return {**{k:self.session.pending[k] for k in ('session_id','owner','request_id')},
                'operation':operation, 'reason':'fixture', 'evidence':'measured fixture', **extra}

    def test_terminal_pause_kinds_preserve_budget_and_use_fresh_inference(self):
        for version in (4, 8):
            self.c.settings = replace(self.c.settings, version=version)
            for kind in (None, 'planning', 'collision_risk', 'observation_uncertainty', 'technical_error'):
                if self.c.mode == 'WAM': self.c.keyboard_toggle()
                self.c.steps = 17
                old = self.session.pending['request_id']
                self.c.decide(self.decision('pause', pause_kind=kind))
                self.assertEqual((self.c.mode,self.c.state), ('WAM','AUTO'))
                self.assertEqual(self.c.steps,17); self.assertEqual(self.c.l2,'pull drawer')
                self.assertIsNone(self.session.pending); self.assertIsNone(self.c.observation)
                self.assertEqual(len(self.c.queue),0)
                self.assertEqual(self.robot.published,[]); self.assertEqual(self.policy.calls,[])
        self.c.auto_step()
        self.assertEqual(len(self.policy.calls),1)  # New inference, no stale Astra trajectory.

    def test_model_hand_back_is_not_a_supported_operation(self):
        for version in (4,8):
            self.c.settings=replace(self.c.settings,version=version)
            with self.assertRaisesRegex(ValueError,'disabled'):
                self.c.decide(self.decision('hand_back',recovery_evidence='recovered'))
            self.assertEqual(self.c.mode,'ASTRA')
        self.assertEqual(self.robot.published,[])

    def test_faults_and_model_hardware_stop_cannot_auto_resume(self):
        self.c.decide(self.decision('pause',pause_kind='hardware_fault'))
        self.assertFalse(self.c.supervisor_stopped('subsequent timeout'))
        self.assertEqual((self.c.mode,self.c.state),('ASTRA','PAUSED_FOR_OPERATOR'))
        self.robot.rearm=Mock(); self.robot.fault=True
        self.assertFalse(self.c.supervisor_stopped('failure'))
        self.robot.rearm.assert_not_called(); self.assertEqual(self.robot.published,[])

    def test_live_hardware_check_blocks_terminal_fallback(self):
        self.robot.fault=True
        self.assertFalse(self.c.supervisor_stopped('network error'))
        self.assertEqual(self.c.mode,'ASTRA'); self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')

    def test_cancel_manual_switch_and_opt_out_do_not_resume(self):
        for gate in ('cancelled','switch_pending'):
            setattr(self.c,gate,lambda:True)
            self.assertFalse(self.c.supervisor_stopped('failure'))
            self.assertEqual(self.c.mode,'ASTRA')
            setattr(self.c,gate,lambda:False)
        self.c.settings=replace(self.c.settings,astra_pause_handback=False)
        self.assertFalse(self.c.supervisor_stopped('failure'))

    def test_observe_and_complete_are_not_fallback(self):
        self.c.decide(self.decision('observe'))
        self.assertEqual(self.c.mode,'ASTRA'); self.assertEqual(self.c.state,'WAITING')
        self.c.decide(self.decision('complete'))
        self.assertEqual(self.c.state,'FINISHED'); self.assertEqual(self.c.mode,'ASTRA')
        self.assertFalse(self.c.supervisor_stopped('late error'))
        self.assertEqual(self.c.state,'FINISHED')

    def test_run_loop_routes_final_supervisor_error_to_controller(self):
        from astra.run import serve
        stop=threading.Event()
        self.c.start=lambda:None  # Already entered Astra in setUp.
        self.c.auto_step=Mock(side_effect=stop.set)
        supervisor=Mock()
        supervisor.poll.side_effect=RuntimeError('final model error after retries')
        serve(self.c,stop,supervisor=supervisor)
        self.c.auto_step.assert_called_once()
        supervisor.invalidate.assert_called()
        self.assertEqual(self.c.mode,'WAM')
        self.assertEqual(self.robot.published,[])

    def test_run_loop_cannot_fallback_through_hardware_fault(self):
        from astra.run import serve
        stop=threading.Event()
        self.c.start=lambda:None
        self.c.auto_step=Mock()
        supervisor=Mock(); count=0
        def poll(request):
            nonlocal count
            count+=1
            if count==1:
                self.robot.fault=True
                raise RuntimeError('final model error')
            stop.set()
            return None
        supervisor.poll.side_effect=poll
        serve(self.c,stop,supervisor=supervisor)
        self.c.auto_step.assert_not_called()
        self.assertEqual(self.c.mode,'ASTRA')
        self.assertEqual(self.robot.published,[])

    def test_review_timeout_and_transport_exhaustion(self):
        self.c.wait_started=time.monotonic()-1000
        self.c.tick()
        self.assertEqual((self.c.mode,self.c.state),('WAM','AUTO'))
        self.c.keyboard_toggle(); self.c.pause('disconnected')
        remote=RemoteSupervisor(self.session); remote.wait_for_reconnect(self.c)
        remote.reconnect_deadline=time.monotonic()-1
        self.assertTrue(remote.recover(self.c))
        self.assertEqual((self.c.mode,self.c.state),('WAM','AUTO'))
        self.assertEqual(self.robot.published,[])

    def test_shared_prompt_contains_no_handoff_option_or_fallback_flag(self):
        for version in (4,8):
            request=dict(self.session.pending,version=version,astra_pause_handback=True)
            bridge=object.__new__(CodexSupervisor)
            bridge.worker=None; bridge.generation=0; bridge.readonly_remote=None; bridge.log=lambda *a:None
            captured=[]
            def call(prompt, images, cancel, tag):
                captured.append(prompt)
                return {**{k:request[k] for k in ('session_id','owner','request_id')},
                        'operation':'observe','reason':'fixture','evidence':'fixture'}
            bridge._call=call
            bridge.start(request); bridge.worker.join(5)
            self.assertFalse(bridge.worker.is_alive()); self.assertIsNone(bridge.error)
            prompt=captured[0]
            for phrase in ('hand_back','astra_pause_handback','OPERATOR HANDOFF','交还','交给WAM'):
                self.assertNotIn(phrase,prompt)
            self.assertIn('hardware_fault',prompt)
            self.assertIn('CURRENT_REQUEST',prompt)
            if version==8: self.assertIn('operation=propose_candidates',prompt)
            self.assertIn('astra_pause_handback',request)  # Do not mutate runner evidence.


if __name__=='__main__': unittest.main()
