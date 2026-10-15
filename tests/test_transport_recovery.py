"""Fake robot only: automatic reconnect cannot clear faults or replay actions."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import Mock

from astra.control import Controller, Settings
from astra.session import Session
from astra.remote_supervisor import RemoteSupervisor, RemoteDisconnected
from astra.remote_io import handle
from test_astra_mode2 import Robot, Policy


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.s=Session(Path(self.temp.name)/'run','fixture',{'settings':{'supervisor_backend':'mac'}})
        self.robot=Robot();self.policy=Policy()
        self.c=Controller(self.robot,self.policy,self.s,Settings(100,version=4,astra_pause_handback=True),'drawer','pull drawer')
        self.c.start();self.c.keyboard_toggle()
        self.remote=RemoteSupervisor(self.s)
        self.client=str(uuid.uuid4());self.thread=str(uuid.uuid4())
        self.peer()
        self.remote.bootstrap(threading.Event())

    def peer(self,generation=0):
        return handle(self.s.path,{'operation':'ready','client_id':self.client,'thread_id':self.thread,
                                  'auto_reconnect':True,'reconnect_generation':generation})

    def disconnected(self):
        old=self.s.pending['request_id']
        self.c.pause(RemoteDisconnected('lost SSH'))
        self.remote.wait_for_reconnect(self.c)
        return old

    def recover(self):
        self.remote.reconnect_seen_at=time.monotonic()-3
        return self.remote.recover(self.c)

    def decision(self,kind):
        return {**{k:self.s.pending[k] for k in ('session_id','owner','request_id')},
                'operation':'pause','pause_kind':kind,'reason':'fixture concern','evidence':'fixture observation'}

    def test_reconnect_creates_fresh_request_and_preserves_budget(self):
        self.c.steps=17
        old=self.disconnected()
        self.assertTrue(self.recover())
        self.assertNotEqual(self.s.pending['request_id'],old)
        self.assertEqual(self.s.pending['stage'],'ASTRA_RETRY')
        self.assertFalse(self.s.pending['retry_context']['operator_requested'])
        self.assertEqual(self.c.steps,17)
        self.assertEqual(self.robot.published,[])

    def test_old_response_rejected_after_recovery(self):
        old=self.disconnected();self.recover()
        out=handle(self.s.path,{'operation':'respond','client_id':self.client,'request_id':old,'response':{}})
        self.assertTrue(out['stale'])
        self.assertEqual(self.robot.published,[])

    def test_model_pause_not_auto_retried_even_if_reason_matches(self):
        self.c.pause('Mac supervisor disconnected/stale; no Astra actions released')
        self.assertFalse(self.remote.recover(self.c))
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')

    def test_hardware_fault_not_rearmed_during_reconnect(self):
        self.disconnected();self.robot.fault=True;self.robot.rearm=Mock()
        with self.assertRaisesRegex(RuntimeError,'ESTOP'):self.recover()
        self.robot.rearm.assert_not_called()
        self.assertEqual(self.robot.published,[])

    def test_manual_change_and_limit_stop_automatic_retry(self):
        self.disconnected();self.c.pause('new physical fault')
        self.assertFalse(self.recover())
        self.remote.wait_for_reconnect(self.c);self.remote.reconnect_attempts=3
        self.assertTrue(self.recover())
        self.assertEqual((self.c.mode,self.c.state),('WAM','AUTO'))

    def test_generation_change_invalidates_request_even_if_heartbeat_fresh(self):
        req=self.s.pending
        self.peer(generation=1)
        with self.assertRaises(RemoteDisconnected):self.remote.poll(req)

    def test_planning_and_collision_pause_hand_back_without_recovery_claim(self):
        for kind in ('planning','collision_risk','observation_uncertainty'):
            if self.c.mode=='WAM':self.c.keyboard_toggle()
            self.c.decide(self.decision(kind))
            self.assertEqual((self.c.mode,self.c.state),('WAM','AUTO'))
            self.assertEqual(self.robot.published,[])
            self.assertEqual(self.policy.calls,[])
        rows=[json.loads(l) for l in (self.s.path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['recovery_confirmed'] is False for r in rows if r['kind']=='astra_pause_handback'))

    def test_hardware_and_unspecified_pause_remain_paused(self):
        self.c.decide(self.decision('hardware_fault'))
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')
        self.assertEqual(self.c.mode,'ASTRA')

    def test_opt_out_retains_previous_pause_behavior(self):
        from dataclasses import replace
        self.c.settings=replace(self.c.settings,astra_pause_handback=False)
        self.c.decide(self.decision('collision_risk'))
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')


if __name__=='__main__':unittest.main()
