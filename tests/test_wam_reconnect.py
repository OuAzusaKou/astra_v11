"""Offline disconnect recovery; mocked inference sockets and robot publisher."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from astra.control import Controller, Settings
from astra.policy import Connection, PolicyTransportError
from astra.session import Session
from policy_server.msgpack_numpy import packb


class ConnectionTests(unittest.TestCase):
    def test_idle_socket_replaced_before_request(self):
        conn = Connection('fixture', 1, threading.Event())
        old, fresh = Mock(), Mock()
        conn.sock = old
        conn.last_response_at = time.monotonic() - 31
        response = {'request_id': 'new', 'success': True}
        payload = packb(response)
        fresh.recv.side_effect = [len(payload).to_bytes(4, 'big'), payload]
        with patch('astra.policy.socket.create_connection', return_value=fresh), patch('astra.policy.select.select', return_value=([fresh], [], [])):
            self.assertEqual(conn.send({'request_id': 'new'}), response)
        old.close.assert_called_once()
        old.sendall.assert_not_called()
        fresh.sendall.assert_called_once()

    def test_disconnect_is_retryable_but_wrong_identity_is_not(self):
        for payload, error in ((b'', PolicyTransportError), (packb({'request_id': 'old'}), ValueError)):
            conn = Connection('fixture', 1, threading.Event())
            sock = Mock(); conn.sock = sock
            sock.recv.side_effect = [len(payload).to_bytes(4, 'big'), payload] if payload else [b'']
            with patch('astra.policy.select.select', return_value=([sock], [], [])):
                with self.assertRaises(error):conn.send({'request_id': 'new'})
            self.assertIsNone(conn.sock)
            sock.close.assert_called_once()

    def test_stop_is_not_a_retryable_transport_failure(self):
        stop=threading.Event();stop.set()
        conn=Connection('fixture',1,stop)
        with self.assertRaises(InterruptedError):conn.send({'request_id':'new'})


class RecoveryTests(unittest.TestCase):
    def make(self, version=4):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        session=Session(Path(temp.name)/'run','fixture',{})
        robot=Mock();policy=Mock()
        counter=[0]
        def observe():
            counter[0]+=1
            return {**{key:np.zeros((8,8,3),np.uint8) for key in ('image','image_wrist_left','image_wrist_right')},
                    'timestamp':float(counter[0]), 'left_ee_pose':np.zeros((1,6)), 'right_ee_pose':np.zeros((1,6)),
                    'left_hand_joint':np.zeros((1,20)), 'right_hand_joint':np.zeros((1,20))}
        robot.observe.side_effect=observe
        c=Controller(robot,policy,session,Settings(400,version=version,astra_pause_handback=True),'task','subtask')
        c.l2='subtask'
        return c,robot,policy

    def test_both_handbacks_retry_with_new_observation_without_publishing(self):
        for version in (4,8):
            for source in ('keyboard','fallback'):
                with self.subTest(version=version,source=source):
                    c,robot,policy=self.make(version)
                    c.mode='ASTRA';c.state='WAITING'
                    if source=='keyboard':c.keyboard_toggle()
                    else:self.assertTrue(c.supervisor_stopped('model stopped'))
                    policy.infer.side_effect=[PolicyTransportError('server disconnected'),(np.zeros((15,52)),{})]
                    c.infer(review=False)
                    self.assertEqual(c.state,'AUTO');self.assertEqual(c.mode,'WAM')
                    seen=[call.args[0]['timestamp'] for call in policy.infer.call_args_list]
                    self.assertEqual(seen,[1.,2.])
                    self.assertEqual(c.wam_active['observation']['state']['timestamp'],2.)
                    self.assertEqual(len(c.wam_history),1)
                    robot.publish.assert_not_called()

    def test_exhaustion_is_bounded_and_validation_is_not_retried(self):
        for error, count in ((PolicyTransportError('disconnected'),3),(ValueError('bad action'),1)):
            c,robot,policy=self.make();policy.infer.side_effect=error
            with self.assertRaises(type(error)):c.infer(review=False)
            self.assertEqual(policy.infer.call_count,count)
            self.assertEqual(len(c.queue),0);robot.publish.assert_not_called()

    def test_pending_switch_cancels_recovery(self):
        c,robot,policy=self.make();pending=threading.Event()
        c.switch_pending=pending.is_set
        def fail(*args):
            pending.set();raise PolicyTransportError('disconnected')
        policy.infer.side_effect=fail
        c.infer(review=False)
        self.assertEqual(policy.infer.call_count,1)
        self.assertEqual(len(c.queue),0);self.assertEqual(c.state,'AUTO')
        robot.publish.assert_not_called()

    def test_hardware_fault_during_recovery_does_not_retry_inference(self):
        c,robot,policy=self.make()
        robot.check.side_effect=[None,RuntimeError('ESTOP')]
        policy.infer.side_effect=PolicyTransportError('disconnected')
        with self.assertRaisesRegex(RuntimeError,'ESTOP'):c.infer(review=False)
        self.assertEqual(policy.infer.call_count,1);robot.publish.assert_not_called()


if __name__=='__main__':unittest.main()
