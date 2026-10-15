"""No network/model calls: bounded reconnect and ambiguous-response semantics."""
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from astra.mac_bridge import Transport, SSHTransportError


def reply(data, code=0):
    return SimpleNamespace(returncode=code, stdout=json.dumps(data), stderr='fixture')


class SSHRetryTests(unittest.TestCase):
    def setUp(self):
        self.transport = Transport('leo3', '/repo', '/session')
        self.transport.stop = Mock(is_set=Mock(return_value=False), wait=Mock(return_value=False))
        self.transport.on_retry = Mock()

    def test_timeout_then_status_success_keeps_identity_and_invalidates(self):
        with patch('astra.mac_bridge.subprocess.run', side_effect=[subprocess.TimeoutExpired('ssh',60),reply({'manifest':{'session_id':'same'}})]) as run:
            self.assertEqual(self.transport.call('status')['manifest']['session_id'], 'same')
        first, second = [json.loads(c.kwargs['input']) for c in run.call_args_list]
        self.assertEqual(first['client_id'], second['client_id'])
        self.assertEqual(second['reconnect_generation'],1)
        self.assertEqual(run.call_args.kwargs['timeout'],60)
        self.assertIn('ConnectTimeout=15',run.call_args.args[0])
        self.transport.on_retry.assert_called_once()

    def test_retries_are_bounded(self):
        with patch('astra.mac_bridge.subprocess.run', side_effect=subprocess.TimeoutExpired('ssh',60)) as run:
            with self.assertRaises(SSHTransportError):self.transport.call('status')
        self.assertEqual(run.call_count,4)

    def test_response_not_retransmitted_after_lost_ack(self):
        with patch('astra.mac_bridge.subprocess.run', side_effect=subprocess.TimeoutExpired('ssh',60)) as run:
            with self.assertRaises(SSHTransportError):self.transport.call('respond', request_id='old')
        self.assertEqual(run.call_count,1)
        self.transport.on_retry.assert_called_once()

    def test_protocol_and_ownership_errors_not_retried(self):
        with patch('astra.mac_bridge.subprocess.run', return_value=reply({'error':'another Mac bridge owns this session'},1)) as run:
            with self.assertRaisesRegex(RuntimeError,'another Mac'):self.transport.call('status')
        self.assertEqual(run.call_count,1)
        self.transport.on_retry.assert_not_called()

    def test_stop_interrupts_backoff(self):
        self.transport.stop.wait.return_value=True
        with patch('astra.mac_bridge.subprocess.run', side_effect=subprocess.TimeoutExpired('ssh',60)) as run:
            with self.assertRaises(InterruptedError):self.transport.call('status')
        self.assertEqual(run.call_count,1)


if __name__=='__main__':unittest.main()
