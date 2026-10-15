"""Mock upstream only: retry/cancellation cannot publish robot commands."""
import http.client
import io
import json
import ssl
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock

from astra.data_only_relay import DataOnlyRelay, check_response


GOOD = b'data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"{\\"action_chunk\\":[]}"}]}]}}\n\n'


class BrokenStream(io.BytesIO):
    def read(self, size=-1):
        raise http.client.IncompleteRead(b'partial invalid action data')


class RetryTests(unittest.TestCase):
    def relay(self, opener, cancel=None, log=None):
        relay = DataOnlyRelay({'tools': []}, None, cancel or threading.Event(),
                              opener=opener, log=log)
        relay.retry_delays = (0., 0.)
        self.addCleanup(relay.server.server_close)
        return relay

    def test_partial_stream_discarded_then_success_cached_for_cli_retry(self):
        broken = BrokenStream(); opener = Mock()
        opener.open.side_effect = [broken, io.BytesIO(GOOD)]
        logs = []; relay = self.relay(opener, log=logs.append)
        with relay:
            client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(relay.url+'/responses', b'{"private":"ignored"}')
            for _ in range(2):
                with client.open(request, timeout=3) as response:
                    self.assertEqual(response.read(), GOOD)
        self.assertTrue(broken.closed)
        self.assertEqual(opener.open.call_count, 2)
        self.assertIsNone(relay.error)
        self.assertEqual(len(logs), 2)
        for call in opener.open.call_args_list:
            self.assertEqual(json.loads(call.args[0].data), {'tools': []})

    def test_transient_errors_retry_and_stop_at_three(self):
        for error in (http.client.RemoteDisconnected('closed'), TimeoutError('write timeout'),
                      urllib.error.URLError(ssl.SSLEOFError(8, 'unexpected EOF')),
                      urllib.error.HTTPError('https://test', 503, 'unavailable', {}, None)):
            opener = Mock(); opener.open.side_effect = error
            relay = self.relay(opener)
            with self.assertRaises(type(error)):
                relay.fetch(urllib.request.Request('https://test'))
            self.assertEqual(opener.open.call_count, 3)
            self.assertIsNone(relay.cached)

    def test_schema_tool_and_auth_errors_do_not_retry(self):
        for raw in (b'data: {"type":"response.function_call_arguments.delta"}\n\n',
                    b'data: {"type":"response.created"}\n\n', b'data: invalid-json\n\n'):
            opener = Mock(); opener.open.return_value = io.BytesIO(raw)
            relay = self.relay(opener)
            with self.assertRaises(ValueError): relay.fetch(urllib.request.Request('https://test'))
            self.assertEqual(opener.open.call_count, 1)
        opener = Mock(); opener.open.side_effect = urllib.error.HTTPError('https://test',401,'auth',{},None)
        with self.assertRaises(urllib.error.HTTPError):
            self.relay(opener).fetch(urllib.request.Request('https://test'))
        self.assertEqual(opener.open.call_count, 1)

    def test_cancel_during_backoff_prevents_next_attempt(self):
        cancel = threading.Event(); opener = Mock()
        opener.open.side_effect = http.client.RemoteDisconnected('closed')
        relay = self.relay(opener, cancel, lambda _: cancel.set())
        relay.retry_delays = (2., 4.)
        with self.assertRaises(InterruptedError): relay.fetch(urllib.request.Request('https://test'))
        self.assertEqual(opener.open.call_count, 1)

    def test_cancel_on_response_prevents_delivery(self):
        cancel = threading.Event()
        class CancelStream(io.BytesIO):
            def read(self, size=-1):
                cancel.set()
                return super().read(size)
        opener = Mock(); opener.open.return_value = CancelStream(GOOD)
        relay = self.relay(opener, cancel)
        with self.assertRaises(InterruptedError): relay.fetch(urllib.request.Request('https://test'))
        self.assertEqual(opener.open.call_count, 1)
        self.assertIsNone(relay.cached)

    def test_total_deadline_and_closed_relay_prevent_new_request(self):
        opener = Mock(); relay = self.relay(opener)
        relay.deadline = time.monotonic()-1
        with self.assertRaises(TimeoutError): relay.fetch(urllib.request.Request('https://test'))
        relay.closed.set()
        with self.assertRaises(InterruptedError): relay.fetch(urllib.request.Request('https://test'))
        opener.open.assert_not_called()


if __name__ == '__main__':
    unittest.main()
