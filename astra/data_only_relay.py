"""V10 private transport boundary: Codex auth is reused, its agent context is not sent.

Only loopback is bound. No credential is stored or logged. Responses are buffered
and checked before Codex sees them, so no model-selected tool can execute.
"""
import copy
import http.client
import http.server
import json
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid

UPSTREAM = 'https://chatgpt.com/backend-api/codex/responses'


def retryable_transport_error(error):
    if isinstance(error, InterruptedError):
        return False
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429, 500, 502, 503, 504)
    if isinstance(error, urllib.error.URLError):
        return retryable_transport_error(error.reason)
    return isinstance(error, (http.client.IncompleteRead, ConnectionError,
                              TimeoutError, ssl.SSLEOFError, ssl.SSLZeroReturnError))


def response_payload(instructions, messages, output_schema):
    return dict(model='gpt-6-astra', instructions=instructions, input=copy.deepcopy(messages),
                tools=[], tool_choice='none', parallel_tool_calls=False, stream=True, store=False,
                reasoning={'effort': 'medium'},
                text={'format': {'type': 'json_schema', 'name': 'action_chunk',
                                 'strict': True, 'schema': output_schema}})


def check_response(raw):
    """Reject every executable output type, including hidden/nested tool calls."""
    complete = False
    def item(value):
        if not isinstance(value, dict) or value.get('type') not in ('message', 'reasoning'):
            raise ValueError('V10 provider emitted a forbidden output/tool type')
        if value['type'] == 'message':
            if value.get('role') != 'assistant':
                raise ValueError('V10 unexpected output role')
            for block in value.get('content', []):
                if block.get('type') not in ('output_text', 'refusal'):
                    raise ValueError('V10 unexpected message content')
    for block in raw.decode('utf-8').replace('\r\n', '\n').split('\n\n'):
        data = '\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith('data:'))
        if not data or data == '[DONE]':
            continue
        event = json.loads(data)
        kind = event.get('type', '')
        if 'function_call' in kind or 'tool_call' in kind or 'custom_tool' in kind:
            raise ValueError('V10 tool response rejected')
        if 'item' in event:
            item(event['item'])
        if 'response' in event:
            for value in event['response'].get('output', []):
                item(value)
        if kind == 'response.completed':
            complete = event['response'].get('status') == 'completed'
    if not complete:
        raise ValueError('V10 incomplete provider response; no decision released')


class DataOnlyRelay:
    def __init__(self, payload, proxy, cancel, timeout=300, opener=None, log=None):
        self.payload, self.cancel, self.timeout = payload, cancel, min(timeout, 300)
        # Reserve time for the outer CLI to consume the buffered response/failure.
        self.deadline = time.monotonic() + max(0, timeout - 10)
        self.closed = threading.Event()
        self.log = log or (lambda message: None)
        self.retry_delays = (2., 4.)
        self.error = None
        self.lock = threading.Lock()
        self.cached = None
        self.upstream_response = None
        self.opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({'http': proxy, 'https': proxy} if proxy else {}))
        token = uuid.uuid4().hex
        relay = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_error(404)

            def do_POST(self):
                if self.path != '/' + token + '/responses':
                    self.send_error(404); return
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > 64 * 1024 * 1024:
                    self.send_error(413); return
                # Discard the CLI prompt, tools, environment, source hints and history entirely.
                self.rfile.read(size)
                try:
                    with relay.lock:
                        if relay.cancel.is_set():
                            raise InterruptedError('cancelled V10 request')
                        if relay.error:
                            raise ValueError('previous V10 boundary failure')
                        if relay.cached is None:
                            headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
                            for key in ('Authorization', 'ChatGPT-Account-ID', 'OpenAI-Beta', 'originator', 'User-Agent'):
                                if self.headers.get(key):
                                    headers[key] = self.headers[key]
                            req = urllib.request.Request(UPSTREAM, json.dumps(relay.payload, allow_nan=False).encode(), headers)
                            relay.cached = relay.fetch(req)
                        relay.ensure_active()
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Content-Length', str(len(relay.cached)))
                        self.end_headers(); self.wfile.write(relay.cached)
                except Exception as error:
                    relay.error = type(error).__name__ + ': ' + str(error)[:180]
                    self.send_error(502, 'V10 data-only response rejected')
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_port}/{token}'

    def ensure_active(self):
        if self.cancel.is_set() or self.closed.is_set():
            raise InterruptedError('cancelled V10 response')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('V10 transport total deadline exceeded')

    def fetch(self, request):
        """Retry only before a complete validated reply is released, never replay actions."""
        attempts = len(self.retry_delays) + 1
        for attempt in range(attempts):
            self.ensure_active()
            try:
                remaining = self.deadline - time.monotonic()
                with self.opener.open(request, timeout=min(self.timeout, remaining)) as upstream:
                    self.upstream_response = upstream
                    chunks, total = [], 0
                    while True:
                        self.ensure_active()
                        chunk = upstream.read(65536)
                        if not chunk:
                            break
                        chunks.append(chunk); total += len(chunk)
                        if total > 8 * 1024 * 1024:
                            raise ValueError('V10 response exceeds size limit')
                    raw = b''.join(chunks)
                self.ensure_active()
                check_response(raw)
                if attempt:
                    self.log(f'TRANSPORT RECOVERED: V10 attempt={attempt+1}/{attempts}')
                return raw
            except Exception as error:
                self.ensure_active()
                if not retryable_transport_error(error) or attempt + 1 == attempts:
                    raise
                delay = self.retry_delays[attempt]
                if time.monotonic() + delay >= self.deadline:
                    raise
                self.log(f'TRANSPORT RETRY: V10 {type(error).__name__}: {str(error)[:180]}; '
                         f'request {attempt+2}/{attempts} in {delay:g}s; partial reply discarded')
                end = time.monotonic() + delay
                while time.monotonic() < end:
                    self.ensure_active()
                    self.cancel.wait(min(.1, max(0, end-time.monotonic())))
            finally:
                self.upstream_response = None

    def __enter__(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.closed.set()
        self.server.shutdown(); self.server.server_close()
        # Worker threads are daemonized; cancelled results cannot be delivered.
