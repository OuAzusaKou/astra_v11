"""Isolated HTTPS transport. Credentials never enter prompts or result logs."""
import ctypes
import http.client
import json
import os
from pathlib import Path
import sys
import socket
import ssl
import time
import urllib.error
import urllib.request

BASE_URL = 'https://api.quickrouter.us/v1'
HTTP_TIMEOUT_SECONDS = 300.0


class TransportFailure(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('QuickRouter HTTP transport failed: '+details['error_type']
                         +' at '+details['stage']
                         +(f" (HTTP {details['http_status']})" if details.get('http_status') else ''))


def failure_details(error, stage, elapsed):
    cause = error.reason if isinstance(error, urllib.error.URLError) and not isinstance(error, urllib.error.HTTPError) else error
    status = error.code if isinstance(error, urllib.error.HTTPError) else None
    retryable = status in (408, 429, 500, 502, 503, 504) if status else isinstance(cause, (
        http.client.RemoteDisconnected, http.client.IncompleteRead, ConnectionError,
        TimeoutError, socket.gaierror, ssl.SSLEOFError, ssl.SSLZeroReturnError))
    if isinstance(cause, ssl.SSLCertVerificationError):retryable = False
    details = dict(error_type=type(cause).__name__, stage=stage,
                   elapsed_seconds=round(elapsed,3), retryable=retryable)
    if status:details['http_status'] = status
    if isinstance(getattr(cause,'errno',None),int):details['errno'] = cause.errno
    if status:
        value = error.headers.get('Retry-After','') if error.headers else ''
        if value.isdigit():details['retry_after_seconds'] = min(int(value),60)
    return details


def api_key():
    key = os.environ.get('QUICKROUTER_API_KEY', '').strip()
    if key:
        return key
    if os.name != 'nt':
        raise RuntimeError('Set QUICKROUTER_API_KEY on the supervisor host')
    path = Path(os.environ.get('QUICKROUTER_KEY_FILE',
                str(Path(__file__).resolve().parents[2]/'.secrets/quickrouter.dpapi')))
    try:
        encrypted = bytes.fromhex(path.read_text(encoding='ascii').strip())
        from ctypes import wintypes
        class Blob(ctypes.Structure):
            _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
        buffer = (ctypes.c_ubyte*len(encrypted)).from_buffer_copy(encrypted)
        source = Blob(len(encrypted), buffer); output = Blob()
        crypt = ctypes.WinDLL('crypt32', use_last_error=True)
        crypt.CryptUnprotectData.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        crypt.CryptUnprotectData.restype = wintypes.BOOL
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        if not crypt.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
            raise RuntimeError('DPAPI unavailable for this Windows account')
        try:
            key = ctypes.string_at(output.data, output.size).decode('utf-16-le').strip()
        finally:
            ctypes.memset(output.data, 0, output.size); kernel.LocalFree(output.data)
    except (OSError, ValueError) as error:
        raise RuntimeError('QuickRouter encrypted credential unavailable') from None
    if not key:
        raise RuntimeError('QuickRouter credential is empty')
    return key


def request(payload, *, timeout=HTTP_TIMEOUT_SECONDS, proxy=None, diagnostic=None):
    key = api_key()
    handlers = [urllib.request.ProxyHandler({'https':proxy} if proxy else {})]
    # Never forward Authorization to a redirect target.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    handlers.append(NoRedirect())
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(BASE_URL+'/chat/completions',
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8'),
        headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'}, method='POST')
    started=time.monotonic(); stage='open_or_wait_response_headers'
    emit=diagnostic or (lambda event: None)
    emit(dict(event='http_open',stage=stage,request_bytes=len(req.data),proxy_enabled=bool(proxy),
              socket_timeout_seconds=timeout, stream=bool(payload.get('stream'))))
    try:
        with opener.open(req, timeout=timeout) as response:
            stage='read_response_body'
            emit(dict(event='http_headers',stage=stage,http_status=response.status,
                      elapsed_seconds=round(time.monotonic()-started,3)))
            if payload.get('stream'):
                from .chat_stream import read_completion
                stage = 'read_stream_body'
                def stream_event(**fields):
                    emit(dict(fields, elapsed_seconds=round(time.monotonic()-started,3)))
                return read_completion(response, stream_event)
            raw = response.read(16_000_001)
        stage='decode_response'
        if len(raw)>16_000_000:
            raise RuntimeError('QuickRouter response exceeds transport limit')
        data = json.loads(raw)
        if not isinstance(data,dict) or 'error' in data:
            raise RuntimeError('QuickRouter returned an invalid/error response')
        emit(dict(event='http_complete',stage=stage,response_bytes=len(raw),
                  elapsed_seconds=round(time.monotonic()-started,3)))
        return data
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, RuntimeError) as error:
        details=failure_details(error,stage,time.monotonic()-started)
        emit(dict(event='http_failure',**details))
        if isinstance(error,urllib.error.HTTPError):error.close()
        # No arbitrary exception messages, response bodies, headers or keys.
        raise TransportFailure(details) from None


def main():
    if hasattr(sys.stdin,'reconfigure'):sys.stdin.reconfigure(encoding='utf-8')
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    try:
        envelope=json.load(sys.stdin)
        result=request(envelope['payload'], timeout=envelope.get('timeout',HTTP_TIMEOUT_SECONDS), proxy=envelope.get('proxy'),
                       diagnostic=lambda event: print(json.dumps(event),file=sys.stderr,flush=True))
        print(json.dumps({'response':result},ensure_ascii=False))
    except TransportFailure as error:
        print(json.dumps({'error':str(error),'transport_error':error.details}))
        return 1
    except Exception as error:
        print(json.dumps({'error':str(error) if isinstance(error,RuntimeError) else type(error).__name__}))
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
