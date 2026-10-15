"""Offline QuickRouter contract tests. No model, ROS or SSH requests."""
import json
import http.client
import ssl
import urllib.error
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

from astra.api_supervisor import QuickRouterSupervisor, parse_decision, image_block, resolve_model
from astra.supervisor_models import model_info
from astra.quickrouter_transport import TransportFailure, failure_details, request as http_request


def completion(value,finish='stop',**extra):
    return {'choices':[{'finish_reason':finish,'message':{'role':'assistant','content':json.dumps(value),**extra}}]}


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.bridge=QuickRouterSupervisor(SimpleNamespace(path=self.root,owner='test'),model='gemini',
            readonly_remote=('leo3','/home/liberai/Liberai_Deploy_astra'))
        self.bridge.active_identity=dict(session_id='s',owner='test',request_id='r')
        self.decision=dict(self.bridge.active_identity,operation='observe',reason='test',evidence='test')

    def call(self,images=(),**kwargs):
        return self.bridge._call('current robot request',images,threading.Event(),'turn',**kwargs)

    def test_aliases_are_exact_and_unknown_rejected(self):
        self.assertEqual(resolve_model('opus5.5'),'claude-opus-5-5')
        self.assertEqual(resolve_model('gemini'),'gemini-3.8-flash')
        self.assertEqual(resolve_model('qwen3.8-max'),'qwen3.8-max')
        self.assertEqual(resolve_model('glm5.3-flash'),'glm-5.3-flash')
        self.assertEqual(model_info('astra')['provider'],'codex')
        with self.assertRaises(ValueError):resolve_model('gemini-other')

    def test_images_all_sent_with_correct_mime_and_history(self):
        images=[]
        for i in range(7):
            p=self.root/f'{i}.jpg';Image.new('RGB',(5,5),(i,30,70)).save(p,format='PNG');images.append(p)
        with patch.object(self.bridge,'_http',return_value=completion(self.decision)) as send:
            self.assertEqual(self.call(images),self.decision)
            payload=send.call_args.args[0]
            blocks=payload['messages'][0]['content']
            self.assertEqual(len(blocks),8)
            self.assertTrue(all(b['image_url']['url'].startswith('data:image/png;base64,') for b in blocks[1:]))
            self.assertEqual(payload['model'],'gemini-3.8-flash')
            self.call([])
            self.assertEqual(len(send.call_args.args[0]['messages']),4)  # response appended in-place
        self.assertEqual(len(self.bridge.history),4)
        self.assertTrue((self.bridge.path/'api_history.json').is_file())

    def test_tools_use_allowlist_and_roundtrip(self):
        call={'id':'tool1','type':'function','function':{'name':'read_file','arguments':json.dumps(
            {'path':'/home/liberai/Liberai_Deploy_astra/envs/leo_wuji_env_infer.py','line_count':12})}}
        tool_result={'path':'/home/liberai/Liberai_Deploy_astra/envs/leo_wuji_env_infer.py','text':'source'}
        with patch.object(self.bridge,'_http',side_effect=[completion(None,'tool_calls',tool_calls=[call]),completion(self.decision)]) as send, \
             patch('astra.api_supervisor.query',return_value=tool_result) as query:
            self.call()
        self.assertTrue(self.bridge.remote_read_verified)
        query.assert_called_once()
        tools=[t['function']['name'] for t in send.call_args.args[0]['tools']]
        self.assertEqual(set(tools),{'read_file','list_directory','search_text','ros_query'})
        self.assertEqual(self.bridge.history[2]['role'],'tool')
        with patch('astra.api_supervisor.query') as query, self.assertRaises(ValueError):
            self.bridge._tool('publish',{},threading.Event())
        query.assert_not_called()

    def test_wrong_identity_and_truncation_do_not_commit_history(self):
        for response in (completion(dict(self.decision,request_id='old')),completion(self.decision,'length')):
            with patch.object(self.bridge,'_http',return_value=response),self.assertRaises(ValueError):self.call()
            self.assertFalse(self.bridge.history)

    def test_cancellation_discards_returned_response(self):
        cancel=threading.Event()
        def reply(*args):cancel.set();return completion(self.decision)
        with patch.object(self.bridge,'_http',side_effect=reply),self.assertRaises(InterruptedError):
            self.bridge._call('test',[],cancel,'cancelled')
        self.assertFalse(self.bridge.history)
        self.assertFalse((self.bridge.path/'cancelled.json').exists())

    def test_strict_json_and_nan(self):
        self.assertEqual(parse_decision('```json\n{"operation":"observe"}\n```'),{'operation':'observe'})
        for value in ('[]','{"x":NaN}','text {"x":1}',None):
            with self.assertRaises((ValueError,TypeError)):parse_decision(value)

    def test_bootstrap_cannot_skip_tool_or_vision(self):
        with patch.object(self.bridge,'_call',return_value={'ready':True,'visual_code':'wrong'}),self.assertRaises(RuntimeError):
            self.bridge.bootstrap(threading.Event())
        def fake(prompt,images,*args,**kwargs):
            self.assertEqual(len(images),1)
            return {'ready':False}
        self.bridge.remote_read_verified=True
        with patch.object(self.bridge,'_call',side_effect=fake),self.assertRaises(RuntimeError):
            self.bridge.bootstrap(threading.Event())

    def test_http_cancel_kills_actual_child(self):
        real=subprocess.Popen;processes=[]
        def child(*args,**kwargs):
            p=real([sys.executable,'-c','import sys,time;sys.stdin.read();time.sleep(60)'],**kwargs)
            processes.append(p);return p
        cancel=threading.Event();timer=threading.Timer(.3,cancel.set);timer.start()
        started=time.monotonic()
        try:
            with patch('astra.api_supervisor.subprocess.Popen',side_effect=child),self.assertRaises(InterruptedError):
                self.bridge._http({'model':'fixture'},cancel,time.monotonic()+10)
        finally:timer.cancel()
        self.assertLess(time.monotonic()-started,3)
        self.assertIsNotNone(processes[0].poll())

    def test_http_wait_is_five_minutes_bounded_by_remaining_turn(self):
        for remaining,expected in ((600,300),(41,41)):
            with patch('astra.api_supervisor.time.monotonic',return_value=100), \
                 patch('astra.api_supervisor.subprocess.Popen') as spawn:
                process=spawn.return_value
                process.communicate.return_value=(json.dumps({'response':{'ok':True}}),'')
                process.returncode=0;process.poll.return_value=0
                self.bridge._http_once({},threading.Event(),100+remaining,'fixture',1)
            envelope=json.loads(process.communicate.call_args_list[0].args[0])
            self.assertEqual(envelope['timeout'],expected)
        log=(self.bridge.path/'api_transport.jsonl').read_text(encoding='utf-8')
        self.assertIn('http_wait_budget',log)
        self.assertIn('socket_timeout_seconds',log)

    def test_transport_default_passes_300_seconds_to_socket(self):
        with patch('astra.quickrouter_transport.api_key',return_value='fixture'), \
             patch('astra.quickrouter_transport.urllib.request.build_opener') as build:
            response=build.return_value.open.return_value.__enter__.return_value
            response.status=200;response.read.return_value=b'{"ok":true}'
            self.assertEqual(http_request({}),{'ok':True})
            self.assertEqual(build.return_value.open.call_args.kwargs['timeout'],300)

    def test_metadata_is_api_conversation_not_codex(self):
        meta=json.loads((self.root/'supervisor.json').read_text())
        self.assertEqual(meta['conversation_kind'],'local_api_history')
        self.assertEqual(meta['model'],'gemini-3.8-flash')
        self.assertIsNone(meta['resume_command'])

    def test_transient_retry_preserves_payload_and_logs_no_content(self):
        failure=TransportFailure(failure_details(http.client.RemoteDisconnected('secret upstream text'), 'open_or_wait_response_headers',60))
        cancel=threading.Event();payload={'model':'fixture','messages':[{'content':'PRIVATE PROMPT'}]}
        with patch.object(self.bridge,'_http_once',side_effect=[failure,completion(self.decision)]) as send, \
             patch.object(cancel,'wait',return_value=False):
            self.assertEqual(self.bridge._http(payload,cancel,time.monotonic()+20),completion(self.decision))
        self.assertEqual(send.call_count,2)
        self.assertTrue(all(c.args[0] is payload for c in send.call_args_list))
        log=(self.bridge.path/'api_transport.jsonl').read_text(encoding='utf-8')
        self.assertIn('RemoteDisconnected',log);self.assertIn('retry_wait',log)
        self.assertNotIn('PRIVATE PROMPT',log);self.assertNotIn('secret upstream text',log)

    def test_retry_bound_and_nonretryable_failure(self):
        for retryable,expected in ((True,3),(False,1)):
            cancel=threading.Event()
            error=TransportFailure(dict(error_type='fixture',stage='open_or_wait_response_headers',retryable=retryable))
            with patch.object(self.bridge,'_http_once',side_effect=error) as send,patch.object(cancel,'wait',return_value=False),self.assertRaises(TransportFailure):
                self.bridge._http({},cancel,time.monotonic()+30)
            self.assertEqual(send.call_count,expected)

    def test_retry_wait_cancel_and_deadline_block_next_attempt(self):
        failure=TransportFailure(dict(error_type='RemoteDisconnected',stage='open_or_wait_response_headers',retryable=True))
        cancel=threading.Event()
        with patch.object(self.bridge,'_http_once',side_effect=failure) as send,patch.object(cancel,'wait',return_value=True),self.assertRaises(InterruptedError):
            self.bridge._http({},cancel,time.monotonic()+30)
        self.assertEqual(send.call_count,1)
        with patch.object(self.bridge,'_http_once',side_effect=failure) as send,self.assertRaises(TimeoutError):
            self.bridge._http({},threading.Event(),time.monotonic()+1)
        self.assertEqual(send.call_count,1)

    def test_transport_classifies_eof_cert_and_http_errors(self):
        for error,want in ((http.client.RemoteDisconnected(),True),(urllib.error.URLError(ssl.SSLEOFError()),True),
                           (urllib.error.URLError(ssl.SSLCertVerificationError()),False),(TimeoutError(),True)):
            self.assertEqual(failure_details(error,'open',1)['retryable'],want)
        for code,want in ((401,False),(403,False),(400,False),(429,True),(502,True)):
            error=urllib.error.HTTPError('unused',code,'private',{},None)
            try:self.assertEqual(failure_details(error,'open',1)['retryable'],want)
            finally:error.close()

    def test_transport_error_carries_stage_without_raw_exception(self):
        events=[]
        with patch('astra.quickrouter_transport.api_key',return_value='DUMMY_SECRET'), \
             patch('astra.quickrouter_transport.urllib.request.build_opener') as build:
            build.return_value.open.side_effect=http.client.RemoteDisconnected('DUMMY_SECRET')
            with self.assertRaises(TransportFailure) as caught:http_request({'model':'fixture'},diagnostic=events.append)
        self.assertEqual(caught.exception.details['stage'],'open_or_wait_response_headers')
        self.assertTrue(caught.exception.details['retryable'])
        self.assertNotIn('DUMMY_SECRET',json.dumps(events)+str(caught.exception))

    def test_partial_body_failure_is_retryable_and_no_partial_response_released(self):
        events=[]
        with patch('astra.quickrouter_transport.api_key',return_value='fixture'), \
             patch('astra.quickrouter_transport.urllib.request.build_opener') as build:
            response=build.return_value.open.return_value.__enter__.return_value
            response.status=200
            response.read.side_effect=http.client.IncompleteRead(b'PRIVATE PARTIAL BODY',100)
            with self.assertRaises(TransportFailure) as caught:http_request({'model':'fixture'},diagnostic=events.append)
        self.assertEqual(caught.exception.details['stage'],'read_response_body')
        self.assertTrue(caught.exception.details['retryable'])
        self.assertNotIn('PRIVATE PARTIAL BODY',json.dumps(events)+str(caught.exception))

    def test_real_v4_v8_prompt_and_images_use_shared_contract(self):
        image=self.root/'frame.png';Image.new('RGB',(4,4)).save(image)
        for version in (4,8):
            request=dict(self.bridge.active_identity,version=version,stage='REVIEW_CANDIDATES' if version==8 else 'ASTRA_TAKEOVER',
                observation={'images':{'chest':{'path':str(image)}}},wam_context={'trajectories':[]},
                imagination={'candidate_set_id':'set','images':[{'path':str(image),'candidate_id':'candidate','label':'left_plus_x'}]} if version==8 else None)
            self.bridge.worker=None
            with patch.object(self.bridge,'_http',return_value=completion(self.decision)):
                self.bridge.start(request);self.bridge.worker.join(3)
            self.assertIsNone(self.bridge.error)
            saved=json.loads((self.bridge.path/'r.input.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved['images']),2 if version==8 else 1)
            if version==8:self.assertIn('operation=propose_candidates',saved['prompt'])


if __name__=='__main__':unittest.main()
