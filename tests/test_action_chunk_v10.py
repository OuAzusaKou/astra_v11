"""No robot, no provider calls: V10 boundaries, history, routing and fake execution."""
import copy
import io
import json
import tempfile
import threading
import unittest
import urllib.request
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from astra.action_chunk import CONTRACT, public_input, schema, validate, to_decision
from astra.chunk_supervisor import ChunkApiSupervisor, ChunkCodexSupervisor
from astra.control import Controller, Settings
from astra.data_only_relay import DataOnlyRelay, response_payload, check_response
from astra.session import Session
from astra.supervisor import CodexSupervisor
from astra.switch_modes import resolve_modes
if os.name == 'nt':
    # Import only the shared fake robot fixtures; no runner or file lock is used.
    sys.modules.setdefault('fcntl', Mock())
from test_astra_mode2 import Robot, Policy
from wam_direct.start_mac_vlm import leo_command


def reply(request, status='action'):
    return dict(action_chunk=[[.01]*52 for _ in range(3)] if status == 'action' else [])


def sse(item=None):
    item = item or dict(type='message', role='assistant', content=[dict(type='output_text', text='{"ready":true}')])
    return ('data: '+json.dumps(dict(type='response.output_item.added', item=item))+'\n\n'
            +'data: '+json.dumps(dict(type='response.completed', response=dict(status='completed', output=[item])))+'\n\n').encode()


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.session = Session(Path(self.temp.name)/'run', 'v9-fixture', {})
        self.robot, self.policy = Robot(), Policy()
        self.controller = Controller(self.robot, self.policy, self.session,
                                     Settings(100, version=10, no_step_limit=True, astra_pause_handback=True),
                                     'notebook and pen', 'open notebook')
        self.controller.start(); self.controller.keyboard_toggle()
        self.request = self.session.pending

    def envelope(self, value):
        return {**{k:self.request[k] for k in ('owner','session_id','request_id')},
                'operation':'v10_action_chunk','chunk_reply':value}

    def test_chunk_is_executed_exactly_then_observed(self):
        value=reply(self.request)
        self.controller.decide(self.envelope(value))
        self.assertEqual(self.robot.published[0].tolist(),value['action_chunk'])
        self.assertEqual(self.controller.mode,'ASTRA')
        self.assertEqual(self.controller.state,'WAITING')
        self.assertNotEqual(self.session.pending['request_id'],self.request['request_id'])
        self.assertEqual(self.policy.calls,[])

    def test_invalid_rows_and_extra_fields_never_publish(self):
        bad=[]
        for v in (float('nan'),float('inf'),True,'0.01'):
            r=reply(self.request);r['action_chunk'][0][0]=v;bad.append(r)
        for rows in (None,[[0]*51],[[0]*53],[[0]*52]*31):
            r=reply(self.request);r['action_chunk']=rows;bad.append(r)
        r=reply(self.request);r['execute_steps']=1;bad.append(r)
        r=reply(self.request);r['request_id']='old';bad.append(r)
        for key in ('reason', 'evidence', 'status', 'schema_version', 'stop_kind'):
            r=reply(self.request);r[key]='unexpected';bad.append(r)
        for r in bad:
            with self.assertRaises(ValueError):self.controller.decide(self.envelope(r))
        self.assertEqual(self.robot.published,[])

    def test_old_controller_interface_not_accepted(self):
        d=to_decision(reply(self.request),self.request)
        with self.assertRaises(ValueError):self.controller.decide(d)
        self.assertEqual(self.robot.published,[])

    def test_wrong_supervisor_route_fails_before_loading_source(self):
        legacy=object.__new__(CodexSupervisor)
        with self.assertRaisesRegex(ValueError,'data-only'):
            legacy.start(self.request)

    def test_hardware_and_trajectory_checks_still_block(self):
        self.robot.reject=True
        self.controller.decide(self.envelope(reply(self.request)))
        self.assertEqual(self.robot.published,[])
        self.assertEqual(self.controller.state,'PAUSED_FOR_OPERATOR')

    def test_stop_fallback_and_hardware_latch(self):
        self.robot.fault = True
        with self.assertRaisesRegex(ValueError, 'no active review'):
            self.controller.decide(self.envelope(reply(self.request,'stop')))
        self.assertEqual(self.controller.mode,'ASTRA')
        self.assertFalse(self.controller.supervisor_stopped('timeout'))
        self.assertEqual(self.robot.published, [])

    def test_request_identity_is_enforced_by_host_envelope(self):
        value = self.envelope(reply(self.request))
        value['request_id'] = 'stale-request'
        with self.assertRaises(ValueError):
            self.controller.decide(value)
        self.assertEqual(self.robot.published, [])

    def test_nonhardware_stop_returns_wam_only_in_code(self):
        r=reply(self.request,'stop')
        self.controller.decide(self.envelope(r))
        self.assertEqual(self.controller.mode,'WAM');self.assertEqual(self.robot.published,[])

    def test_manual_t_cancels_late_chunk(self):
        r=reply(self.request);self.controller.keyboard_toggle()
        with self.assertRaises(ValueError):self.controller.decide(self.envelope(r))
        self.assertEqual(self.robot.published,[])

    def test_public_context_is_allowlisted_and_has_demo(self):
        req=copy.deepcopy(self.request)
        req.update(source_code='PRIVATE_CODE_MARKER',control_config={'code':'PRIVATE_CODE_MARKER'})
        req['observation']['state_path']='PRIVATE_CODE_MARKER'
        req['reference_video']={'source_path':'PRIVATE_CODE_MARKER','images':[
            {'path':'demo.jpg','frame_index':5,'time_seconds':.5}]}
        context,images=public_input(req)
        self.assertNotIn('PRIVATE_CODE_MARKER',json.dumps(context))
        self.assertNotIn('state_path',json.dumps(context))
        self.assertEqual(context['L2'],'open notebook');self.assertEqual(len(images),4)
        self.assertEqual(context['demonstration'][0]['frame_index'],5)

    def test_v10_launcher_manual_and_unlimited_v4_unchanged(self):
        self.assertEqual(resolve_modes(10,'manual','manual'),('manual','manual'))
        self.assertIsNone(Settings(1,version=10,no_step_limit=True).budget)
        cmd=leo_command('fixture','Task: x Subtask: y',[],version=10,takeover_mode='manual',
                        l2_switch_mode='manual',no_step_limit=True,astra_model='gemini')
        self.assertIn('version10_action_chunk.json',cmd);self.assertIn('--no-step-limit',cmd)
        self.assertNotIn('--dynamics-view',cmd)
        self.assertEqual(Settings(100,version=4).budget,300)

    def test_api_tools_absent_history_retained_and_tool_reply_rejected(self):
        session=SimpleNamespace(path=Path(self.temp.name)/'api',owner='fixture')
        session.path.mkdir()
        bridge=ChunkApiSupervisor(session,model='gemini')
        bridge._http=Mock(return_value={'choices':[{'finish_reason':'stop','message':{
            'role':'assistant','content':'{"action_chunk":[]}'}}]})
        bridge.bootstrap(threading.Event())
        self.assertEqual(len(bridge.data_history),2)
        sent=bridge._http.call_args.args[0]
        self.assertEqual(sent['response_format']['json_schema']['schema']['properties']['action_chunk']['maxItems'], 0)
        self.assertEqual(schema()['properties']['action_chunk']['maxItems'], 30)
        self.assertEqual(sent['tools'],[]);self.assertEqual(sent['tool_choice'],'none')
        self.assertNotIn('read_file',json.dumps(sent))
        bridge._http.return_value={'choices':[{'finish_reason':'stop','message':{
            'role':'assistant','content':'{}','tool_calls':[{'function':{'name':'read_file'}}]}}]}
        with self.assertRaises(ValueError):bridge.exchange([],schema(),threading.Event(),'bad')
        self.assertEqual(len(bridge.data_history),2)

    def test_outgoing_history_keeps_two_turns_for_both_providers(self):
        history = []
        for i in range(5):
            history.extend([
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'observation-'+str(i)},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,image-'+str(i)}}]},
                {'role': 'assistant', 'content': json.dumps({'action_chunk': [[i]*52]})}])
        current = [{'type': 'text', 'text': 'current observation'}]
        session = SimpleNamespace(path=Path(self.temp.name)/'history', owner='fixture')
        session.path.mkdir()
        api = ChunkApiSupervisor(session, model='gemini')
        api._http = Mock(return_value={'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{"action_chunk":[]}'}}]})
        codex = object.__new__(ChunkCodexSupervisor)
        codex.path, codex.proxy, codex.timeout = session.path, None, 600
        for count in (0, 1, 2, 5):
            with self.subTest(completed_turns=count):
                original = copy.deepcopy(history[:count*2])
                api.data_history = copy.deepcopy(original)
                codex.data_history = copy.deepcopy(original)
                api.exchange(current, schema(), threading.Event(), 'api-history')
                with patch('astra.data_only_relay.DataOnlyRelay') as relay, \
                     patch.object(CodexSupervisor, '_call', return_value={'action_chunk': []}):
                    relay.return_value.__enter__.return_value.url = 'http://fixture'
                    codex.exchange(current, schema(), threading.Event(), 'codex-history')
                    sent = relay.call_args.args[0]['input']
                api_sent = api._http.call_args.args[0]['messages'][1:]
                expected_indices = list(range(max(0, count-2), count))
                self.assertEqual(len(sent), 2*len(expected_indices)+1)
                self.assertEqual(len(api_sent), len(sent))
                for index, turn in enumerate(expected_indices):
                    self.assertEqual(sent[2*index]['content'][0]['text'], 'observation-'+str(turn))
                    self.assertEqual(sent[2*index]['content'][1]['image_url'], 'data:image/png;base64,image-'+str(turn))
                    self.assertEqual(json.loads(sent[2*index+1]['content'][0]['text'])['action_chunk'], [[turn]*52])
                    self.assertEqual(api_sent[2*index], original[2*turn])
                    self.assertEqual(api_sent[2*index+1], original[2*turn+1])
                self.assertEqual(sent[-1]['content'][0]['text'], 'current observation')
                self.assertEqual(api_sent[-1]['content'], current)
                self.assertEqual(api.data_history, original)
                self.assertEqual(codex.data_history, original)


    def test_demo_frames_only_in_current_request_for_both_providers(self):
        public = {'L3': 'task', 'L2': 'subtask', 'measured_state': {'left_ee_pose': [[1, 2, 3]]},
                  'image_order': ['CURRENT image', 'DEMONSTRATION frame 0', 'HISTORICAL trajectory image'],
                  'demonstration': [{'frame_index': 0, 'time_seconds': 0}],
                  'policy_history': [{'actions': [[.1]*52]}]}
        def content(prefix):
            return [{'type': 'text', 'text': json.dumps(public)}] + [
                {'type': 'image_url', 'image_url': {'url': prefix+'-'+name}}
                for name in ('observation', 'demo', 'policy-observation')]
        original = []
        for name in ('old', 'previous', 'recent'):
            original.extend([{'role': 'user', 'content': content(name)},
                             {'role': 'assistant', 'content': '{"action_chunk": []}'}])
        current = content('current')
        session = SimpleNamespace(path=Path(self.temp.name)/'demo-history', owner='fixture')
        session.path.mkdir()
        api = ChunkApiSupervisor(session, model='gemini')
        api.data_history = copy.deepcopy(original)
        api._http = Mock(return_value={'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{"action_chunk":[]}'}}]})
        api.exchange(current, schema(), threading.Event(), 'demo-api')
        api_sent = api._http.call_args.args[0]['messages'][1:]
        codex = object.__new__(ChunkCodexSupervisor)
        codex.path, codex.proxy, codex.timeout = session.path, None, 600
        codex.data_history = copy.deepcopy(original)
        with patch('astra.data_only_relay.DataOnlyRelay') as relay, \
             patch.object(CodexSupervisor, '_call', return_value={'action_chunk': []}):
            relay.return_value.__enter__.return_value.url = 'http://fixture'
            codex.exchange(current, schema(), threading.Event(), 'demo-codex')
            codex_sent = relay.call_args.args[0]['input']
        for sent in (api_sent, codex_sent):
            self.assertEqual(len(sent), 5)
            for i, name in ((0, 'previous'), (2, 'recent')):
                blocks = sent[i]['content']
                metadata = json.loads(blocks[0]['text'])
                expected = copy.deepcopy(public)
                expected.update(image_order=['CURRENT image', 'HISTORICAL trajectory image'], demonstration=[])
                self.assertEqual(metadata, expected)
                urls = [b['image_url']['url'] if isinstance(b['image_url'], dict) else b['image_url'] for b in blocks[1:]]
                self.assertEqual(urls, [name+'-observation', name+'-policy-observation'])
            self.assertEqual(json.loads(sent[-1]['content'][0]['text']), public)
            self.assertEqual(len(sent[-1]['content']), 4)
            self.assertNotIn('previous-demo', json.dumps(sent))
            self.assertNotIn('recent-demo', json.dumps(sent))
            self.assertIn('current-demo', json.dumps(sent))
        self.assertEqual(api.data_history, original)
        self.assertEqual(codex.data_history, original)
        self.assertEqual(current, content('current'))

    def test_mismatched_historical_image_labels_fail_closed(self):
        bridge = object.__new__(ChunkApiSupervisor)
        bridge.data_history = [{'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
            'image_order': ['CURRENT image', 'DEMONSTRATION frame 0']})},
            {'type': 'image_url', 'image_url': {'url': 'one-image'}}]}]
        with self.assertRaisesRegex(ValueError, 'labels do not match'):
            bridge.recent_history()


    def test_wam_images_deduplicated_without_losing_identity_or_state(self):
        def observation(rid, extra_label, extra_image):
            public = {'request_id': rid, 'measured_state': {'timestamp': rid},
                      'policy_history': [{'trajectory_id': 'shared', 'row_status': [rid], 'actions': [[1]*52]}],
                      'image_order': ['CURRENT image', 'HISTORICAL shared image', extra_label]}
            blocks = [{'type': 'text', 'text': json.dumps(public)}]
            blocks += [{'type': 'image_url', 'image_url': {'url': image}}
                       for image in (rid+'-live', 'shared-bytes', extra_image)]
            return blocks
        old = observation('old', 'HISTORICAL other-trajectory image', 'shared-bytes')
        recent = observation('recent', 'HISTORICAL changing image', 'earlier-bytes')
        current = observation('current', 'HISTORICAL changing image', 'different-bytes')
        original = [{'role': 'user', 'content': old}, {'role': 'assistant', 'content': '{"action_chunk":[]}'},
                    {'role': 'user', 'content': recent}, {'role': 'assistant', 'content': '{"action_chunk":[]}'}]
        session = SimpleNamespace(path=Path(self.temp.name)/'wam-images', owner='fixture')
        session.path.mkdir()
        api = ChunkApiSupervisor(session, model='gemini')
        api.data_history = copy.deepcopy(original)
        api._http = Mock(return_value={'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{"action_chunk":[]}'}}]})
        api.exchange(current, schema(), threading.Event(), 'wam-api')
        codex = object.__new__(ChunkCodexSupervisor)
        codex.path, codex.proxy, codex.timeout = session.path, None, 600
        codex.data_history = copy.deepcopy(original)
        with patch('astra.data_only_relay.DataOnlyRelay') as relay, \
             patch.object(CodexSupervisor, '_call', return_value={'action_chunk': []}):
            relay.return_value.__enter__.return_value.url = 'http://fixture'
            codex.exchange(current, schema(), threading.Event(), 'wam-codex')
            codex_sent = relay.call_args.args[0]['input']
        for sent in (api._http.call_args.args[0]['messages'][1:], codex_sent):
            for i in (0, 2):
                before = json.loads(original[i]['content'][0]['text'])
                after = json.loads(sent[i]['content'][0]['text'])
                self.assertEqual(after['measured_state'], before['measured_state'])
                self.assertEqual(after['policy_history'], before['policy_history'])
                self.assertEqual(after['image_order'], [before['image_order'][0], before['image_order'][2]])
                self.assertEqual(after['reused_wam_images'], [{
                    'image_label': 'HISTORICAL shared image', 'source_request_id': 'current',
                    'source_image_label': 'HISTORICAL shared image'}])
                self.assertEqual(len(sent[i]['content']), 3)
                url = sent[i]['content'][-1]['image_url']
                if isinstance(url, dict): url = url['url']
                self.assertEqual(url, original[i]['content'][-1]['image_url']['url'])
            self.assertEqual(json.loads(sent[-1]['content'][0]['text']), json.loads(current[0]['text']))
            self.assertEqual(len(sent[-1]['content']), 4)
        self.assertEqual(api.data_history, original)
        self.assertEqual(codex.data_history, original)
        self.assertEqual(current, observation('current', 'HISTORICAL changing image', 'different-bytes'))

    def test_wam_label_mismatch_rejected(self):
        bridge = object.__new__(ChunkApiSupervisor)
        bridge.data_history = []
        with self.assertRaisesRegex(ValueError, 'WAM image labels'):
            bridge.request_messages([{'type': 'text', 'text': json.dumps({
                'request_id': 'current', 'image_order': ['HISTORICAL trajectory image']})}])


class RelayTests(unittest.TestCase):
    def test_sse_rejects_every_tool_and_incomplete_response(self):
        check_response(sse())
        for kind in ('function_call','custom_tool_call','web_search_call','computer_call','mcp_call'):
            with self.assertRaises(ValueError):check_response(sse({'type':kind}))
        with self.assertRaises(ValueError):check_response(b'data: {"type":"response.created"}\n\n')

    def test_cli_context_replaced_at_http_boundary(self):
        payload=response_payload('PUBLIC_CONTRACT',[{'role':'user','content':'PUBLIC_STATE'}],schema())
        opener=Mock();opener.open.return_value=io.BytesIO(sse())
        with DataOnlyRelay(payload,None,threading.Event(),opener=opener) as relay:
            raw=json.dumps({'instructions':'PRIVATE_CODE_MARKER','input':[{'text':'PRIVATE_CODE_MARKER'}],
                            'tools':[{'name':'exec'}]}).encode()
            req=urllib.request.Request(relay.url+'/responses',raw,{'Authorization':'test-token'})
            client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with client.open(req,timeout=3) as response:check_response(response.read())
        sent=json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(sent,payload);self.assertNotIn('PRIVATE_CODE_MARKER',json.dumps(sent))
        self.assertEqual(sent['tools'],[]);self.assertEqual(sent['tool_choice'],'none')

    def test_cancelled_request_never_calls_upstream(self):
        cancel=threading.Event();cancel.set();opener=Mock()
        with DataOnlyRelay({},None,cancel,opener=opener) as relay:
            req=urllib.request.Request(relay.url+'/responses',b'{}')
            client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with self.assertRaises(Exception):client.open(req,timeout=3)
        opener.open.assert_not_called()


if __name__=='__main__':unittest.main()
