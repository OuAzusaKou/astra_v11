import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from astra.api_supervisor import QuickRouterSupervisor
from astra.qwen_context import build_context
from astra.supervisor_models import model_info


def completion(value,**extra):
    return {'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(value),**extra}}]}


class QwenMode(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.bridge=QuickRouterSupervisor(SimpleNamespace(path=self.root,owner='fixture'),'qwen3.8-max',
            readonly_remote=('leo3','/home/liberai/Liberai_Deploy_astra'))
        self.identity=dict(session_id='s',owner='fixture',request_id='r')
        self.bridge.active_identity=self.identity
        self.context=dict(context_sha256='digest',files=[],action_contract={'dimensions':52,'frame':'chest_link'})
        self.decision=dict(self.identity,operation='observe',reason='fresh view',evidence='fixture')

    def test_no_tools_sent_and_wrong_identity_still_rejected(self):
        self.bridge.control_context=self.context
        with patch.object(self.bridge,'_http',return_value=completion(self.decision)) as send:
            self.bridge._call('request',[],threading.Event(),'r')
        self.assertNotIn('tools',send.call_args.args[0]);self.assertNotIn('tool_choice',send.call_args.args[0])
        self.assertIn('QWEN TOOL-FREE TRANSPORT OVERRIDE',send.call_args.args[0]['messages'][0]['content'][0]['text'])
        with patch.object(self.bridge,'_http',return_value=completion(dict(self.decision,request_id='old'))),self.assertRaises(ValueError):
            self.bridge._call('request',[],threading.Event(),'old')
        self.assertFalse((self.bridge.path/'old.json').exists())

    def test_unexpected_tool_call_never_executed(self):
        self.bridge.control_context=self.context
        with patch.object(self.bridge,'_http',return_value=completion(None,tool_calls=[{'id':'bad'}])), \
             patch.object(self.bridge,'_tool') as tool,self.assertRaises(ValueError):
            self.bridge._call('request',[],threading.Event(),'r')
        tool.assert_not_called();self.assertFalse(self.bridge.history)

    def test_shared_v4_v8_prompts_keep_version_rules_without_tools(self):
        self.bridge.control_context=self.context
        for version in (4,8):
            request=dict(self.identity,version=version,stage='ASTRA_TAKEOVER',observation={'images':{}},astra_pause_handback=True)
            self.bridge.worker=None
            with patch.object(self.bridge,'_http',return_value=completion(self.decision)) as send:
                self.bridge.start(request);self.bridge.worker.join(3)
            self.assertIsNone(self.bridge.error)
            self.assertNotIn('tools',send.call_args.args[0])
            prompt=json.loads((self.bridge.path/'r.input.json').read_text(encoding='utf-8'))['prompt']
            self.assertIn('RETAIN TAKEOVER OWNERSHIP',prompt)
            self.assertIn('QWEN TOOL-FREE TRANSPORT OVERRIDE',prompt)
            if version==8:self.assertIn('operation=propose_candidates',prompt)

    def test_no_context_no_api_request(self):
        with patch.object(self.bridge,'_http') as send,self.assertRaises(RuntimeError):
            self.bridge._call('request',[],threading.Event(),'r')
        send.assert_not_called()

    def test_bootstrap_vision_context_ack_and_metadata(self):
        expected=dict(ready=True,visual_code='ABCDEF',context_sha256='digest',action_dimensions=52,action_frame='chest_link')
        with patch('astra.qwen_context.build_context',return_value=self.context), \
             patch('astra.api_supervisor.uuid.uuid4',return_value=SimpleNamespace(hex='abcdef123456')), \
             patch.object(self.bridge,'_http',return_value=completion(expected)) as send:
            self.bridge.bootstrap(threading.Event())
        self.assertNotIn('tools',send.call_args.args[0])
        initialization=next(m for m in send.call_args.args[0]['messages'] if m['role']=='user')['content'][0]['text']
        self.assertIn('There is no CURRENT_REQUEST',initialization)
        self.assertNotIn('Return one request-bound decision JSON',initialization)
        self.assertTrue((self.bridge.path/'control_context.json').exists())
        self.assertEqual(model_info('qwen3.8-max')['inspection_mode'],'preloaded_no_tools')
        self.assertEqual(json.loads((self.root/'supervisor.json').read_text())['inspection_mode'],'preloaded_no_tools')
        with patch('astra.qwen_context.build_context',return_value=self.context),patch.object(self.bridge,'_http',return_value=completion(dict(expected,action_dimensions=58))),self.assertRaises(RuntimeError):
            self.bridge.bootstrap(threading.Event())

    def test_source_failure_prevents_model_handshake(self):
        with patch('astra.qwen_context.build_context',side_effect=RuntimeError('missing source')),patch.object(self.bridge,'_http') as send,self.assertRaises(RuntimeError):
            self.bridge.bootstrap(threading.Event())
        send.assert_not_called()

    def test_source_pagination_hash_and_truncation(self):
        def reader(path,start,count):
            return dict(path=path,start_line=start,total_lines=105,truncated=False,
                text='\n'.join(str(i)+': source '+str(i) for i in range(start,min(106,start+count))))
        context=build_context('/repo',reader,threading.Event())
        self.assertEqual(len(context['files']),6)
        self.assertTrue(all(len(f['text'].splitlines())==105 for f in context['files']))
        self.assertEqual(context['context_sha256'],build_context('/repo',reader,threading.Event())['context_sha256'])
        with self.assertRaises(RuntimeError):build_context('/repo',lambda *a:dict(reader(*a),truncated=True),threading.Event())
        cancel=threading.Event();cancel.set()
        with self.assertRaises(InterruptedError):build_context('/repo',reader,cancel)


if __name__=='__main__':unittest.main()
