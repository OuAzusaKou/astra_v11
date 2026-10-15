import json
from pathlib import Path
import tempfile
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from astra import readonly_query as query
from astra import readonly_mcp as mcp
from astra.supervisor import CodexSupervisor


class ReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'repo';self.root.mkdir()
        self.patch=patch.object(query,'ROOTS',(self.root,));self.patch.start();self.addCleanup(self.patch.stop)
        (self.root/'control.py').write_text('chest_link\n中文 hand mapping\nthird\n',encoding='utf-8')

    def test_paged_utf8_read(self):
        result=query.handle('read_file',{'path':'control.py','start_line':2,'line_count':1})
        self.assertEqual(result['text'],'2: 中文 hand mapping')
        self.assertEqual(result['total_lines'],3)

    def test_search_and_listing(self):
        self.assertEqual(query.handle('search_text',{'path':str(self.root),'query':'CHEST'})['matches'][0]['line'],1)
        self.assertEqual(query.handle('list_directory',{'path':str(self.root)})['entries'][0]['name'],'control.py')

    def test_private_outside_and_binary_rejected(self):
        (self.root/'.ssh').mkdir();(self.root/'.ssh/key.txt').write_text('secret')
        outside=self.root.parent/'outside.txt';outside.write_text('outside')
        (self.root/'image.png').write_bytes(b'not text')
        for path in (str(outside),'.ssh/key.txt','image.png'):
            with self.assertRaises(ValueError): query.handle('read_file',{'path':path})

    def test_unknown_and_control_queries_rejected(self):
        with patch.object(query.subprocess,'run') as run:
            for kind in ('publish','service_call','reset','parameter_set','launch'):
                with self.assertRaises(ValueError):query.handle('ros_query',{'kind':kind})
            with self.assertRaises(ValueError):query.handle('execute',{'command':'rm'})
            with self.assertRaises(ValueError):query.handle('ros_query',{'kind':'tf','target':'chest_link; rm','source':'left_link_ee'})
            run.assert_not_called()

    def test_ros_only_fixed_argv(self):
        with patch.object(query.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='TF',stderr='')) as run:
            query.handle('ros_query',{'kind':'topic_info','topic':'/joint_states'})
        self.assertEqual(run.call_args.args[0],['ros2','topic','info','/joint_states','--verbose'])
        self.assertNotIn('shell',run.call_args.kwargs)

    def test_query_limit(self):
        with self.assertRaises(ValueError):query.handle('read_file',{'path':'control.py','line_count':201})
        with self.assertRaises(ValueError):query.handle('search_text',{'path':str(self.root),'query':''})

    def test_stdio_protocol_and_error(self):
        result=mcp.dispatch({'method':'initialize','params':{'protocolVersion':'2025-03-26'}},'leo3','/repo')
        self.assertEqual(result['protocolVersion'],'2025-03-26')
        self.assertTrue(all(t['annotations']['readOnlyHint'] for t in mcp.TOOLS))
        with patch.object(mcp,'query',side_effect=RuntimeError('unavailable')):
            result=mcp.dispatch({'method':'tools/call','params':{'name':'read_file'}},'leo3','/repo')
        self.assertTrue(result['isError'])

    def test_ssh_transports_data_not_interpolated_shell(self):
        value='control.py; touch /tmp/should_not_exist'
        with patch.object(mcp.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='{}',stderr='')) as run:
            mcp.query('leo3','/repo space','read_file',{'path':value})
        args=run.call_args.args[0]
        self.assertNotIn(value,args[-1]);self.assertIn('StrictHostKeyChecking=yes',args)
        self.assertEqual(json.loads(run.call_args.kwargs['input'])['arguments']['path'],value)

    def test_ready_requires_actual_remote_tool_result(self):
        b=CodexSupervisor(SimpleNamespace(path=self.root,owner='test'),binary=sys.executable,
                          readonly_remote=('leo3','/repo'))
        events=b.path/'bootstrap.events.jsonl'
        events.write_text('',encoding='utf-8')
        with patch('astra.supervisor.platform.system',return_value='Windows'),patch.object(b,'_call',return_value={'ready':True}):
            with self.assertRaisesRegex(RuntimeError,'startup refused'): b.bootstrap(threading.Event())
            item={'type':'mcp_tool_call','server':'astra_readonly','tool':'read_file','status':'completed',
                  'arguments':{'path':'/repo/envs/leo_wuji_env_infer.py'},'error':None,
                  'result':{'content':[{'type':'text','text':json.dumps({'path':'/repo/envs/leo_wuji_env_infer.py','text':'source'})}]}}
            events.write_text(json.dumps({'item':item}),encoding='utf-8')
            b.bootstrap(threading.Event())
            item['error']='SSH failed'
            events.write_text(json.dumps({'item':item}),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'startup refused'): b.bootstrap(threading.Event())


if __name__=='__main__':unittest.main()
