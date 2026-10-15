import io
import json
import unittest
from unittest.mock import patch

from astra.chat_stream import read_completion, IncompleteStream
from astra.quickrouter_transport import request, TransportFailure


def sse(events, done=True):
    raw=b': keepalive\r\n\r\n'
    for event in events:
        raw+=('data: '+json.dumps(event,ensure_ascii=False)+'\r\n\r\n').encode()
    if done:raw+=b'data: [DONE]\n\n'
    return io.BytesIO(raw)


def chunk(delta,finish=None):
    return {'choices':[{'index':0,'delta':delta,'finish_reason':finish}]}


class ChatStreamTests(unittest.TestCase):
    def test_text_unicode_reasoning_usage_and_complete_marker(self):
        events=[]
        out=read_completion(sse([chunk({'role':'assistant','content':'{"答'}),
            chunk({'content':'案":1}','reasoning_content':'reason'}),chunk({},'stop'),
            {'choices':[],'usage':{'completion_tokens':10}}]),lambda **e:events.append(e))
        message=out['choices'][0]['message']
        self.assertEqual(json.loads(message['content']),{'答案':1})
        self.assertEqual(message['reasoning_content'],'reason')
        self.assertEqual(out['usage']['completion_tokens'],10)
        self.assertEqual(events[-1]['event'],'stream_complete')

    def test_parallel_tool_arguments_and_extensions_remain_separate(self):
        out=read_completion(sse([chunk({'role':'assistant','tool_calls':[
            {'index':1,'id':'second','type':'function','function':{'name':'ros_query','arguments':'{"x":'}},
            {'index':0,'id':'first','type':'function','function':{'name':'read_file','arguments':'{"path":"'}}]}),
            chunk({'tool_calls':[{'index':0,'function':{'arguments':'a"}'},'extra_content':{'google':{'thought_signature':'sig'}}},
                {'index':1,'function':{'arguments':'2}'}}]}),chunk({},'tool_calls')]))
        calls=out['choices'][0]['message']['tool_calls']
        self.assertEqual([c['id'] for c in calls],['first','second'])
        self.assertEqual(json.loads(calls[0]['function']['arguments']),{'path':'a'})
        self.assertEqual(json.loads(calls[1]['function']['arguments']),{'x':2})
        self.assertEqual(calls[0]['extra_content']['google']['thought_signature'],'sig')

    def test_truncation_missing_finish_upstream_error_never_return_partial(self):
        cases=[(sse([chunk({'role':'assistant','content':'{}'},'stop')],False),IncompleteStream),
               (sse([chunk({'role':'assistant','content':'{}'})]),IncompleteStream),
               (sse([{'error':{'message':'PRIVATE'}}]),RuntimeError)]
        for data,error in cases:
            with self.assertRaises(error):read_completion(data)

    def test_transport_assembles_stream_and_keeps_five_minute_wait(self):
        data=sse([chunk({'role':'assistant','content':'{}'},'stop')])
        with patch('astra.quickrouter_transport.api_key',return_value='fixture'),patch('astra.quickrouter_transport.urllib.request.build_opener') as build:
            response=build.return_value.open.return_value.__enter__.return_value
            response.status=200;response.readline.side_effect=data.readline
            result=request({'stream':True})
            self.assertEqual(result['choices'][0]['message']['content'],'{}')
            response.read.assert_not_called()
            self.assertEqual(build.return_value.open.call_args.kwargs['timeout'],300)

    def test_truncated_transport_is_retryable_without_content_leak(self):
        data=sse([chunk({'role':'assistant','content':'PRIVATE PARTIAL'})],False);events=[]
        with patch('astra.quickrouter_transport.api_key',return_value='fixture'),patch('astra.quickrouter_transport.urllib.request.build_opener') as build:
            response=build.return_value.open.return_value.__enter__.return_value
            response.status=200;response.readline.side_effect=data.readline
            with self.assertRaises(TransportFailure) as caught:request({'stream':True},diagnostic=events.append)
        self.assertEqual(caught.exception.details['stage'],'read_stream_body')
        self.assertTrue(caught.exception.details['retryable'])
        self.assertNotIn('PRIVATE PARTIAL',json.dumps(events)+str(caught.exception))


if __name__=='__main__':unittest.main()
