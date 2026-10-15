"""Stdio MCP server owned by the logged-in user, with bounded SSH read tools.

The model receives no credentials and no arbitrary remote-command tool. This
process is explicitly registered only for the Windows remote-supervisor calls.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def schema(properties,required):
    return {'type':'object','properties':properties,'required':required,'additionalProperties':False}


TOOLS = [
    {'name':'read_file','description':'Read numbered lines of a Leo3 source/URDF/config text file in approved project/ROS roots.',
     'inputSchema':schema({'path':{'type':'string'},'start_line':{'type':'integer','minimum':1},'line_count':{'type':'integer','minimum':1,'maximum':200}},['path'])},
    {'name':'list_directory','description':'List one approved Leo3 directory. Use this to locate actual paths before reading.',
     'inputSchema':schema({'path':{'type':'string'}},['path'])},
    {'name':'search_text','description':'Bounded literal text search in a Leo3 directory; no regex or shell. Narrow the path if truncated.',
     'inputSchema':schema({'path':{'type':'string'},'query':{'type':'string'}},['path','query'])},
    {'name':'ros_query','description':'Read existing ROS domain 103 topics, one topic message, or TF transform; cannot publish, call services, or change state.',
     'inputSchema':schema({'kind':{'type':'string','enum':['topics','topic_info','topic_sample','tf']},'topic':{'type':'string'},'target':{'type':'string'},'source':{'type':'string'}},['kind'])},
]
for tool in TOOLS:
    tool['annotations']={'readOnlyHint':True,'destructiveHint':False,'idempotentHint':True,'openWorldHint':True}


def query(host,repo,tool,arguments):
    if not host or host.startswith('-') or any(c in host for c in '\r\n\0'): raise ValueError('invalid host')
    if tool not in {t['name'] for t in TOOLS}: raise ValueError('unknown read-only tool')
    data=json.dumps({'tool':tool,'arguments':arguments},ensure_ascii=True)
    if len(data)>16000: raise ValueError('query too large')
    command='cd '+shlex.quote(repo)+' && '
    if tool=='ros_query':
        command+='source /home/liberai/Desktop/colcon_ws/install/setup.bash && export ROS_DOMAIN_ID=103 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET && '
    command+='python3 -B -m astra.readonly_query'
    argv=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=5']
    config=Path.home()/'.ssh/config'
    if sys.platform=='win32' and config.is_file():argv+=['-F',str(config)]
    result=subprocess.run([*argv,host,command],input=data,capture_output=True,text=True,encoding='utf-8',timeout=15)
    try: reply=json.loads(result.stdout)
    except ValueError: raise RuntimeError('SSH inspection failed: '+result.stderr[-600:]) from None
    if result.returncode or 'error' in reply: raise RuntimeError(reply.get('error','SSH inspection failed'))
    return reply


def dispatch(message,host,repo):
    method=message.get('method');params=message.get('params',{})
    if method=='initialize':
        return {'protocolVersion':params.get('protocolVersion','2024-11-05'),'capabilities':{'tools':{}},'serverInfo':{'name':'astra-readonly','version':'1.0'}}
    if method=='ping':return {}
    if method=='tools/list':return {'tools':TOOLS}
    if method=='tools/call':
        try:
            result=query(host,repo,params.get('name'),params.get('arguments',{}))
            return {'content':[{'type':'text','text':json.dumps(result,ensure_ascii=False)}],'isError':False}
        except Exception as error:
            return {'content':[{'type':'text','text':str(error)}],'isError':True}
    raise ValueError('unsupported method')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);parser.add_argument('--repo',required=True)
    args=parser.parse_args()
    sys.stdin.reconfigure(encoding='utf-8');sys.stdout.reconfigure(encoding='utf-8')
    for line in sys.stdin:
        try:
            message=json.loads(line)
            if 'id' not in message:continue
            response={'jsonrpc':'2.0','id':message['id']}
            try:response['result']=dispatch(message,args.host,args.repo)
            except Exception as error:response['error']={'code':-32601,'message':str(error)}
            print(json.dumps(response,ensure_ascii=False),flush=True)
        except (ValueError,TypeError):continue


if __name__=='__main__':main()
