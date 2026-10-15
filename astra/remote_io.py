"""SSH mailbox helper; limited to an explicit remote-backend runner session."""
import base64
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
from .session import atomic_json, submit


def handle(directory, payload):
    directory=Path(directory).expanduser().resolve()
    if not (directory/'manifest.json').exists():
        return {'waiting_for_runner':True}
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('settings',{}).get('supervisor_backend')!='mac':
        raise ValueError('runner was not started with --supervisor-backend mac')
    with (directory/'mac_peer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return locked_handle(directory, manifest, payload)


def locked_handle(directory, manifest, payload):
    sid,owner=manifest['session_id'],manifest['owner']
    client=str(uuid.UUID(payload['client_id']))
    status=json.loads((directory/'status.json').read_text())
    if status['state'] in ('FINISHED','EXITED'):
        return {'status':status, 'closed':True}
    now=time.time(); path=directory/'mac_peer.json'
    peer=json.loads(path.read_text()) if path.exists() else {}
    if peer.get('client_id')!=client:
        if peer and now-peer.get('heartbeat_at',0)<30:
            raise ValueError('another Mac bridge owns this session')
        peer={'client_id':client,'session_id':sid,'owner':owner,'ready':False}
    if payload.get('session_id',sid)!=sid:
        raise ValueError('foreign session id')
    op=payload['operation']
    peer['reconnect_generation'] = int(payload.get('reconnect_generation', 0))
    if op=='ready':
        from .supervisor_models import model_info
        expected=model_info(manifest.get('settings',{}).get('astra_model','astra'))
        supplied={k:payload.get(k,v if expected['provider']=='codex' else None) for k,v in expected.items()}
        if supplied!=expected:
            raise ValueError('Bridge model does not match fixed runner startup model')
        peer.update(ready=True,thread_id=str(uuid.UUID(payload['thread_id'])))
        peer.update(expected)
        peer['auto_reconnect'] = payload.get('auto_reconnect') is True
    if op=='disconnect':
        peer['ready']=False
    peer['heartbeat_at']=now
    atomic_json(path,peer)
    result={'manifest':{'session_id':sid,'owner':owner,'instruction':manifest.get('instruction'),
                        'astra_model':manifest.get('settings',{}).get('astra_model','astra'),
                        'control_version':manifest.get('control_config',{}).get('version')},
            'status':status, 'peer':peer}
    if op in ('status','ready','disconnect'):
        return result
    if op not in ('request','respond'):
        raise ValueError('unsupported bridge operation')
    if status['state']!='WAITING' or status.get('request_id')!=payload.get('request_id'):
        return {**result,'stale':True}
    req=json.loads((directory/'request.json').read_text())
    if any(req[k]!=status[k] for k in ('session_id','owner','request_id')):
        return {**result,'stale':True}
    if op=='respond':
        response=payload['response']
        if any(response.get(k)!=req[k] for k in ('session_id','owner','request_id')):
            raise ValueError('mismatched response identity')
        submit(directory,response)
        return {**result,'submitted':True}
    request=copy.deepcopy(req); images={};size=0
    if request.get('version') == 10:
        request.pop('wam_context', None)
    snapshots=[request['observation']]+[r['observation'] for r in request.get('wam_context',{}).get('trajectories',[])]
    groups=[s['images'].values() for s in snapshots]
    for snapshot in snapshots:
        depth=snapshot.get('depth',{})
        if depth.get('status')=='available':
            groups.append([depth['raw'],depth['valid_mask']])
    groups.append(request.get('reference_video', {}).get('images', []))
    groups.append((request.get('imagination') or {}).get('images', []))
    groups.extend(s.get('images', {}).values() for s in
                  (request.get('execution_feedback') or {}).get('samples', []))
    for group in groups:
        for image in group:
            file=Path(image['path']).resolve()
            if not file.is_relative_to(directory):
                raise ValueError('image outside runner session')
            data=file.read_bytes();size+=len(data)
            if size>32_000_000:
                raise ValueError('request images exceed transport limit')
            if hashlib.sha256(data).hexdigest()!=image['sha256']:
                raise ValueError('image checksum mismatch')
            images[str(file)]=base64.b64encode(data).decode('ascii')
    return {**result,'request':request,'images':images}


def main():
    try:
        payload=json.loads(sys.stdin.read(2_000_001))
        print(json.dumps(handle(sys.argv[1],payload),ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'error':str(error)}))
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
