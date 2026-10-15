"""Mac-owned Astra supervisor, transporting only bound observations/decisions over SSH."""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

from .session import atomic_json
from .supervisor import CodexSupervisor


class SSHTransportError(RuntimeError):
    """Transient I/O failure, never a protocol/ownership validation failure."""


class Transport:
    def __init__(self, host, repo, session, timeout=60., connect_timeout=15,
                 retries=3, stop=None, auto_reconnect=True):
        if host.startswith('-'):
            raise ValueError('invalid SSH host')
        self.host, self.repo, self.session=host,repo,session
        self.client_id=str(uuid.uuid4())
        self.session_id=None
        self.timeout, self.connect_timeout, self.retries = timeout, connect_timeout, retries
        self.stop = stop or threading.Event()
        self.auto_reconnect = auto_reconnect
        self.on_retry = None
        self.reconnect_generation = 0

    def call(self, operation, **fields):
        # Do not retransmit a response after an ambiguous timeout: it may have
        # reached the robot. The main loop resynchronizes status instead.
        attempts = self.retries + 1 if operation in ('status', 'request', 'ready') else 1
        for attempt in range(attempts):
            if self.stop.is_set() and operation != 'disconnect':
                raise InterruptedError('bridge stopped')
            try:
                return self._call_once(operation, **fields)
            except SSHTransportError as error:
                self.reconnect_generation += 1
                if self.on_retry:
                    self.on_retry()
                if attempt + 1 == attempts:
                    raise
                delay = min(2 ** (attempt + 1), 8)
                print(f'SSH RETRY {attempt+1}/{self.retries}: {error}; retry in {delay}s', flush=True)
                if self.stop.wait(delay):
                    raise InterruptedError('bridge stopped')

    def _call_once(self, operation, **fields):
        data={'operation':operation,'client_id':self.client_id,**fields}
        data['reconnect_generation'] = self.reconnect_generation
        if operation == 'ready':
            data['auto_reconnect'] = self.auto_reconnect
        if self.session_id:
            data['session_id']=self.session_id
        command='cd '+shlex.quote(self.repo)+' && python3 -B -m astra.remote_io '+shlex.quote(self.session)
        argv=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
              '-o',f'ConnectTimeout={self.connect_timeout}','-o','ServerAliveInterval=15',
              '-o','ServerAliveCountMax=3']
        config=Path.home()/'.ssh/config'
        if os.name == 'nt' and config.is_file():
            argv += ['-F',str(config)]
        try:
            r=subprocess.run([*argv,self.host,command], input=json.dumps(data),capture_output=True,
                             text=True,encoding='utf-8',timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            raise SSHTransportError(f'{operation} exceeded {self.timeout:g}s') from error
        if r.returncode == 255:
            raise SSHTransportError(r.stderr[-600:])
        try:
            result=json.loads(r.stdout)
        except ValueError:
            if not r.stdout.strip():
                raise SSHTransportError('SSH returned no helper response: '+r.stderr[-600:]) from None
            raise RuntimeError('SSH transport failed: '+r.stderr[-600:]) from None
        if r.returncode or result.get('error'):
            raise RuntimeError(result.get('error','SSH helper failed'))
        if 'manifest' in result:
            sid=result['manifest']['session_id']
            if self.session_id and sid!=self.session_id:
                raise ValueError('runner session changed under bridge')
            self.session_id=sid
        return result


def materialize(bundle, directory):
    """Rewrite only observation paths, verify bytes against original checksums."""
    req=copy.deepcopy(bundle['request'])
    directory=Path(directory);directory.mkdir(exist_ok=True)
    snapshots=[req['observation']]+[r['observation'] for r in req.get('wam_context',{}).get('trajectories',[])]
    for index,snapshot in enumerate(snapshots):
        for image in snapshot['images'].values():
            source=image['path']
            data=base64.b64decode(bundle['images'][source],validate=True)
            digest=hashlib.sha256(data).hexdigest()
            if digest!=image['sha256']:
                raise ValueError('downloaded image checksum mismatch')
            target=directory/(digest+'.png')
            target.write_bytes(data)
            image['remote_path']=source
            image['path']=str(target.resolve())
        state_file=directory/f'{req["request_id"]}_{index}_state.json'
        atomic_json(state_file,snapshot['state'])
        snapshot['remote_state_path']=snapshot['state_path']
        snapshot['state_path']=str(state_file.resolve())
        depth=snapshot.get('depth')
        if depth is not None:
            if depth.get('status')=='available':
                for key in ('raw','valid_mask'):
                    asset=depth[key]
                    source=asset['path']
                    data=base64.b64decode(bundle['images'][source],validate=True)
                    digest=hashlib.sha256(data).hexdigest()
                    if digest!=asset['sha256']:
                        raise ValueError('downloaded depth checksum mismatch')
                    target=directory/(digest+'.png')
                    target.write_bytes(data)
                    asset['remote_path']=source
                    asset['path']=str(target.resolve())
            depth['remote_metadata_path']=depth['metadata_path']
            depth['metadata_path']=str((directory/f'{req["request_id"]}_{index}_depth.json').resolve())
            atomic_json(depth['metadata_path'],depth)
    for image in (req.get('reference_video', {}).get('images', [])
                  + (req.get('imagination') or {}).get('images', [])
                  + [image for sample in (req.get('execution_feedback') or {}).get('samples', [])
                     for image in sample.get('images', {}).values()]):
        source=image['path']
        data=base64.b64decode(bundle['images'][source],validate=True)
        digest=hashlib.sha256(data).hexdigest()
        if digest!=image['sha256']:
            raise ValueError('downloaded reference image checksum mismatch')
        target=directory/(digest+Path(source).suffix)
        target.write_bytes(data)
        image['remote_path']=source
        image['path']=str(target.resolve())
    return req


def run(args, stop):
    transport=Transport(args.host,args.remote_repo,args.session,
                        timeout=args.ssh_timeout,connect_timeout=args.ssh_connect_timeout,
                        retries=args.ssh_retries,stop=stop,auto_reconnect=not args.no_auto_reconnect)
    bridge=None
    try:
        print('Waiting for the exact leo3 session: '+args.session,flush=True)
        while not stop.is_set():
            info=transport.call('status')
            if info.get('closed'):
                raise RuntimeError('runner already ended; use a new session')
            if not info.get('waiting_for_runner'):
                break
            stop.wait(1)
        if stop.is_set():
            return
        manifest=info['manifest']
        from .supervisor_models import model_info
        chosen=getattr(args,'astra_model','astra')
        if model_info(manifest.get('astra_model','astra'))!=model_info(chosen):
            raise ValueError('Laptop and runner --astra-model differ; use the same startup selection')
        cache=Path(args.cache_root).expanduser().resolve()/f'{manifest["session_id"]}_{transport.client_id}'
        cache.mkdir(parents=True,exist_ok=False)
        atomic_json(cache/'remote_binding.json',{'host':args.host,'remote_session':args.session,**manifest})
        local=SimpleNamespace(path=cache,owner=manifest['owner'],id=manifest['session_id'])
        if manifest.get('control_version') in (10, 11):
            from .supervisor_models import build_supervisor
            bridge=build_supervisor(local, model=chosen, version=manifest['control_version'],
                                    timeout=args.timeout, proxy=args.proxy or None,
                                    binary=getattr(args,'codex_bin',None),
                                    readonly_remote=(args.host,args.remote_repo))
        elif chosen=='astra':
            bridge=CodexSupervisor(local,timeout=args.timeout,proxy=args.proxy or None,
                                   binary=getattr(args,'codex_bin',None),
                                   readonly_remote=(args.host,args.remote_repo))
        else:
            from .api_supervisor import QuickRouterSupervisor
            bridge=QuickRouterSupervisor(local,model=chosen,timeout=args.timeout,proxy=args.proxy or None,
                                         readonly_remote=(args.host,args.remote_repo))
        # Cancel in-flight model work immediately on an SSH failure. Even when
        # the runner's 30s heartbeat has not expired, never replay its result.
        transport.on_retry = bridge.invalidate
        errors=[]
        def bootstrap():
            try: bridge.bootstrap(stop)
            except Exception as error:errors.append(error)
        init=threading.Thread(target=bootstrap,daemon=True)
        init.start()
        while init.is_alive() and not stop.is_set():
            info=transport.call('status')  # heartbeat while Mac Codex initializes
            if info.get('closed'):
                stop.set()
            stop.wait(1)
        init.join(timeout=8)
        if errors:raise errors[0]
        if stop.is_set():return
        transport.call('ready',thread_id=bridge.thread_id,**model_info(chosen))
        print(f'MAC READY | thread={bridge.thread_id} | logs={cache}',flush=True)
        request=None; sent_id=None
        while not stop.is_set():
            info=transport.call('status')
            if info.get('closed'):break
            status=info['status']
            rid=status.get('request_id') if status['state']=='WAITING' else None
            if request and request['request_id']!=rid:
                bridge.invalidate()
                request=None
            if rid and rid!=sent_id and request is None:
                bundle=transport.call('request',request_id=rid)
                if 'request' in bundle:
                    request=materialize(bundle,cache/'observations')
                    atomic_json(cache/'request.json',request)
            active=request if request and rid!=sent_id else None
            try:
                response=bridge.poll(active)
            except Exception as error:
                if not active:raise
                response={k:active[k] for k in ('session_id','owner','request_id')}
                response.update(operation='pause',pause_kind='technical_error',reason='Mac Astra error: '+str(error),
                                evidence='supervisor failed; no action generated')
            if response is not None:
                try:
                    result=transport.call('respond',request_id=response['request_id'],response=response)
                except SSHTransportError as error:
                    print(f'SSH response acknowledgement lost; resynchronizing, no retransmission: {error}',flush=True)
                    request=None
                    sent_id=response['request_id']
                    continue
                sent_id=response['request_id'];request=None
                print('DECISION '+('submitted' if result.get('submitted') else 'discarded (stale)'),flush=True)
            stop.wait(1)
    finally:
        stop.set()
        if bridge:bridge.close()
        if transport.session_id:
            try:transport.call('disconnect')
            except Exception:pass


def main(argv=None):
    if os.name == 'nt':
        for stream in (sys.stdout,sys.stderr):
            if hasattr(stream,'reconfigure'):
                stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='leo3')
    parser.add_argument('--remote-repo',default='/home/liberai/Liberai_Deploy_astra')
    parser.add_argument('--session',required=True,help='exact absolute runner session path on leo3')
    parser.add_argument('--cache-root',default=str(Path(__file__).resolve().parents[1]/'outputs/mac_supervisor'))
    parser.add_argument('--proxy',default='http://127.0.0.1:7897')
    parser.add_argument('--codex-bin',help='optional Codex executable; on Windows use codex.exe')
    from .supervisor_models import CHOICES
    parser.add_argument('--astra-model',choices=CHOICES,default='astra',help='fixed for this entire session')
    parser.add_argument('--timeout',type=float,default=600.)
    parser.add_argument('--ssh-timeout',type=float,default=60.,help='overall deadline for one SSH helper call')
    parser.add_argument('--ssh-connect-timeout',type=int,default=15)
    parser.add_argument('--ssh-retries',type=int,default=3,help='extra attempts for read/handshake calls')
    parser.add_argument('--no-auto-reconnect',action='store_true',help='do not allow runner communication recovery')
    args=parser.parse_args(argv)
    if args.timeout<=0 or not args.timeout<float('inf'):
        parser.error('timeout must be finite and positive')
    if not 0 < args.ssh_timeout < float('inf') or args.ssh_connect_timeout <= 0 or not 0 <= args.ssh_retries <= 10:
        parser.error('SSH timeouts must be positive and retries must be 0..10')
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    run(args,stop)

if __name__=='__main__':main()
