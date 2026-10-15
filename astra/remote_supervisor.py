"""Robot-side mailbox adapter. No Codex/model process runs in this backend."""
import json
import time
from .session import atomic_json


class RemoteDisconnected(RuntimeError):
    pass


class RemoteSupervisor:
    def __init__(self, session, timeout=600., heartbeat_timeout=30.):
        self.session, self.timeout = session, timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.request_id = None
        self.thread_id = None
        self.started = None
        self.reconnect_pause = None
        self.reconnect_deadline = None
        self.reconnect_seen_at = None
        self.reconnect_attempts = 0
        self.peer_generation = None
        self.model_metadata = {}
        self.log('WAITING_FOR_MAC: start the Mac bridge for this exact runner session')

    def log(self, message):
        print('[REMOTE ASTRA] ' + message, flush=True)
        with (self.session.path/'supervisor.log').open('a') as f:
            f.write(f'[{time.strftime("%H:%M:%S")}] {message}\n')
        atomic_json(self.session.path/'supervisor.json', {
            'backend':'mac', 'thread_id':self.thread_id, 'owner':self.session.owner,
            'runner_session':str(self.session.path), 'request_id':self.request_id,
            'status':message, 'updated_at':time.time(),
            'model_reasoning_effort':None if self.model_metadata.get('provider')=='quickrouter' else 'medium',
            'timeout_seconds':self.timeout, **self.model_metadata})

    def peer(self):
        try:
            p=json.loads((self.session.path/'mac_peer.json').read_text())
        except FileNotFoundError:
            return None
        if p.get('session_id') != self.session.id or p.get('owner') != self.session.owner:
            raise ValueError('foreign Mac connection')
        if not p.get('ready') or time.time()-p['heartbeat_at'] > self.heartbeat_timeout:
            return None
        return p

    def bootstrap(self, stop):
        deadline=time.monotonic()+self.timeout
        while not stop.is_set():
            peer=self.peer()
            if peer:
                self.peer_generation = (peer['client_id'], peer.get('reconnect_generation', 0))
                self.thread_id=peer['thread_id']
                self.model_metadata={k:peer[k] for k in ('provider','model','conversation_kind','inspection_mode') if k in peer}
                self.log('READY: '+self.model_metadata.get('model','Mac Astra')+' conversation='+self.thread_id)
                return
            if time.monotonic()>deadline:
                raise TimeoutError('Mac supervisor handshake timed out')
            stop.wait(.2)
        raise InterruptedError('runner stopped before Mac connected')

    def invalidate(self):
        self.request_id=None
        self.started=None
        self.session.consume('response.json')  # archive, never apply, any late response

    def poll(self, request):
        if request is None:
            if self.request_id:
                self.log('RECONNECTING: old response invalidated; fresh-observation retry pending'
                         if self.reconnect_pause else 'CANCELLED: pending Mac response invalidated')
            self.invalidate()
            return None
        peer=self.peer()
        if not peer:
            raise RemoteDisconnected('Mac supervisor disconnected/stale; no Astra actions released')
        self.thread_id=peer['thread_id']
        generation = (peer['client_id'], peer.get('reconnect_generation', 0))
        if self.peer_generation is not None and generation != self.peer_generation:
            self.peer_generation = generation
            raise RemoteDisconnected('SSH connection recovered; discard old request and refresh observations')
        self.peer_generation = generation
        if self.request_id != request['request_id']:
            self.request_id=request['request_id']
            self.started=time.monotonic()
            self.log('WAITING: Mac processing request='+self.request_id)
        if time.monotonic()-self.started>self.timeout:
            raise TimeoutError('Mac Astra response timeout')
        response=self.session.consume('response.json')
        if response is None:
            return None
        if any(response.get(k)!=request[k] for k in ('session_id','owner','request_id')):
            self.session.event('remote_stale_response_discarded')
            return None
        self.log('RESPONSE: '+str(response.get('operation'))+' | '+str(response.get('reason')))
        return response

    def wait_for_reconnect(self, controller):
        """Called only for our typed communication failure, never model pause text."""
        self.reconnect_pause = controller.pause_id
        self.reconnect_deadline = time.monotonic() + 300
        self.reconnect_seen_at = None
        self.log('RECONNECTING: transport paused; wait up to 300s for a fresh ready peer')

    def recover(self, controller):
        if self.reconnect_pause is None:
            return False
        if (controller.pause_id != self.reconnect_pause or controller.state != 'PAUSED_FOR_OPERATOR'
                or controller.mode != 'ASTRA' or controller.settings.version < 2):
            self.reconnect_pause = None
            return False
        if time.monotonic() > self.reconnect_deadline or self.reconnect_attempts >= 3:
            self.reconnect_pause = None
            if (controller.settings.version in (4, 8, 10) and controller.settings.astra_pause_handback):
                resumed = controller.supervisor_stopped('Supervisor transport recovery exhausted',
                                                       source='transport_exhausted')
                self.log('RECONNECT STOPPED: '+('runner fallback to WAM' if resumed else 'remains paused'))
                return resumed
            self.log('RECONNECT STOPPED: retry window/limit reached; operator R required')
            return False
        peer = self.peer()
        if not peer or not peer.get('auto_reconnect') or time.time()-peer['heartbeat_at'] > 5:
            self.reconnect_seen_at = None
            return False
        if self.reconnect_seen_at is None:
            self.reconnect_seen_at = time.monotonic()
            return False
        if time.monotonic()-self.reconnect_seen_at < 2 or controller.switch_pending() or controller.cancelled():
            return False
        self.reconnect_pause = None
        self.reconnect_attempts += 1
        self.peer_generation = (peer['client_id'], peer.get('reconnect_generation', 0))
        self.invalidate()
        # Includes robot.check; never rearm/clear a hardware fault or reuse actions.
        controller.retry_astra(source='transport')
        self.log(f'RECONNECTED: fresh Astra observation requested ({self.reconnect_attempts}/3)')
        return True

    def close(self):
        self.invalidate()
        self.log('STOPPED: runner no longer accepts Mac decisions')
