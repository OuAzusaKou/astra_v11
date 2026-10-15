import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import uuid

from astra.session import Session
from astra.remote_io import handle
from astra.remote_supervisor import RemoteSupervisor
from astra.mac_bridge import materialize
from test_astra_mode2 import observation

class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.s=Session(Path(self.temp.name)/'run','test',{'settings':{'supervisor_backend':'mac'}})
        self.s.status(state='STARTING')
        self.client=str(uuid.uuid4())
        self.remote=RemoteSupervisor(self.s)

    def call(self,op,**data):
        return handle(self.s.path,{'client_id':self.client,'operation':op,**data})

    def request(self):
        self.s.request(stage='ASTRA_TAKEOVER',observation=self.s.snapshot(observation(),'test'),wam_context={'trajectories':[]})
        self.s.status(state='WAITING')
        return self.s.pending

    def ready(self):
        self.call('ready',thread_id=str(uuid.uuid4()))
        self.remote.bootstrap(threading.Event())

    def test_handshake_and_roundtrip_materialized_images(self):
        self.ready();req=self.request()
        bundle=self.call('request',request_id=req['request_id'])
        copied=materialize(bundle,Path(self.temp.name)/'cache')
        self.assertEqual(copied['request_id'],req['request_id'])
        for key,image in copied['observation']['images'].items():
            self.assertTrue(Path(image['path']).exists())
            self.assertEqual(hashlib.sha256(Path(image['path']).read_bytes()).hexdigest(),image['sha256'])
            self.assertNotEqual(image['path'],req['observation']['images'][key]['path'])
        d={k:req[k] for k in ('session_id','owner','request_id')}
        d.update(operation='pause',reason='test',evidence='fixture')
        self.call('respond',request_id=req['request_id'],response=d)
        self.assertEqual(self.remote.poll(req),d)
        self.assertIsNone(self.remote.poll(req))

    def test_stale_response_after_handback_is_rejected(self):
        self.ready();req=self.request()
        self.s.pending=None;self.s.status(state='AUTO')
        d={k:req[k] for k in ('session_id','owner','request_id')}
        self.assertTrue(self.call('respond',request_id=req['request_id'],response=d)['stale'])
        self.assertFalse((self.s.path/'response.json').exists())

    def test_already_queued_response_invalidated_on_reset(self):
        self.ready();req=self.request()
        d={k:req[k] for k in ('session_id','owner','request_id')}
        self.call('respond',request_id=req['request_id'],response=d)
        self.remote.invalidate()
        self.assertFalse((self.s.path/'response.json').exists())

    def test_disconnected_mac_blocks_response(self):
        self.ready();req=self.request();self.call('disconnect')
        with self.assertRaisesRegex(RuntimeError,'disconnected'):
            self.remote.poll(req)

    def test_second_bridge_cannot_steal_live_session(self):
        self.call('status')
        with self.assertRaisesRegex(ValueError,'another Mac'):
            handle(self.s.path,{'client_id':str(uuid.uuid4()),'operation':'status'})

    def test_local_backend_cannot_be_claimed(self):
        manifest=json.loads((self.s.path/'manifest.json').read_text())
        manifest['settings']['supervisor_backend']='local'
        (self.s.path/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'not started'):
            self.call('status')

    def test_corrupt_image_rejected(self):
        self.ready();req=self.request()
        bundle=self.call('request',request_id=req['request_id'])
        k=next(iter(bundle['images']));bundle['images'][k]='YQ=='
        with self.assertRaisesRegex(ValueError,'checksum'):
            materialize(bundle,Path(self.temp.name)/'cache')

    def test_reference_image_export_and_rewrite(self):
        self.ready();req=self.request()
        req['wam_context']['trajectories']=[{'observation':self.s.snapshot(observation(),'history')}]
        (self.s.path/'request.json').write_text(json.dumps(req))
        bundle=self.call('request',request_id=req['request_id'])
        copied=materialize(bundle,Path(self.temp.name)/'cache')
        self.assertTrue(Path(copied['wam_context']['trajectories'][0]['observation']['state_path']).exists())
