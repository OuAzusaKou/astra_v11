"""V5 contract tests; fake dynamics and robot, never ROS/hardware."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from astra.control import Controller,Settings
from astra.session import Session
from astra.dynamics import Imagination,ADAPTER_SHA,NORM_SHA,native_request,same_anchor
from test_astra_mode2 import Robot,Policy,observation
from wam_direct.start_mac_vlm import leo_command


def result(*args):
    return dict(success=True,future_image=np.zeros((368,320,3),np.uint8),
                input_image_preview=np.zeros((368,320,3),np.uint8),
                future_proprio=np.zeros((1,346),np.float32),
                future_proprio_valid_mask=np.zeros((1,346),bool),state_space='canonical_normalized',
                adapter_sha256=ADAPTER_SHA,norm_sha256=NORM_SHA)


class V5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.robot=Robot();self.session=Session(Path(self.tmp.name)/'run','v5',{})
        self.c=Controller(self.robot,Policy(),self.session,Settings(400,version=5),'task',initial_l2='pick')
        self.c.imagination=Imagination(mode='review');self.c.start();self.c.keyboard_toggle()

    def decision(self,**kwargs):
        return {**{k:self.session.pending[k] for k in ('session_id','owner','request_id')},
                'reason':'test','evidence':'real test evidence',**kwargs}

    def imagine(self):
        with patch('astra.dynamics.native_request',return_value={}),patch('astra.dynamics.rpc',side_effect=result):
            self.c.decide(self.decision(operation='takeover',actions=np.zeros((30,52)).tolist(),
                         new_attempt=True,change_reason='test',execute_steps=30))
            self.c.imagination.worker.join(2)
            self.assertEqual(self.c.state,'IMAGINING');self.assertEqual(self.c.steps,0)
            self.c.tick()

    def test_candidate_waits_then_exact_prefix_executes_once(self):
        self.imagine();self.assertFalse(self.robot.published)
        self.assertEqual(self.session.pending['stage'],'REVIEW_IMAGINATION')
        info=self.session.pending['imagination']
        self.assertEqual(len(info['actions']),18)
        d=self.decision(operation='approve',execute_steps=18,candidate_id=info['candidate_id'],candidate_sha256=info['candidate_sha256'])
        self.c.decide(d);self.assertEqual(self.c.steps,18);self.assertEqual(len(self.robot.published),1)
        self.assertEqual(self.session.pending['stage'],'VERIFY_RECOVERY')
        with self.assertRaises(ValueError):self.c.decide(d)

    def test_preview_never_publishes(self):
        self.c.imagination.mode='preview';self.imagine()
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR');self.assertFalse(self.robot.published)
        self.assertTrue(list(self.session.path.glob('imagination/*/panel.png')))

    def test_prediction_bundle_and_index(self):
        self.imagine()
        root=self.session.path/'imagination'
        latest=json.loads((root/'latest.json').read_text())
        directory=root/latest['candidate_id']
        for name in ('current.png','future.png','panel.png','prediction.npz','metadata.json'):
            self.assertTrue((directory/name).is_file(),name)
        meta=json.loads((directory/'metadata.json').read_text())
        self.assertEqual(meta['l3'],self.c.l3);self.assertEqual(meta['l2'],self.c.l2)
        self.assertEqual(meta['status_at_save'],'prediction_only')
        self.assertEqual(meta['target_timestamp'],meta['anchor_timestamp']+1.)
        entries=[json.loads(line) for line in (root/'index.jsonl').read_text().splitlines()]
        self.assertEqual(len(entries),1)
        self.assertEqual(entries[0]['metadata'],latest['metadata'])
        # Another reviewed candidate keeps the first bundle and appends the index.
        self.c.imagination.cancel();self.c.queue=np.zeros((30,52))
        with patch('astra.dynamics.native_request',return_value={}),patch('astra.dynamics.rpc',side_effect=result):
            self.c.execute(30);self.c.imagination.worker.join(2);self.c.tick()
        self.assertTrue((directory/'future.png').is_file())
        self.assertEqual(len((root/'index.jsonl').read_text().splitlines()),2)
        self.assertNotEqual(json.loads((root/'latest.json').read_text())['candidate_id'],latest['candidate_id'])

    def test_missing_anchor_mosaic_does_not_save_fake_real_image(self):
        bad=result();bad.pop('input_image_preview')
        with patch('astra.dynamics.native_request',return_value={}),patch('astra.dynamics.rpc',return_value=bad):
            self.c.queue=np.zeros((30,52));self.c.execute(30)
            self.c.imagination.worker.join(2);self.c.tick()
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')
        self.assertFalse((self.session.path/'imagination/latest.json').exists())
        self.assertFalse(self.robot.published)

    def test_wrong_candidate_rejected(self):
        self.imagine()
        with self.assertRaises(ValueError):self.c.decide(self.decision(operation='approve',execute_steps=18,candidate_id='wrong',candidate_sha256='wrong'))
        self.assertFalse(self.robot.published)

    def test_manual_handback_invalidates(self):
        self.imagine();self.c.keyboard_toggle()
        self.assertIsNone(self.c.imagination.pending);self.assertEqual(self.c.mode,'WAM')
        self.assertFalse(self.robot.published)

    def test_changed_scene_blocks_approval(self):
        self.imagine();info=self.session.pending['imagination'];fresh=observation();fresh['left_ee_pose'][0,0]=.1
        with patch.object(self.robot,'observe',return_value=fresh):
            with self.assertRaises(RuntimeError):self.c.decide(self.decision(operation='approve',execute_steps=18,candidate_id=info['candidate_id'],candidate_sha256=info['candidate_sha256']))
        self.assertFalse(self.robot.published)

    def test_prediction_failure_pauses(self):
        with patch('astra.dynamics.native_request',side_effect=ValueError('adapter failure')):
            self.c.queue=np.zeros((30,52));self.c.execute(30);self.c.imagination.worker.join(2);self.c.tick()
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR');self.assertFalse(self.robot.published)

    def test_v5_launcher(self):
        cmd=leo_command('test','Task: a Subtask: b',[],version=5,auto_takeover=True)
        self.assertIn('version5_dynamics.json',cmd);self.assertIn('--dynamics-mode preview',cmd)
        self.assertIn('--vlm-progress-log',cmd)

    def test_native_time_grid_and_real_converter_inputs(self):
        class Converter:
            def convert(self,fields):return fields
        req=native_request(observation(),np.zeros((18,52)),20,.1,'id',Converter())
        self.assertEqual(req['action_length'],30);self.assertEqual(req['duration_seconds'],1.)
        self.assertEqual(req['fields']['action.left_arm_ee_action'].shape,(30,6))
        with self.assertRaises(ValueError):native_request(observation(),np.zeros((30,52)),20,.1,'id',Converter())
        with self.assertRaises(ValueError):native_request(observation(),np.zeros((17,52)),20,.1,'id',Converter())

    def test_native_images_convert_bgr_to_rgb_once(self):
        class Converter:
            def convert(self,fields):return fields
        obs=observation()
        keys=('image','image_wrist_left','image_wrist_right')
        for i,k in enumerate(keys):obs[k][:]=[11+i,37+i,203+i]
        req=native_request(obs,np.zeros((18,52)),20,.1,'color',Converter())
        for i,(k,im) in enumerate(zip(keys,req['images'])):
            np.testing.assert_array_equal(im[0,0],[203+i,37+i,11+i])
            np.testing.assert_array_equal(obs[k][0,0],[11+i,37+i,203+i])
            self.assertTrue(im.flags.c_contiguous)

    def test_anchor_image_drift(self):
        a=observation();b=copy.deepcopy(a);b['image'][:]=255
        self.assertFalse(same_anchor(a,b))

    def test_clock_change_blocks_approval(self):
        self.imagine();info=self.session.pending['imagination'];self.robot.fps=30
        with self.assertRaises(ValueError):self.c.decide(self.decision(operation='approve',execute_steps=18,candidate_id=info['candidate_id'],candidate_sha256=info['candidate_sha256']))
        self.assertFalse(self.robot.published)

    def test_timeout_and_bad_fingerprint_do_not_publish(self):
        self.imagine();self.c.state='IMAGINING';self.c.imagination.pending['created']-=100
        self.c.tick();self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')
        self.assertFalse(self.robot.published)

    def test_v5_imagined_images_transport_to_mac(self):
        import json,uuid
        from astra.remote_io import handle
        from astra.mac_bridge import materialize
        self.imagine()
        mf=self.session.path/'manifest.json';manifest=json.loads(mf.read_text())
        manifest['settings']={'supervisor_backend':'mac'};mf.write_text(json.dumps(manifest))
        req=self.session.pending
        bundle=handle(self.session.path,{'client_id':str(uuid.uuid4()),'operation':'request','request_id':req['request_id']})
        copied=materialize(bundle,Path(self.tmp.name)/'mac')
        image=copied['imagination']['images'][0]
        self.assertTrue(Path(image['path']).is_file())
        self.assertNotEqual(image['path'],req['imagination']['images'][0]['path'])
        self.assertTrue(copied['imagination']['synthetic'])
