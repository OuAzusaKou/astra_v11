"""V6 geometry and exact-candidate control tests, no ROS/hardware."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from PIL import Image
from astra.pointcloud_geometry import HandKinematics,render_reviews
from astra.pointcloud import PointcloudImagination
from astra.pointcloud_capture import choose_cloud,validate_frozen
from astra.control import Controller,Settings
from astra.session import Session,atomic_json
from test_astra_mode2 import Robot,Policy
from wam_direct.start_mac_vlm import leo_command


class GeometryTests(unittest.TestCase):
    def fixture(self):
        xml='<robot>'
        for side in ('left','right'):
            xml+=f'<joint name="{side}_hand_fixed" type="fixed"><parent link="{side}_link_ee"/><child link="{side}_hand_palm_link"/><origin xyz="0 0 0.1"/></joint>'
            xml+=f'<joint name="{side}_hand_finger1_joint1" type="revolute"><parent link="{side}_hand_palm_link"/><child link="{side}_hand_tip"/><axis xyz="0 0 1"/></joint>'
        xml+='</robot>'
        groups=[(side+'_hand_tip',np.array([[.1,0,0.]])) for side in ('left','right')]
        return HandKinematics(xml,groups)

    def test_ee_and_joint_motion(self):
        fk=self.fixture();a=np.zeros(52);p=fk.points(a)
        np.testing.assert_allclose(p,[[.1,0,.1],[.1,0,.1]])
        a[0]=.2;a[12]=np.pi/2
        np.testing.assert_allclose(fk.points(a),[[.2,.1,.1],[.1,0,.1]],atol=1e-8)
        a[12]=0;a[5]=np.pi/2
        np.testing.assert_allclose(fk.points(a)[0],[.2,.1,.1],atol=1e-8)

    def test_missing_chain_rejected(self):
        with self.assertRaises(ValueError):HandKinematics('<robot/>',[('unknown',np.zeros((1,3)))]).points(np.zeros(52))

    def test_review_images(self):
        with tempfile.TemporaryDirectory() as t:
            anchor=self.fixture().points(np.zeros(52));frames=np.stack([anchor,anchor+.01])
            files=render_reviews(anchor,np.full(anchor.shape,100,np.uint8),anchor,frames,Path(t))
            self.assertEqual(len(files),3)
            self.assertEqual(Image.open(files[0]).size,(960,540))


class CaptureTests(unittest.TestCase):
    def test_nearest_fresh_cloud(self):
        self.assertEqual(choose_cloud([(9.5,'a'),(10.2,'b')],10,10.3),(10.2,'b'))
        with self.assertRaises(RuntimeError):choose_cloud([(5.,'old')],10,10.3)
        with self.assertRaises(RuntimeError):choose_cloud([(15.,'new')],10,15.1)

    def test_disk_and_worker_delay_do_not_age_frozen_input(self):
        data={'scene_timestamp':100.,'captured_at':100.2,'anchor_timestamp':100.1,
              'scene':np.zeros((100,3)),'tf_matrices':np.eye(4)[None]}
        # Validation is against capture time, even if rendering starts much later.
        validate_frozen(data,100.1)
        data['captured_at']=103.
        with self.assertRaises(ValueError):validate_frozen(data,100.1)
        data['captured_at']=100.2
        with self.assertRaises(ValueError):validate_frozen(data,101.)


class V6Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.robot=Robot();self.session=Session(Path(self.tmp.name)/'run','v6',{})
        self.c=Controller(self.robot,Policy(),self.session,Settings(400,version=6),'task',initial_l2='pick')
        self.c.imagination=PointcloudImagination()
        self.c.imagination.capture=Mock();self.c.imagination.capture.freeze.return_value={"fixture":np.zeros(1)}
        self.c.start();self.c.keyboard_toggle()
        self.process=Mock();self.process.poll.return_value=None
        self.patch=patch('astra.pointcloud.subprocess.Popen',return_value=self.process);self.patch.start();self.addCleanup(self.patch.stop)
        self.addCleanup(self.c.imagination.cancel)

    def decision(self,**kw):
        return {**{k:self.session.pending[k] for k in ('session_id','owner','request_id')},'reason':'test','evidence':'test',**kw}

    def imagine(self):
        self.c.decide(self.decision(operation='takeover',actions=np.zeros((30,52)).tolist(),new_attempt=True,change_reason='test',execute_steps=30))
        self.assertEqual(self.c.state,'IMAGINING');self.assertFalse(self.robot.published)
        p=self.c.imagination.pending;d=self.c.imagination.directory
        f=d/'top_XY.png';Image.new('RGB',(12,12)).save(f)
        atomic_json(d/'result.json',{'candidate_id':p['id'],'candidate_sha256':p['sha256'],'backend':'urdf_pointcloud',
                                    'images':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}]})
        self.c.tick()

    def test_review_approve_executes_exact_prefix(self):
        self.imagine();info=self.session.pending['imagination']
        self.assertEqual(self.session.pending['stage'],'REVIEW_IMAGINATION');self.assertEqual(len(info['actions']),18)
        self.assertTrue((self.c.imagination.directory/'metadata.json').exists())
        self.c.decide(self.decision(operation='approve',execute_steps=18,candidate_id=info['candidate_id'],candidate_sha256=info['candidate_sha256']))
        self.assertEqual(len(self.robot.published),1);self.assertEqual(self.c.steps,18)
        self.assertEqual(self.session.pending['stage'],'VERIFY_RECOVERY')
        self.assertTrue((self.c.imagination.directory/'cancel').exists())

    def test_reject_wrong_id(self):
        self.imagine();info=self.session.pending['imagination']
        with self.assertRaises(ValueError):self.c.decide(self.decision(operation='approve',execute_steps=18,candidate_id='wrong',candidate_sha256=info['candidate_sha256']))
        self.assertFalse(self.robot.published)

    def test_cancel_clears_candidate(self):
        self.imagine();self.c.keyboard_toggle();self.assertIsNone(self.c.imagination.pending)
        self.assertTrue((self.c.imagination.directory/'cancel').exists());self.assertFalse(self.robot.published)

    def test_failed_worker_pauses(self):
        self.c.queue=np.zeros((18,52));self.c.execute(18)
        self.process.poll.return_value=1;self.c.tick()
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR');self.assertFalse(self.robot.published)

    def test_v6_launcher_does_not_need_dynamics(self):
        cmd=leo_command('test','Task: a Subtask: b',[],version=6,auto_takeover=True)
        self.assertIn('version6_pointcloud.json',cmd);self.assertIn('--pointcloud-mode review',cmd)
        self.assertNotIn('--dynamics',cmd)

if __name__=='__main__':unittest.main()
