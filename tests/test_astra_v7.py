"""Offline V7 integration: fake robot and display worker; no hardware/network."""
import hashlib
import numpy as np
from PIL import Image
from astra.control import Settings
from astra.session import atomic_json
from astra.prefix_plan import prefix_limit, validate_stages
from test_astra_pointcloud import V6Tests
from wam_direct.start_mac_vlm import leo_command


class V7Tests(V6Tests):
    def setUp(self):
        super().setUp()
        self.c.settings=Settings(400,version=7)
        self.c.imagination.long_plan=True

    def imagine(self):
        actions=np.zeros((60,52));actions[:,0]=np.arange(1,61)*.001
        self.c.decide(self.decision(operation='takeover',actions=actions.tolist(),new_attempt=True,
            change_reason='test',execute_steps=60,short_batch_reason='next milestone',
            plan_stages=[{'kind':'transport','end_step':60,'evidence':'fixture grasp and clearance'}]))
        self.assertEqual(self.c.state,'IMAGINING');self.assertFalse(self.robot.published)
        p=self.c.imagination.pending;d=self.c.imagination.directory
        f=d/'top_XY.png';Image.new('RGB',(12,12)).save(f)
        atomic_json(d/'result.json',{'candidate_id':p['id'],'candidate_sha256':p['sha256'],
            'images':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}]})
        self.c.tick()

    def test_review_approve_executes_exact_prefix(self):
        self.imagine();info=self.session.pending['imagination']
        self.assertEqual(len(info['actions']),60)
        self.assertEqual(info['max_execute_steps'],28)
        self.c.decide(self.decision(operation='approve',execute_steps=28,
            candidate_id=info['candidate_id'],candidate_sha256=info['candidate_sha256']))
        self.assertEqual(self.c.steps,28);self.assertEqual(len(self.robot.published[0]),28)
        self.assertEqual(len(self.c.queue),0)
        self.assertEqual(self.session.pending['stage'],'VERIFY_RECOVERY')
        # Remaining rows cannot be replayed by another approval.
        with self.assertRaises(ValueError):self.c.decide(self.decision(operation='approve',execute_steps=28))
        self.c.decide(self.decision(operation='hand_back',recovery_evidence='fixture recovery'))
        self.assertEqual(self.c.mode,'WAM')

    def test_oversized_prefix_rejected(self):
        self.imagine();p=self.c.imagination.pending
        with self.assertRaises(ValueError):self.c.decide(self.decision(operation='approve',execute_steps=29,
            candidate_id=p['id'],candidate_sha256=p['sha256']))
        self.assertFalse(self.robot.published)

    def test_failed_worker_pauses(self):
        self.c.plan_stages=[{'kind':'approach','end_step':18,'evidence':'fixture'}]
        super().test_failed_worker_pauses()

    def test_bounds_and_stage_boundary(self):
        a=np.zeros((60,52));anchor=np.zeros(52)
        stages=[{'kind':'transport','end_step':5,'evidence':'test'},
                {'kind':'release','end_step':60,'evidence':'test'}]
        self.assertEqual(prefix_limit(a,anchor,stages,20,.1,100)['count'],5)
        stages=[{'kind':'approach','end_step':60,'evidence':'test'}]
        self.assertEqual(prefix_limit(a,anchor,stages,20,.1,100)['count'],8)
        self.assertEqual(prefix_limit(a,anchor,stages,20,.1,3)['count'],3)
        a[0,0]=.1
        with self.assertRaises(ValueError):prefix_limit(a,anchor,stages,20,.1,100)
        with self.assertRaises(ValueError):validate_stages(stages,59)
        with self.assertRaises(ValueError):validate_stages(None,60)

    def test_plan_mutation_rejected(self):
        self.imagine();p=self.c.imagination.pending;self.c.queue[0,0]+=.001
        with self.assertRaises(ValueError):self.c.decide(self.decision(operation='approve',execute_steps=10,
            candidate_id=p['id'],candidate_sha256=p['sha256']))
        self.assertFalse(self.robot.published)

    def test_v6_launcher_does_not_need_dynamics(self):
        cmd=leo_command('test','Task: a Subtask: b',[],version=7,auto_takeover=True)
        self.assertIn('version7_prefix.json',cmd);self.assertIn('--rviz-record',cmd)
        self.assertNotIn('--dynamics',cmd)

    def test_transport_cannot_release_and_distance_is_cumulative(self):
        a=np.zeros((60,52));anchor=np.zeros(52)
        stages=[{'kind':'transport','end_step':60,'evidence':'fixture'}]
        a[5:,12]=.06
        self.assertEqual(prefix_limit(a,anchor,stages,20,.1,100)['count'],5)
        a[:]=0;a[:,0]=np.arange(1,61)*.007
        self.assertEqual(prefix_limit(a,anchor,stages,20,.1,100)['count'],25)

    def test_short_approval_and_remaining_budget(self):
        self.c.steps=self.c.settings.budget-4
        self.imagine();p=self.c.imagination.pending
        self.assertEqual(p['prefix']['count'],4)
        self.c.decide(self.decision(operation='approve',execute_steps=2,
            candidate_id=p['id'],candidate_sha256=p['sha256']))
        self.assertEqual(len(self.robot.published[0]),2)
        self.assertEqual(len(self.c.queue),0)
