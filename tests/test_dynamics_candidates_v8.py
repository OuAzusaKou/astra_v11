"""V8 contracts with fake robot/policy/Dynamics. Never publishes to ROS."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

from astra.control import Controller, Settings
from astra.session import Session
from astra.dynamics_candidates import CandidateImagination, directional_candidates, prefix_steps
from test_astra_mode2 import Robot, Policy, observation
from test_astra_dynamics import result
from wam_direct.start_mac_vlm import leo_command


class V8Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.robot = Robot(); self.policy = Policy()
        self.session = Session(Path(self.tmp.name)/'run', 'v8', {})
        self.c = Controller(self.robot, self.policy, self.session, Settings(400, version=8),
                            'open drawer', initial_l2='pull out the drawer')
        self.c.imagination = CandidateImagination(mode='review')
        self.c.start(); self.c.keyboard_toggle()

    def decision(self, **kwargs):
        return {**{k:self.session.pending[k] for k in ('session_id','owner','request_id')},
                'reason':'test', 'evidence':'real fixture evidence', **kwargs}

    def imagine(self, hand='left', response=result):
        published, steps = len(self.robot.published), self.c.steps
        with patch('astra.dynamics_candidates.native_request', return_value={}) as native, \
             patch('astra.dynamics_candidates.rpc', side_effect=response) as rpc:
            self.c.decide(self.decision(operation='propose_candidates', active_hand=hand))
            self.c.imagination.worker.join(3)
            self.assertFalse(self.c.imagination.worker.is_alive())
            self.assertEqual(self.c.state, 'IMAGINING')
            self.assertEqual(len(self.robot.published), published)
            self.assertEqual(self.c.steps, steps)
            self.c.tick()
            return native, rpc

    def approval(self, label='left_plus_x', **overrides):
        info = self.session.pending['imagination']
        selected = next(c for c in info['candidates'] if c['label'] == label)
        args = dict(operation='approve', candidate_set_id=info['candidate_set_id'],
                    candidate_id=selected['candidate_id'], candidate_sha256=selected['candidate_sha256'],
                    execute_steps=len(selected['actions']))
        args.update(overrides)
        return self.decision(**args)

    def test_both_hands_six_directions_only_selected_xyz_changes(self):
        obs = observation()
        keys = ('left_ee_pose','right_ee_pose','left_hand_joint','right_hand_joint')
        for index,key in enumerate(keys):
            obs[key][:] = (index+1)*.1
        initial = np.concatenate([obs[key][0] for key in keys])
        for hand, offset in (('left',0),('right',6)):
            rows = list(directional_candidates(obs,hand,18,self.c.imagination.distance))
            self.assertEqual(len(rows),6)
            for i,(label,actions) in enumerate(rows):
                coordinate = offset+i//2; sign = 1 if i%2==0 else -1
                self.assertTrue(label.startswith(hand))
                self.assertAlmostEqual(actions[-1,coordinate]-initial[coordinate],sign*.10)
                self.assertTrue((np.diff(sign*actions[:,coordinate])>0).all())
                np.testing.assert_array_equal(np.delete(actions,coordinate,axis=1),
                    np.repeat(np.delete(initial,coordinate)[None,:],18,axis=0))
        for hand, distance in (('both',.05),('left',0),('right',float('nan')),('left',.11)):
            with self.assertRaises(ValueError):list(directional_candidates(obs,hand,18,distance))
        self.assertEqual(prefix_steps(30,20,.1),18)
        with self.assertRaises(ValueError):prefix_steps(1,20,.1)

    def test_same_real_anchor_fresh_policy_and_exact_selection(self):
        native, rpc = self.imagine('right')
        self.assertEqual(len(self.policy.calls),1)
        self.assertEqual(self.policy.calls[0],self.c.prompt)
        self.assertEqual(rpc.call_count,7)
        calls = native.call_args_list
        self.assertTrue(all(a.args[0] is calls[0].args[0] for a in calls))
        self.assertTrue(all(a.args[1].shape == (18,52) for a in calls))
        self.assertEqual(len({a.args[4] for a in calls}),7)
        self.assertEqual(self.session.pending['stage'],'REVIEW_CANDIDATES')
        d=self.approval('right_minus_z'); self.c.decide(d)
        self.assertEqual(self.c.steps,18); self.assertEqual(len(self.robot.published),1)
        self.assertAlmostEqual(self.robot.published[0][-1,8],-.10)
        self.assertIsNone(self.c.imagination.pending)
        self.assertEqual(self.session.pending['stage'],'VERIFY_RECOVERY')
        with self.assertRaises(ValueError):self.c.decide(d)
        self.imagine('left')
        self.assertEqual(len(self.policy.calls),2)

    def test_policy_is_unmodified_bounded_candidate(self):
        self.imagine(); self.c.decide(self.approval('policy'))
        np.testing.assert_array_equal(self.robot.published[0], np.full((18,52),.01))

    def test_no_bypass_raw_actions_or_unreviewed_execute(self):
        for op in ('takeover','correct','propose_candidates'):
            with self.assertRaises(ValueError):
                self.c.decide(self.decision(operation=op,active_hand='left',actions=np.zeros((18,52)).tolist()))
        self.c.queue=np.zeros((18,52))
        with self.assertRaises(ValueError):self.c.execute(18)
        with self.assertRaises(ValueError):self.c.execute(18,_imagined=True)
        self.assertFalse(self.robot.published)

    def test_wrong_binding_modified_actions_and_partial_prefix_rejected(self):
        self.imagine()
        for change in ({'candidate_set_id':'old'}, {'candidate_id':'bad'}, {'candidate_sha256':'bad'},
                       {'execute_steps':17}, {'execute_steps':True}, {'actions':[]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.c.decide(self.approval(**change))
        self.c.imagination.pending['candidates'][0]['actions'][0,0]+=.001
        with self.assertRaises(ValueError):self.c.decide(self.approval())
        self.assertFalse(self.robot.published)

    def test_changed_anchor_clock_budget_hardware_block(self):
        self.imagine(); d=self.approval()
        changed=observation(); changed['right_ee_pose'][0,0]=.1
        with patch.object(self.robot,'observe',return_value=changed), self.assertRaises(RuntimeError):self.c.decide(d)
        self.robot.fps=30
        with self.assertRaises(ValueError):self.c.decide(d)
        self.robot.fps=20; self.c.steps=self.c.settings.budget-1
        with self.assertRaises(ValueError):self.c.decide(d)
        self.c.steps=0; self.robot.fault=True
        with self.assertRaises(ValueError):self.c.decide(d)
        self.assertFalse(self.robot.published)

    def test_partial_failure_cannot_select_failed_candidate(self):
        self.imagine(response=[TimeoutError('fixture')] + [result() for _ in range(6)])
        self.assertEqual(self.session.pending['imagination']['candidates'][0]['status'],'failed')
        with self.assertRaises(ValueError):self.c.decide(self.approval())
        self.c.decide(self.approval('policy'))
        self.assertEqual(len(self.robot.published),1)

    def test_all_failures_pause_and_save_failure_metadata(self):
        self.imagine(response=TimeoutError('fixture'))
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')
        self.assertEqual(len(list(self.session.path.glob('imagination/*/*/metadata.json'))),7)
        self.assertFalse(self.robot.published)

    def test_preview_artifacts_never_publish(self):
        self.c.imagination.mode='preview'; self.imagine()
        self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')
        self.assertEqual(len(list(self.session.path.glob('imagination/*/*/future.png'))),7)
        self.assertEqual(len(list(self.session.path.glob('imagination/*/panel.png'))),1)
        self.assertFalse(self.robot.published)

    def test_toggle_invalidates_entire_set_and_draining_worker(self):
        entered=threading.Event(); release=threading.Event()
        def slow(*args):
            entered.set(); release.wait(3); return result()
        with patch('astra.dynamics_candidates.native_request',return_value={}), \
             patch('astra.dynamics_candidates.rpc',side_effect=slow) as rpc:
            try:
                self.c.decide(self.decision(operation='propose_candidates',active_hand='left'))
                self.assertTrue(entered.wait(1))
                self.c.keyboard_toggle(); self.c.keyboard_toggle()
                with self.assertRaises(RuntimeError):
                    self.c.decide(self.decision(operation='propose_candidates',active_hand='right'))
            finally:
                release.set(); self.c.imagination.worker.join(3)
            self.assertEqual(rpc.call_count,1)
        self.assertIsNone(self.c.imagination.pending)
        self.assertIsNone(self.c.imagination.result)
        self.assertFalse(self.robot.published)

    def test_all_seven_images_survive_windows_bridge_transport(self):
        import uuid
        from astra.remote_io import handle
        from astra.mac_bridge import materialize
        self.imagine()
        mf=self.session.path/'manifest.json'; manifest=json.loads(mf.read_text())
        manifest['settings']={'supervisor_backend':'mac'}; mf.write_text(json.dumps(manifest))
        req=self.session.pending
        bundle=handle(self.session.path,{'client_id':str(uuid.uuid4()),'operation':'request','request_id':req['request_id']})
        copied=materialize(bundle,Path(self.tmp.name)/'windows')
        images=copied['imagination']['images']; self.assertEqual(len(images),7)
        for im in images:
            self.assertTrue(Path(im['path']).is_file())
            self.assertIn(im['candidate_id'], {c['candidate_id'] for c in copied['imagination']['candidates']})

    def test_supervisor_receives_hand_choice_and_labeled_candidates(self):
        from astra.supervisor import CodexSupervisor
        self.imagine(); prompts=[]
        supervisor=CodexSupervisor(self.session)
        def capture(prompt,images,*args):
            prompts.append((prompt,images)); return self.approval('policy')
        with patch.object(supervisor,'_call',side_effect=capture):
            supervisor.start(self.session.pending)
            supervisor.worker.join(3)
        self.assertEqual(len(prompts),1)
        self.assertIn('operation=propose_candidates',prompts[0][0])
        self.assertIn('candidate_set_id',prompts[0][0])
        for c in self.session.pending['imagination']['candidates']:
            self.assertIn('candidate='+c['candidate_id']+' '+c['label'],prompts[0][0])

    def test_launcher_dynamics_and_vlm_flags(self):
        command=leo_command('v8test',self.c.prompt,[],version=8,auto_takeover=True,dynamics_mode='review',candidate_distance_m=.03)
        for flag in ('version8_dynamics_candidates.json','--dynamics-view','--dynamics-mode review',
                     '--vlm-auto-takeover','--candidate-distance-m 0.03'):
            self.assertIn(flag,command)
        self.assertNotIn('--rviz-record',command)


if __name__=='__main__':unittest.main()
