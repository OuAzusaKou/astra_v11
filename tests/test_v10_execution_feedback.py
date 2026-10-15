"""Offline execution-camera alignment and transport; no robot or model connections."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock

import numpy as np

if os.name == 'nt':
    sys.modules.setdefault('fcntl', Mock())
from astra.action_chunk import public_input
from astra.chunk_supervisor import ChunkApiSupervisor
from astra.control import Controller, Settings
from astra.execution_feedback import CAMERAS, ExecutionCapture, sample_indices
from astra.mac_bridge import materialize
from astra.remote_io import handle
from astra.session import Session
from test_astra_mode2 import Robot, Policy, observation


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.session = Session(Path(self.temp.name)/'run', 'fixture',
                               {'settings': {'supervisor_backend': 'mac'}})
        self.actions = np.full((30, 52), .01)
        self.offsets = .1+(np.arange(30)+1)/20
        self.receipt = {'start_stamp': [100, 0], 'time_from_start': self.offsets.tolist(), 'published_rows': 30}
        self.meta = {'chunk_id': 'fixture-chunk', 'source_request_id': 'source-decision',
                     'actions': self.actions.tolist(), 'publication_status': 'published',
                     'conditioning_version': 1, 'l2_instruction': 'pick up pen'}

    def capture(self, shift=0.):
        frames = {camera: [{'timestamp': 100+self.offsets[row]+shift,
                            'raw_recv_timestamp': 100+self.offsets[row]+shift+.1,
                            'image': np.full((8, 12, 3), row, np.uint8)} for row in sample_indices(30)]
                  for camera in CAMERAS}
        robot = SimpleNamespace(execution_camera_frames=lambda: frames)
        capture = ExecutionCapture(robot, self.receipt, self.actions, self.meta)
        capture.poll()
        return capture, frames

    def test_uniform_five_samples_and_action_mapping(self):
        self.assertEqual(sample_indices(30), [0, 7, 14, 22, 29])
        self.assertEqual(sample_indices(3), [0, 1, 2])
        self.assertEqual(sample_indices(1), [0])
        capture, frames = self.capture()
        record = capture.finish(self.session, drained=True)
        self.assertEqual(record['capture_status'], 'complete')
        self.assertFalse(record['physical_execution_confirmed'])
        self.assertEqual(record['source_request_id'], 'source-decision')
        self.assertEqual([s['action_row'] for s in record['samples']], [1, 8, 15, 23, 30])
        for sample in record['samples']:
            self.assertEqual(sample['action_target'], self.actions[sample['action_index']].tolist())
            self.assertEqual(set(sample['images']), set(CAMERAS))
            for image in sample['images'].values():
                self.assertEqual(image['nearest_action_row'], sample['action_row'])
                self.assertAlmostEqual(image['delta_to_scheduled_s'], 0.)
                self.assertTrue(Path(image['path']).exists())

    def test_actual_timestamp_can_map_to_adjacent_row(self):
        capture, _ = self.capture(shift=.04)
        record = capture.finish(self.session, drained=True)
        image = record['samples'][0]['images']['image']
        self.assertEqual(record['samples'][0]['action_row'], 1)
        self.assertEqual(image['nearest_action_row'], 2)
        self.assertAlmostEqual(image['delta_to_scheduled_s'], .04)

    def test_sampler_collects_during_motion_from_short_rolling_buffers(self):
        started = time.monotonic()
        def frames():
            # Like the live camera: 30 Hz, 100 ms latency, only 10 buffered frames.
            latest = int((time.monotonic()-started-.1)*30)
            return {c: [{'timestamp': 100+i/30, 'image': np.full((8, 12, 3), i % 255, np.uint8)}
                        for i in range(max(0, latest-9), max(0, latest+1))] for c in CAMERAS}
        capture = ExecutionCapture(SimpleNamespace(execution_camera_frames=frames),
                                   self.receipt, self.actions, self.meta).start()
        capture.thread.join(timeout=3.)
        record = capture.finish(self.session, drained=True)
        self.assertEqual(record['capture_status'], 'complete')
        self.assertEqual(sum(len(s['images']) for s in record['samples']), 15)
        for sample in record['samples']:
            for image in sample['images'].values():
                self.assertLessEqual(abs(image['delta_to_scheduled_s']), 1/30+.001)

    def test_cancelled_sampler_does_not_read_frames(self):
        read = Mock()
        capture = ExecutionCapture(SimpleNamespace(execution_camera_frames=read),
                                   self.receipt, self.actions, self.meta, cancelled=lambda: True).start()
        capture.thread.join(timeout=1.)
        record = capture.finish(self.session, drained=False)
        read.assert_not_called()
        self.assertEqual(record['capture_status'], 'unavailable')

    def test_stale_or_missing_frames_not_replaced_by_post_execution_images(self):
        robot = SimpleNamespace(execution_camera_frames=lambda: {
            c: [{'timestamp': 99., 'image': np.zeros((8, 12, 3), np.uint8)},
                {'timestamp': 105., 'image': np.zeros((8, 12, 3), np.uint8)}] for c in CAMERAS})
        capture = ExecutionCapture(robot, self.receipt, self.actions, self.meta)
        capture.poll();record = capture.finish(self.session, drained=False)
        self.assertEqual(record['capture_status'], 'unavailable')
        self.assertFalse(record['drain_completed'])
        for sample in record['samples']:
            self.assertFalse(sample['images'])
            self.assertEqual(sample['missing_cameras'], list(CAMERAS))

    def test_camera_failure_does_not_publish_or_fake_frames(self):
        robot = SimpleNamespace(execution_camera_frames=Mock(side_effect=RuntimeError('sensor offline')))
        capture = ExecutionCapture(robot, self.receipt, self.actions, self.meta).start()
        capture.thread.join(timeout=1.)
        record = capture.finish(self.session, drained=True)
        self.assertEqual(record['capture_status'], 'unavailable')
        self.assertIn('sensor offline', record['capture_error'])

    def test_controller_links_executed_chunk_to_next_request_without_wam(self):
        robot = Robot();c = Controller(robot, Policy(), self.session, Settings(100, version=10), 'task', 'pick pen')
        c.start();c.keyboard_toggle();request = self.session.pending
        identity = {k: request[k] for k in ('session_id', 'owner', 'request_id')}
        c.decide({**identity, 'operation': 'v10_action_chunk', 'chunk_reply': {'action_chunk': self.actions.tolist()}})
        np.testing.assert_array_equal(robot.published[0], self.actions)
        next_request = self.session.pending
        self.assertEqual(next_request['execution_feedback']['source_request_id'], request['request_id'])
        self.assertEqual(next_request['execution_feedback']['actions'], self.actions.tolist())
        self.assertEqual(next_request['execution_feedback']['capture_status'], 'unavailable')
        self.assertEqual(next_request['wam_context'], {})
        c.operator_reset()
        self.assertIsNone(c.latest_execution)

    def test_execution_images_export_materialize_and_public_context(self):
        capture, _ = self.capture();record = capture.finish(self.session, drained=True)
        obs = self.session.snapshot(observation(), 'current')
        request = self.session.request(version=10, instruction='task', l2_instruction='pick pen',
            conditioning_version=1, observation=obs, execution_feedback=record,
            wam_context={'trajectories': [{'observation': {'images': {'bad': {'path': 'WAM_SECRET'}}}}]})
        self.session.status(state='WAITING')
        bundle = handle(self.session.path, {'client_id': str(uuid.uuid4()), 'operation': 'request',
                                          'request_id': request['request_id']})
        self.assertNotIn('wam_context', bundle['request'])
        local = materialize(bundle, Path(self.temp.name)/'cache')
        context, images = public_input(local)
        self.assertNotIn('policy_history', context)
        self.assertEqual(len(images), 18)
        self.assertNotIn('WAM_SECRET', json.dumps(context))
        self.assertNotIn('path', json.dumps(context))
        for image in images: self.assertTrue(Path(image).exists())
        feedback = context['execution_feedback']
        self.assertTrue(feedback['matches_current_condition'])
        self.assertEqual(feedback['source_request_id'], 'source-decision')
        for sample in feedback['samples']:
            for view in sample['views'].values(): self.assertIn(view['image_label'], context['image_order'])
        # Only the most recent execution sequence is attached; full archives remain intact.
        bridge = object.__new__(ChunkApiSupervisor)
        content = [{'type': 'text', 'text': json.dumps(context)}] + [
            {'type': 'image_url', 'image_url': {'url': p}} for p in images]
        bridge.data_history = [{'role': 'user', 'content': content},
                               {'role': 'assistant', 'content': '{"action_chunk":[]}'}]
        original = copy.deepcopy(bridge.data_history)
        history = bridge.recent_history()
        self.assertEqual(len(history[0]['content']), 4)
        self.assertNotIn('execution_feedback', json.loads(history[0]['content'][0]['text']))
        self.assertEqual(bridge.data_history, original)


if __name__ == '__main__':
    unittest.main()
