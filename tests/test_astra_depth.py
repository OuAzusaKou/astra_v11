"""Offline depth contracts: data fidelity, evidence quality, request transport and opt-in."""
import base64
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np
from PIL import Image

from astra.depth_camera import DepthCamera, save_depth_observation
from astra.depth_query import query
from astra.mac_bridge import materialize
from astra.remote_io import handle
from astra.session import Session
from astra.supervisor import CodexSupervisor
from test_astra_mode2 import observation, Robot, Policy
from astra.control import Controller, Settings


def frame():
    raw = np.full((12, 16), 700, dtype=np.uint16)
    raw[:, :8] = 500
    raw[0, :8] = 0
    return {"raw": raw, "rgb": np.full((12, 16, 3), [24, 48, 96], dtype=np.uint8),
            "depth_scale_m": .001, "camera_serial": "synthetic", "frame_number": 91,
            "host_received_at": time.time(), "sensor_timestamp_ms": 1234.,
            "sensor_timestamp_domain": "hardware_clock", "color_depth_skew_ms": 5.}


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = Session(self.root / 'session', 'test', {'settings': {'supervisor_backend': 'mac'}})
        self.data = frame()
        self.session.depth_source = SimpleNamespace(latest=lambda: self.data)

    def snapshot(self):
        obs = observation()
        obs['timestamp'] = self.data['host_received_at']
        return self.session.snapshot(obs, 'depth', include_depth=True)

    def test_lossless_raw_and_rgb_depth_roi_correspondence(self):
        snap = self.snapshot()
        with Image.open(snap['depth']['raw']['path']) as image:
            np.testing.assert_array_equal(np.asarray(image), self.data['raw'])
        with Image.open(snap['images']['realsense_rgb']['path']) as image:
            np.testing.assert_array_equal(np.asarray(image), self.data['rgb'])
        result = query(snap['depth']['metadata_path'], [(0, 0, 8, 12), (8, 0, 16, 12)])
        self.assertAlmostEqual(result['regions'][0]['median_m'], .5)
        self.assertAlmostEqual(result['regions'][0]['valid_fraction'], 11 / 12)
        self.assertEqual(result['comparison']['ordering'], 'region_a_nearer_camera')
        self.assertNotIn('xyz', result)

    def test_zeros_do_not_become_near_surface(self):
        self.data['raw'][:, :8] = 0
        result = query(self.snapshot()['depth']['metadata_path'], [(0, 0, 8, 12), (8, 0, 16, 12)])
        self.assertFalse(result['regions'][0]['usable'])
        self.assertNotIn('median_m', result['regions'][0])
        self.assertEqual(result['comparison']['ordering'], 'uncertain')

    def test_mixed_foreground_background_is_not_confident_order(self):
        self.data['raw'][:, :4] = 200
        result = query(self.snapshot()['depth']['metadata_path'], [(0, 0, 8, 12), (8, 0, 16, 12)])
        self.assertTrue(result['regions'][0]['mixed_depth_warning'])
        self.assertEqual(result['comparison']['ordering'], 'uncertain')

    def test_roi_bounds_not_silently_clamped(self):
        path = self.snapshot()['depth']['metadata_path']
        for roi in [(-1, 0, 8, 12), (0, 0, 17, 12), (5, 5, 5, 8)]:
            with self.assertRaises(ValueError):
                query(path, [roi])

    def test_stale_frame_and_sync_mismatch_are_unavailable(self):
        self.data['host_received_at'] -= 3
        snap = self.snapshot()
        self.assertEqual(snap['depth']['status'], 'unavailable')
        self.assertGreater(snap['depth']['age_at_snapshot_s'], 2.9)
        self.assertNotIn('realsense_rgb', snap['images'])
        self.data = frame()
        self.data['color_depth_skew_ms'] = 99
        self.assertEqual(self.snapshot()['depth']['status'], 'unavailable')
        self.data = frame()
        obs = observation()
        obs['timestamp'] = self.data['host_received_at'] - 2
        snap = self.session.snapshot(obs, 'skew', include_depth=True)
        self.assertEqual(snap['depth']['status'], 'unavailable')
        self.assertEqual(snap['depth']['robot_observation_time_delta_s'], 2.)
        self.assertEqual(snap['depth']['robot_observation_timestamp'], obs['timestamp'])

    def test_optional_depth_does_not_modify_policy_snapshot(self):
        snap = self.session.snapshot(observation(), 'inference')
        self.assertNotIn('depth', snap)
        self.assertEqual(set(snap['images']), {'image', 'image_wrist_left', 'image_wrist_right'})

    def test_rgb_encoding_delay_does_not_change_depth_sampling_time(self):
        # Real runs spend >0.5 s encoding the three 1280x720 RGB images.
        # A continuous camera advances while saving; pin depth before that work.
        clock = [self.data['host_received_at']]
        obs = observation()
        obs['timestamp'] = clock[0] - .15
        sampled_at = []
        def latest():
            sampled_at.append(clock[0])
            return {**self.data, 'host_received_at': clock[0] - .02}
        self.session.depth_source = SimpleNamespace(latest=latest)
        original_save = Image.Image.save
        def delayed_save(image, path, *args, **kwargs):
            if Path(path).name in ('image.png', 'image_wrist_left.png', 'image_wrist_right.png'):
                clock[0] += .3
            return original_save(image, path, *args, **kwargs)
        with patch('time.time', side_effect=lambda: clock[0]), patch.object(Image.Image, 'save', delayed_save):
            snap = self.session.snapshot(obs, 'slow_rgb', include_depth=True)
        self.assertEqual(snap['depth']['status'], 'available')
        self.assertEqual(len(sampled_at), 1)
        self.assertAlmostEqual(snap['depth']['robot_observation_time_delta_s'], .13, places=5)
        self.assertAlmostEqual(snap['depth']['age_at_snapshot_s'], .02, places=5)
        self.assertGreater(clock[0] - sampled_at[0], .8)

    def test_camera_waits_for_paired_frames_before_ready(self):
        data = self.data
        class SensorFrame:
            def __init__(self, color, skew):
                self.color, self.skew = color, skew
            def get_frame_timestamp_domain(self):
                return 'hardware_clock'
            def get_timestamp(self):
                return 1000. + (self.skew if self.color else 0.)
            def get_data(self):
                return data['rgb' if self.color else 'raw']
            def get_frame_number(self):
                return 1
        def pair(skew):
            return SimpleNamespace(get_depth_frame=lambda: SensorFrame(False, skew),
                                   get_color_frame=lambda: SensorFrame(True, skew))
        camera = DepthCamera.__new__(DepthCamera)
        camera.rs = SimpleNamespace(stream=SimpleNamespace(color='color'),
                                    align=lambda _: SimpleNamespace(process=lambda frame: frame))
        camera.pipeline = SimpleNamespace(wait_for_frames=Mock(side_effect=[pair(130.), pair(5.)]))
        camera.stop = SimpleNamespace(is_set=Mock(side_effect=[False, False, True]))
        camera.ready = SimpleNamespace(set=Mock())
        camera.lock = threading.Lock()
        camera.frame, camera.error, camera.serial, camera.scale = None, None, 'synthetic', .001
        camera._read()
        self.assertIsNone(camera.error)
        self.assertEqual(camera.frame['color_depth_skew_ms'], 5.)
        camera.ready.set.assert_called_once()

    def test_depth_failure_keeps_rgb_observations(self):
        def broken():
            raise RuntimeError('camera disconnected')
        self.session.depth_source = SimpleNamespace(latest=broken)
        snap = self.snapshot()
        self.assertEqual(len(snap['images']), 3)
        self.assertEqual(snap['depth']['status'], 'unavailable')
        with self.assertRaises(ValueError):
            query(snap['depth']['metadata_path'], [(0, 0, 8, 12)])

    def bundle(self):
        request = self.session.request(stage='ASTRA_TAKEOVER', observation=self.snapshot(), wam_context={'trajectories': []})
        self.session.status(state='WAITING')
        payload = {'client_id': str(uuid.uuid4()), 'operation': 'request', 'request_id': request['request_id']}
        return handle(self.session.path, payload)

    def test_remote_roundtrip_preserves_16bit_and_rewrites_query_paths(self):
        copied = materialize(self.bundle(), self.root / 'mac-cache')
        depth = copied['observation']['depth']
        self.assertNotEqual(depth['raw']['path'], depth['raw']['remote_path'])
        self.assertNotEqual(depth['metadata_path'], depth['remote_metadata_path'])
        with Image.open(depth['raw']['path']) as image:
            np.testing.assert_array_equal(np.asarray(image), self.data['raw'])
        result = query(depth['metadata_path'], [(0, 1, 8, 12)])
        self.assertAlmostEqual(result['regions'][0]['median_m'], .5)

    def test_transport_rejects_corrupt_raw_depth(self):
        bundle = self.bundle()
        source = bundle['request']['observation']['depth']['raw']['path']
        bundle['images'][source] = base64.b64encode(b'corrupt').decode()
        with self.assertRaisesRegex(ValueError, 'depth checksum'):
            materialize(bundle, self.root / 'mac-cache')

    def test_query_rejects_modified_depth_file(self):
        snap = self.snapshot()
        Path(snap['depth']['raw']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            query(snap['depth']['metadata_path'], [(0, 0, 8, 12)])

    def test_review_carries_depth_without_publishing_motion(self):
        robot = Robot()
        robot.observe = lambda: {**observation(), 'timestamp': self.data['host_received_at']}
        controller = Controller(robot, Policy(), self.session, Settings(400, version=2),
                                'open drawer', initial_l2='pull out drawer')
        controller.mode = 'ASTRA'
        controller.review('ASTRA_TAKEOVER')
        self.assertEqual(self.session.pending['observation']['depth']['status'], 'available')
        self.assertEqual(robot.published, [])

    def test_supervisor_gets_query_command_and_preview_not_raw_as_photo(self):
        request = self.session.request(stage='ASTRA_TAKEOVER', observation=self.snapshot())
        supervisor = CodexSupervisor.__new__(CodexSupervisor)
        supervisor.worker = None
        supervisor.generation = 0
        supervisor.log = lambda message: None
        received = {}
        def call(prompt, images, cancel, tag):
            received.update(prompt=prompt, images=images)
            return {**{key: request[key] for key in ('session_id', 'owner', 'request_id')},
                    'operation': 'observe', 'reason': 'fixture', 'evidence': 'fixture'}
        supervisor._call = call
        supervisor.start(request)
        supervisor.worker.join(3)
        self.assertIsNone(supervisor.error)
        self.assertIn('depth_query.py', received['prompt'])
        self.assertIn(request['observation']['depth']['metadata_path'], received['prompt'])
        self.assertEqual(len(received['images']), 5)
        self.assertNotIn(request['observation']['depth']['raw']['path'], received['images'])


if __name__ == '__main__':
    unittest.main()
