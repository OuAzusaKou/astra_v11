import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astra.video_context import prepare_reference_video
from astra.supervisor import CodexSupervisor
from astra.session import atomic_json
import test_astra_remote as remote_tests


class VideoTransportTests(unittest.TestCase):
    setUp = remote_tests.RemoteTests.setUp
    call = remote_tests.RemoteTests.call
    request = remote_tests.RemoteTests.request
    ready = remote_tests.RemoteTests.ready

    def test_video_reaches_mac_and_supervisor(self):
        frame = self.s.path / 'demo.jpg'
        frame.write_bytes(b'demonstration fixture')
        self.s.reference_video = {'images': [{'path': str(frame),
            'sha256': hashlib.sha256(frame.read_bytes()).hexdigest(), 'frame_index': 7, 'time_seconds': .7}]}
        self.ready()
        req = self.request()
        future = self.s.path / 'candidate.jpg'
        future.write_bytes(b'synthetic future fixture')
        req['imagination'] = {'images': [{'path': str(future),
            'sha256': hashlib.sha256(future.read_bytes()).hexdigest(),
            'candidate_id': 'left_x_plus', 'label': 'predicted future'}]}
        atomic_json(self.s.path/'request.json', req)
        from astra.mac_bridge import materialize
        bundle = self.call('request', request_id=req['request_id'])
        copied = materialize(bundle, Path(self.temp.name)/'cache')
        image = copied['reference_video']['images'][0]
        self.assertEqual(Path(image['path']).read_bytes(), frame.read_bytes())
        self.assertEqual(image['remote_path'], str(frame))
        # Same request path feeds local and Mac supervisors; no Codex/API call.
        variants = []
        for version in (4, 8):
            for stage in ('ASTRA_TAKEOVER', 'VERIFY_RECOVERY', 'REVIEW_CANDIDATES'):
                for request in (req, copied):
                    variants.append({**request, 'version': version, 'stage': stage})
        for request in variants:
            supervisor = CodexSupervisor.__new__(CodexSupervisor)
            supervisor.worker = None
            supervisor.generation = 0
            supervisor.log = lambda *_: None
            calls = []
            def fake_call(prompt, images, *args):
                calls.append((prompt, images))
                return {**{k:request[k] for k in ('session_id','owner','request_id')},
                        'operation':'observe', 'reason':'fixture', 'evidence':'fixture'}
            supervisor._call = fake_call
            supervisor.start(request)
            supervisor.worker.join(2)
            self.assertIsNone(supervisor.error)
            self.assertIn(request['reference_video']['images'][0]['path'], calls[0][1])
            self.assertIn('DEMONSTRATION frame=7 time_seconds=0.7', calls[0][0])
            self.assertIn('not live observations or WAM predictions', calls[0][0])
            self.assertEqual(calls[0][1][-1], request['reference_video']['images'][0]['path'])
            self.assertEqual(calls[0][1][-2], request['imagination']['images'][0]['path'])
            self.assertEqual(Path(calls[0][1][-2]).read_bytes(), future.read_bytes())
            self.assertIn('SYNTHETIC, NOT OBSERVED / candidate=left_x_plus', calls[0][0])
        # New request IDs (including after a manual L2 change) retain the demo.
        later = self.s.request(stage='ASTRA_TAKEOVER', version=4, l2_instruction='close notebook')
        self.assertNotEqual(later['request_id'], req['request_id'])
        self.assertEqual(later['reference_video'], req['reference_video'])
        bundle['images'][str(frame)] = 'YQ=='
        with self.assertRaisesRegex(ValueError, 'checksum'):
            materialize(bundle, Path(self.temp.name)/'cache')


class VideoExtractionTests(unittest.TestCase):
    def test_uniform_frames_and_metadata(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV extraction is tested on the runner host")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'demo.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64,48))
            self.assertTrue(writer.isOpened())
            for i in range(10):
                writer.write(np.full((48,64,3), i*20, dtype=np.uint8))
            writer.release()
            result = prepare_reference_video(source, Path(tmp)/'frames', 3)
            self.assertEqual([f['frame_index'] for f in result['images']], [0,4,9])
            self.assertEqual([f['time_seconds'] for f in result['images']], [0,.4,.9])
            self.assertEqual(result['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            for frame in result['images']:
                self.assertIsNotNone(cv2.imread(frame['path']))
            with self.assertRaises(ValueError):
                prepare_reference_video(source, Path(tmp)/'bad', 33)
