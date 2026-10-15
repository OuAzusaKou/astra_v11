"""Read-only camera sampling tied to published V10 waypoint timestamps."""
import copy
import hashlib
import math
import threading
import time

import numpy as np

from .session import atomic_json

CAMERAS = ('image', 'image_wrist_left', 'image_wrist_right')


def sample_indices(count):
    if count <= 0:
        return []
    return sorted({round(i * (count - 1) / (min(5, count) - 1))
                   for i in range(min(5, count))}) if count > 1 else [0]


class ExecutionCapture:
    """Sample camera buffers without calling get_obs or changing the command schedule.

    Camera timestamp is the environment's latency-adjusted timestamp, in the same
    ROS time domain as start_stamp. A timestamp match is not execution success.
    Image encoding happens only after drain, outside the sampling thread.
    """
    max_skew = .075

    def __init__(self, robot, receipt, actions, record, cancelled=lambda: False):
        self.robot, self.cancelled = robot, cancelled
        self.record = copy.deepcopy(record)
        self.stop = threading.Event()
        self.thread = None
        self.selected = {}
        self.error = None
        self.record.update(receipt=copy.deepcopy(receipt), samples=[], capture_status='pending')
        self.actions = np.asarray(actions)
        self.rows = sample_indices(len(self.actions))
        self.times = None
        try:
            stamp = receipt['start_stamp']
            offsets = np.asarray(receipt['time_from_start'], dtype=float)
            if len(offsets) != len(self.actions) or not np.isfinite(offsets).all() or np.any(np.diff(offsets) <= 0):
                raise ValueError('invalid publication schedule')
            self.origin = float(stamp[0]) + float(stamp[1]) / 1e9
            if not math.isfinite(self.origin):
                raise ValueError('invalid publication clock')
            self.times = self.origin + offsets
            self.deadline = time.monotonic() + max(0., float(offsets[-1])) + 1.
            if not callable(getattr(robot, 'execution_camera_frames', None)):
                raise ValueError('camera execution sampling unavailable')
        except Exception as error:
            self.error = str(error)

    def poll(self):
        frames = self.robot.execution_camera_frames()
        for camera in CAMERAS:
            available = [f for f in frames.get(camera, [])
                         if math.isfinite(float(f['timestamp']))
                         and self.origin <= float(f['timestamp']) <= self.times[-1] + self.max_skew]
            if not available:
                continue
            latest = max(float(f['timestamp']) for f in available)
            for row in self.rows:
                key = (row, camera)
                if key in self.selected or latest < self.times[row]:
                    continue
                frame = min(available, key=lambda f: abs(float(f['timestamp']) - self.times[row]))
                stamp = float(frame['timestamp'])
                if abs(stamp - self.times[row]) > self.max_skew:
                    continue
                # Do not present one stale image as several distinct time samples.
                if any(c == camera and f['timestamp'] == stamp for (_, c), f in self.selected.items()):
                    continue
                image = np.asarray(frame['image'])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError('invalid execution camera image')
                self.selected[key] = {'timestamp': stamp, 'image': image.copy(),
                                      'raw_recv_timestamp': frame.get('raw_recv_timestamp')}

    def start(self):
        if self.error:
            return self
        def run():
            try:
                while not self.stop.is_set() and not self.cancelled() and time.monotonic() < self.deadline:
                    self.poll()
                    if len(self.selected) == len(self.rows) * len(CAMERAS):
                        break
                    self.stop.wait(.01)
            except Exception as error:
                self.error = str(error)
        try:
            self.thread = threading.Thread(target=run, name='v10-execution-images', daemon=True)
            self.thread.start()
        except Exception as error:
            self.thread = None
            self.error = str(error)
        return self

    def finish(self, session, *, drained):
        from PIL import Image
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
            if self.thread.is_alive():
                # Never read data concurrently with a stuck camera provider.
                raise RuntimeError('execution camera sampler did not stop')
        directory = session.path/'execution_feedback'/self.record['chunk_id']
        directory.mkdir(parents=True, exist_ok=False)
        self.record.update(drain_completed=drained, physical_execution_confirmed=False,
                           sampling='five_uniform_waypoint_times_three_views',
                           timestamp_semantics='latency-adjusted camera ROS timestamp; approximate temporal alignment, not action completion',
                           max_image_time_error_s=self.max_skew)
        for row in self.rows:
            sample = {'action_index': row, 'action_row': row+1,
                      'action_target': self.actions[row].tolist(), 'images': {}, 'missing_cameras': []}
            if self.times is not None:
                sample['scheduled_timestamp'] = float(self.times[row])
            for camera in CAMERAS:
                frame = self.selected.get((row, camera))
                if frame is None:
                    sample['missing_cameras'].append(camera)
                    continue
                path = directory/f'row_{row+1:03d}_{camera}.png'
                Image.fromarray(frame['image'][..., ::-1]).save(path)
                nearest = int(np.argmin(np.abs(self.times-frame['timestamp'])))
                sample['images'][camera] = {
                    'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'timestamp': frame['timestamp'], 'raw_recv_timestamp': frame['raw_recv_timestamp'],
                    'delta_to_scheduled_s': float(frame['timestamp'] - self.times[row]),
                    'nearest_action_index': nearest, 'nearest_action_row': nearest+1}
            self.record['samples'].append(sample)
        self.record['capture_status'] = ('complete' if len(self.selected) == len(self.rows)*3 and not self.error
                                         else 'partial' if self.selected else 'unavailable')
        if self.error:
            self.record['capture_error'] = self.error
        atomic_json(directory/'record.json', self.record)
        return self.record
