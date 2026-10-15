"""Blend overlapping absolute action predictions at matching ROS deadlines."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class TemporalActionEnsembler:
    def __init__(self, max_chunks=0):
        # Includes the current prediction; 0 keeps all overlapping chunks.
        self.max_chunks = max_chunks
        self.history = []

    def apply(self, actions, deadlines):
        """Average predictions, interpolating past chunks without extrapolation.

        Actions are absolute [xyz, rotvec] x 2 + 40 hand joints. Positions and
        joints use an arithmetic mean; orientations use hemisphere-aligned
        quaternion averaging after SLERP interpolation. Store raw predictions
        so previously averaged outputs are never counted again.
        """
        actions = np.asarray(actions, dtype=np.float64)
        deadlines = np.asarray(deadlines, dtype=np.float64)
        if not len(actions):
            return actions
        # Repeated observations or a backwards clock start a fresh history.
        if self.history and deadlines[0] <= self.history[-1][0][0]:
            self.history.clear()
        self.history = [(t, a) for t, a in self.history if t[-1] >= deadlines[0]]
        if self.max_chunks:
            self.history = self.history[-(self.max_chunks - 1):] if self.max_chunks > 1 else []
        result = actions.copy()
        counts = np.ones(len(actions))
        rotations = (slice(3, 6), slice(9, 12))
        reference = [Rotation.from_rotvec(actions[:, part]).as_quat() for part in rotations]
        quat_sums = [q.copy() for q in reference]
        for times, past in self.history:
            overlap = (deadlines >= times[0]) & (deadlines <= times[-1])
            target = deadlines[overlap]
            if not len(target):
                continue
            if len(times) == 1:
                interpolated = past.copy()
            else:
                # Work with relative seconds to preserve sub-frame precision.
                index = np.searchsorted(times, target, side="right").clip(1, len(times) - 1)
                fraction = ((target - times[index - 1]) / (times[index] - times[index - 1]))[:, None]
                interpolated = past[index - 1] * (1 - fraction) + past[index] * fraction
            result[overlap] += interpolated
            counts[overlap] += 1
            for side, part in enumerate(rotations):
                rotation = Rotation.from_rotvec(past[:, part])
                quat = (Slerp(times - times[0], rotation)(target - times[0]).as_quat()
                        if len(times) > 1 else rotation.as_quat())
                sign = np.where(np.sum(quat * reference[side][overlap], axis=1) < 0, -1., 1.)
                quat_sums[side][overlap] += quat * sign[:, None]
        result /= counts[:, None]
        for part, quats in zip(rotations, quat_sums):
            result[:, part] = Rotation.from_quat(quats).as_rotvec()
        self.history.append((deadlines.copy(), actions.copy()))
        return result
