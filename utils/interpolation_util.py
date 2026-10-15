import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


# ── Numpy ring buffer ────────────────────────────────────────────────────────

class NpRingBuf:
    """Pre-allocated ring buffer for time-series data.

    Uses a doubled backing array so view() always returns a contiguous,
    sorted (oldest-to-newest) numpy slice — zero copy, no Python iteration.
    push() and view() are O(1) numpy writes/slices.

    This eliminates the list(deque) + comprehension + np.array stacking that
    caused 15–20 ms GIL contention in hot loops under executor thread load.
    """

    def __init__(self, maxlen: int, ndim: int, dtype=np.float64):
        self._n    = maxlen
        self._ts   = np.empty(2 * maxlen, dtype=np.float64)
        self._data = np.empty((2 * maxlen, ndim), dtype=dtype)
        self._head = 0   # next write index (mod maxlen)
        self._cnt  = 0   # valid entries (≤ maxlen)

    def push(self, ts: float, row: np.ndarray) -> None:
        i = self._head
        # Write to both halves so view() can always return a contiguous slice.
        self._ts[i] = self._ts[i + self._n] = ts
        self._data[i] = self._data[i + self._n] = row
        self._head = (i + 1) % self._n
        if self._cnt < self._n:
            self._cnt += 1

    def view(self):
        """(timestamps, data) sorted oldest-to-newest as contiguous numpy slices."""
        n = self._cnt
        start = self._head if n == self._n else 0
        return self._ts[start:start + n], self._data[start:start + n]

    def __len__(self) -> int:
        return self._cnt


def fast_interp_at(timestamps: np.ndarray, data: np.ndarray, t_query: float) -> np.ndarray:
    """Binary-search + linear interpolation for a single query timestamp.

    For scalar/joint data only (positions, joint angles, gripper widths).
    Replaces interp1d for the common case of one query on a small sorted array.
    O(log n + D) instead of O(n) scipy setup.

    ⚠️ DO NOT use for rotation data of any kind (rotvec, quaternion, Euler).
    Linear interpolation does not respect SO(3) topology and produces
    catastrophic shortest-arc bugs (e.g. +179° → -179° jumps sweep ~358°
    instead of ~2°). Use `fast_pose_interp_at` for (N, 6) = [xyz, rotvec]
    poses, which lerps position and SLERPs rotation.

    Args:
        timestamps: (N,) sorted float64 array.
        data:       (N, D) float64 array. Must not contain rotation components.
        t_query:    scalar query time.

    Returns:
        (D,) interpolated array.
    """
    n = len(timestamps)
    if n == 1:
        return data[0].copy()
    idx = int(np.clip(np.searchsorted(timestamps, t_query, side='right'), 1, n - 1))
    t0, t1 = timestamps[idx - 1], timestamps[idx]
    dt = t1 - t0
    if dt < 1e-12:
        return data[idx - 1].copy()
    alpha = float(np.clip((t_query - t0) / dt, 0.0, 1.0))
    return data[idx - 1] + alpha * (data[idx] - data[idx - 1])


def fast_interp_multi(timestamps: np.ndarray, data: np.ndarray,
                      t_queries: np.ndarray) -> np.ndarray:
    """Vectorised binary-search + linear interpolation for multiple query timestamps.

    For scalar/joint data only. See `fast_interp_at` for why this MUST NOT be
    used for rotation data; use `fast_pose_interp_multi` for poses.

    Args:
        timestamps: (N,) sorted float64 array.
        data:       (N, D) float64 array. Must not contain rotation components.
        t_queries:  (Q,) query times.

    Returns:
        (Q, D) interpolated array.
    """
    n = len(timestamps)
    t_queries = np.asarray(t_queries, dtype=np.float64)
    idxs = np.clip(np.searchsorted(timestamps, t_queries, side='right'), 1, n - 1)

    t0 = timestamps[idxs - 1]
    t1 = timestamps[idxs]
    dt = t1 - t0
    alphas = np.where(dt > 1e-12,
                      np.clip((t_queries - t0) / dt, 0.0, 1.0),
                      0.0)                                          # (Q,)
    d0 = data[idxs - 1]                                            # (Q, D)
    d1 = data[idxs]                                                # (Q, D)
    return d0 + alphas[:, np.newaxis] * (d1 - d0)                  # (Q, D)


def fast_pose_interp_at(timestamps: np.ndarray, data: np.ndarray,
                        t_query: float) -> np.ndarray:
    """Binary-search + lerp(pos) + Slerp(rot) for a single query timestamp.

    Correct interpolation of 6D pose data [xyz, rotvec]. Position is linearly
    interpolated; rotation is SLERPed via scipy.

    Why this is needed:
        Linear interpolation of any rotation representation (rotvec, quaternion,
        Euler) violates SO(3) topology and produces shortest-arc bugs:
        adjacent samples that represent nearly the same rotation can have
        numerical representations that differ by ~360°. Lerping them sweeps
        a 358° arc through 0° instead of the true ~2° geodesic. SLERP fixes
        this by converting to quaternions and walking the unit-sphere geodesic
        (the internal `dot(q0, q1) < 0` flip guarantees shortest arc).

    Why a 2-point Slerp is fine:
        scipy.spatial.transform.Slerp is piecewise — given N samples, a query
        is interpolated using only the bracketing pair. Constructing Slerp on
        just those 2 samples gives the identical result with O(1) setup
        instead of O(N).

    Args:
        timestamps: (N,) sorted float64 array.
        data:       (N, 6) float64 array, columns [x, y, z, rx, ry, rz].
        t_query:    scalar query time.

    Returns:
        (6,) interpolated pose [x, y, z, rx, ry, rz].
    """
    n = len(timestamps)
    if n == 1:
        return data[0].copy()
    idx = int(np.clip(np.searchsorted(timestamps, t_query, side='right'), 1, n - 1))
    t0 = float(timestamps[idx - 1])
    t1 = float(timestamps[idx])
    dt = t1 - t0
    if dt < 1e-12:
        return data[idx - 1].copy()
    alpha = float(np.clip((t_query - t0) / dt, 0.0, 1.0))

    pos = data[idx - 1, :3] + alpha * (data[idx, :3] - data[idx - 1, :3])

    key_rots = st.Rotation.from_rotvec(data[idx - 1:idx + 1, 3:])
    t_q_clipped = float(np.clip(t_query, t0, t1))
    rotvec = st.Slerp(np.array([t0, t1]), key_rots)(t_q_clipped).as_rotvec()

    return np.concatenate([pos, rotvec])


def fast_pose_interp_multi(timestamps: np.ndarray, data: np.ndarray,
                           t_queries: np.ndarray) -> np.ndarray:
    """Vectorised lerp(pos) + Slerp(rot) for multiple query timestamps on 6D pose data.

    See `fast_pose_interp_at` for the rationale on why SLERP is mandatory for
    rotation interpolation.

    Position uses the same vectorised lerp as `fast_interp_multi`. Rotation
    uses a full-buffer scipy `Slerp` (the construction is O(N) but Slerp object
    construction on N≈150 samples is cheap; per-query work is then O(log N)).
    Equivalent to PoseInterpolator without the wrapper overhead on position.

    Args:
        timestamps: (N,) sorted float64 array.
        data:       (N, 6) float64 array, columns [x, y, z, rx, ry, rz].
        t_queries:  (Q,) query times.

    Returns:
        (Q, 6) interpolated poses.
    """
    n = len(timestamps)
    t_queries = np.asarray(t_queries, dtype=np.float64)

    idxs = np.clip(np.searchsorted(timestamps, t_queries, side='right'), 1, n - 1)
    t0 = timestamps[idxs - 1]
    t1 = timestamps[idxs]
    dt = t1 - t0
    alphas = np.where(dt > 1e-12,
                      np.clip((t_queries - t0) / dt, 0.0, 1.0),
                      0.0)
    p0 = data[idxs - 1, :3]
    p1 = data[idxs, :3]
    pos = p0 + alphas[:, np.newaxis] * (p1 - p0)                   # (Q, 3)

    rots = st.Rotation.from_rotvec(data[:, 3:])
    slerp = st.Slerp(timestamps, rots)
    t_q_clipped = np.clip(t_queries, float(timestamps[0]), float(timestamps[-1]))
    rotvecs = slerp(t_q_clipped).as_rotvec()                       # (Q, 3)

    return np.concatenate([pos, rotvecs], axis=1)                  # (Q, 6)


def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))
    return gripper_interp


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_rotvec(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose

def get_gripper_calibration_interpolator(
        aruco_measured_width, 
        aruco_actual_width):
    """
    Assumes the minimum width in aruco_actual_width
    is measured when the gripper is fully closed
    and maximum width is when the gripper is fully opened
    """
    aruco_measured_width = np.array(aruco_measured_width)
    aruco_actual_width = np.array(aruco_actual_width)
    assert len(aruco_measured_width) == len(aruco_actual_width)
    assert len(aruco_actual_width) >= 2
    aruco_min_width = np.min(aruco_actual_width)
    gripper_actual_width = aruco_actual_width - aruco_min_width
    interp = get_interp1d(aruco_measured_width, gripper_actual_width)
    return interp
