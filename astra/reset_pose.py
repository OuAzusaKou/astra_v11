"""Read the same frame-zero task pose as the legacy client, without ROS imports."""
import json
from pathlib import Path
import numpy as np


def load_reset_pose(path):
    if not path:
        raise ValueError('startup.reset_state_path is required for B reset')
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / 'state.npz'
    fields = {'left_ee_pose': 6, 'right_ee_pose': 6, 'left_hand_state': 20, 'right_hand_state': 20}
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            values = {k: np.asarray(data[k][0]) for k in fields}
    elif path.suffix == '.json':
        data = json.loads(path.read_text())
        values = data.get('mean_by_key', data)
    else:
        raise ValueError('reset pose must be state.npz or JSON; no pickle fallback')
    result = []
    for key, size in fields.items():
        a = np.asarray(values[key], dtype=float).reshape(-1)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f'invalid reset pose: {key}')
        result.append(a)
    return str(path.resolve()), np.concatenate(result)
