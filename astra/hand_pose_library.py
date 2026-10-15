"""Verified, session-frozen hand targets from the 18 operator-supplied captures."""
import hashlib
import json
import math
from pathlib import Path

POSE_IDS = {side: tuple(f'{prefix}{i:02d}' for i in range(1, 10))
            for side, prefix in (('left', 'L'), ('right', 'R'))}
ROOT = Path(__file__).resolve().parents[1] / 'config/astra/hand_poses_v11'


def numeric_vector(value, length, label):
    if (not isinstance(value, list) or len(value) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f'{label} must contain exactly {length} finite numbers')
    return list(value)


def validate_library(library):
    if not isinstance(library, dict) or set(library) != {'left', 'right'}:
        raise ValueError('V11 requires left and right hand pose libraries')
    for side in POSE_IDS:
        poses = library[side]
        if not isinstance(poses, list) or len(poses) != 9:
            raise ValueError('V11 requires nine poses per hand')
        if [p.get('id') for p in poses if isinstance(p, dict)] != list(POSE_IDS[side]):
            raise ValueError('V11 missing, duplicate or wrong-side pose IDs')
        for pose in poses:
            if pose.get('side') != side or not isinstance(pose.get('name'), str) or not pose['name'].strip():
                raise ValueError('V11 invalid hand pose metadata')
            numeric_vector(pose.get('joints'), 20, 'hand pose joints')
    return library


def load_library(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or len(manifest.get('poses', [])) != 18:
        raise ValueError('V11 invalid hand pose manifest')
    library = {'left': [], 'right': []}
    for entry in manifest['poses']:
        path = (root / entry['file']).resolve()
        if path.parent != root:
            raise ValueError('V11 pose file must be inside the library')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('V11 hand pose checksum mismatch: ' + entry['id'])
        pose = json.loads(raw)
        side = entry['side']
        expected_names = [f'{side}_hand_finger{finger}_joint{joint}'
                          for finger in range(1, 6) for joint in range(1, 5)]
        if (side not in library or pose.get('side') != side or pose.get('unit') != 'rad'
                or pose.get('source') != 'measured' or pose.get('schema_version') != 1
                or pose.get('joint_names') != expected_names or pose.get('name') != entry['name']):
            raise ValueError('V11 incompatible hand pose: ' + entry['id'])
        joints = numeric_vector(pose.get('joints'), 20, 'hand pose joints')
        if joints != pose.get('measured', {}).get('joints'):
            raise ValueError('V11 hand pose must use the captured measured joints')
        library[side].append(dict(id=entry['id'], side=side, name=entry['name'], joints=joints,
                                  source_id=pose['id'], sha256=entry['sha256']))
    return validate_library(library)
