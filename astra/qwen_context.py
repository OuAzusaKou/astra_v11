"""Bridge-owned static evidence for Qwen's explicitly tool-free transport."""
import hashlib
import json
import time

ROS_URDF = '/home/liberai/Desktop/colcon_ws/src/leo_description/urdf/_leo.urdf.xacro'
SOURCES = ('envs/leo_wuji_env_infer.py', 'astra/robot.py', 'utils/pose_utils.py',
           'config/astra/version4_vlm_subtasks.json', 'config/astra/version8_dynamics_candidates.json')


def build_context(repo, read_page, cancel):
    files=[]
    for path in [repo.rstrip('/')+'/'+name for name in SOURCES]+[ROS_URDF]:
        start=1;total=None;parts=[]
        while total is None or start<=total:
            if cancel.is_set():raise InterruptedError('cancelled control-context preload')
            page=read_page(path,start,100)
            count=page.get('total_lines')
            if (page.get('error') or page.get('path')!=path or page.get('start_line')!=start
                    or type(count) is not int or not 0<count<=2500 or page.get('truncated')
                    or (total is not None and count!=total)):
                raise RuntimeError('Incomplete or changing control source: '+path)
            total=count;rows=page.get('text','').splitlines()
            expected=min(100,total-start+1)
            if len(rows)!=expected or any(not row.startswith(str(start+i)+': ') for i,row in enumerate(rows)):
                raise RuntimeError('Missing control source lines: '+path)
            parts.extend(rows);start+=len(rows)
        text='\n'.join(parts)
        files.append(dict(path=path,total_lines=total,text=text,
                          sha256_numbered_text=hashlib.sha256(text.encode()).hexdigest()))
    content=dict(schema='qwen_preloaded_control_v1',files=files,
        action_contract=dict(dimensions=52,frame='chest_link',position_unit='m',rotation='rotation_vector_rad',
                             left_ee='0:6',right_ee='6:12',left_fingers='12:32',right_fingers='32:52'),
        limitations='Static source snapshot, not live TF, full robot calibration, collision clearance or grasp proof. '
                     'Raw xacro includes may refer to files not included here. Do not assume axis directions from names. '
                     'Use CURRENT_REQUEST and its real observations for current geometry and state; do not execute source code.')
    raw=json.dumps(content,ensure_ascii=False,sort_keys=True)
    if len(raw.encode('utf-8'))>250000:raise RuntimeError('Control-context preload exceeds bound')
    if cancel.is_set():raise InterruptedError('cancelled control-context preload')
    return dict(content,context_sha256=hashlib.sha256(raw.encode()).hexdigest(),loaded_at=time.time())
