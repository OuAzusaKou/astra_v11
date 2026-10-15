"""V5 native candidate transport and cancellable, single-flight imagination."""
import copy
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from policy_server.msgpack_numpy import packb, unpackb

PROTOCOL = 'leo-dynamics-native-v1'
ADAPTER_SHA = 'ca9e908431a5dd347a4850317fff29c1ed6c596bb3d76a9fa675684612b48e66'
NORM_SHA = 'a825706e5c6da9e86187cc3d9f60e348297050651d873082c817da37ac06a155'


def rpc(host, port, request, timeout=20):
    def read(sock, n):
        data = bytearray()
        while len(data) < n:
            block = sock.recv(n-len(data))
            if not block:
                raise ConnectionError('Dynamics socket closed')
            data.extend(block)
        return bytes(data)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        data = packb(request)
        if len(data) > 16*1024*1024:
            raise ValueError('Dynamics request too large')
        sock.sendall(len(data).to_bytes(4, 'big')+data)
        size = int.from_bytes(read(sock, 4), 'big')
        if not 0 < size <= 16*1024*1024:
            raise ValueError('Dynamics response too large')
        result = unpackb(read(sock, size))
    if result.get('success') is not True:
        raise RuntimeError('Dynamics: '+str(result.get('error')))
    if result.get('request_id') != request['request_id']:
        raise ValueError('Dynamics request identity mismatch')
    return result


def current52(obs):
    return np.concatenate([np.asarray(obs[k])[0] for k in
                           ('left_ee_pose','right_ee_pose','left_hand_joint','right_hand_joint')])


def same_anchor(anchor, fresh):
    """Conservative pre-publication drift guard, not a collision detector."""
    from scipy.spatial.transform import Rotation
    a, b = current52(anchor), current52(fresh)
    for off in (0, 6):
        if np.linalg.norm(a[off:off+3]-b[off:off+3]) > .02:
            return False
        if (Rotation.from_rotvec(a[off+3:off+6]).inv()*Rotation.from_rotvec(b[off+3:off+6])).magnitude() > .15:
            return False
    if np.max(np.abs(a[12:]-b[12:])) > .15:
        return False
    for key in ('image','image_wrist_left','image_wrist_right'):
        x = np.asarray(Image.fromarray(np.asarray(anchor[key], dtype=np.uint8)).resize((64,48)),float)
        y = np.asarray(Image.fromarray(np.asarray(fresh[key], dtype=np.uint8)).resize((64,48)),float)
        if np.abs(x-y).mean()/255 > .08:
            return False
    return True


def native_request(obs, actions, fps, lead, candidate_id, converter=None):
    """Resample absolute robot targets; MANO conversion is the deployed converter."""
    from utils.interpolation_util import PoseInterpolator, get_interp1d
    if converter is None:
        from mano.converter import TeleopManoFieldConverter
        converter = TeleopManoFieldConverter()
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 52 or not np.isfinite(actions).all():
        raise ValueError('expected finite absolute [N,52] targets')
    if not np.isfinite([fps,lead]).all() or fps <= 0 or lead < 0:
        raise ValueError('invalid action clock')
    duration = lead+len(actions)/fps
    if not .5 <= duration <= 1.000001:
        raise ValueError('V5 first release covers 0.5–1.0 seconds only')
    # Fixed 30Hz query times, no squeezing a 1.5s action into a 1s prediction.
    h = int(round(duration*30))
    if abs(h/30-duration) > 1e-6 or not 15 <= h <= 30:
        raise ValueError('candidate endpoint must align with 30Hz training grid')
    times = np.r_[0., lead+(np.arange(len(actions))+1)/fps]
    query = np.arange(30)/30.
    path = np.vstack([current52(obs), actions])
    q = np.minimum(query, duration)
    sampled = np.concatenate([PoseInterpolator(times,path[:,:6])(q),
                              PoseInterpolator(times,path[:,6:12])(q),
                              get_interp1d(times,path[:,12:])(q)],axis=1)
    fields = {}
    for side, ee, hand in (('left',0,12),('right',6,32)):
        fields[f'observation.{side}_arm_ee_state'] = np.repeat(path[:1,ee:ee+6],30,axis=0).astype(np.float32)
        fields[f'observation.{side}_hand_joint_state'] = np.repeat(path[:1,hand:hand+20],30,axis=0).astype(np.float32)
        fields[f'action.{side}_arm_ee_action'] = sampled[:,ee:ee+6].astype(np.float32)
        fields[f'action.{side}_hand_joint_action'] = sampled[:,hand:hand+20].astype(np.float32)
    converted = converter.convert(fields)
    return {'protocol':PROTOCOL,'request_id':candidate_id,'dataset_name':'teleop_v22',
            'adapter_sha256':ADAPTER_SHA,'norm_sha256':NORM_SHA,
            # Robot observations are BGR; the training/gateway contract is RGB.
            'fields':{**fields, **converted}, 'images':[np.ascontiguousarray(np.asarray(obs[k],dtype=np.uint8)[..., ::-1]) for k in
                                        ('image','image_wrist_left','image_wrist_right')],
            'action_length':h,'duration_seconds':duration}


class Imagination:
    def __init__(self, host='10.128.39.115', port=29642, mode='preview', timeout=30.):
        if mode not in ('preview','review'):
            raise ValueError('invalid imagination mode')
        self.host,self.port,self.mode,self.timeout = host,port,mode,timeout
        self.pending = None
        self.worker = None
        self.result = None
        self.error = None

    def cancel(self):
        self.pending = None  # The in-flight socket is bounded; never queue another worker.

    def start(self, controller, batch):
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError('Previous imagination is still draining; retry after timeout')
        obs = controller.robot.observe()
        if controller.observation is not None and not same_anchor(controller.observation,obs):
            raise RuntimeError('Scene/state changed since Astra planned; observe and replan')
        fps = getattr(controller.robot,'fps',20.)
        lead = getattr(controller.robot,'lead_time',.1)
        # Choose the largest available prefix with an endpoint on the 30Hz grid.
        valid = [n for n in range(1,len(batch)+1) if .5 <= lead+n/fps <= 1.000001
                 and abs((lead+n/fps)*30-round((lead+n/fps)*30)) < 1e-6]
        if not valid:
            raise ValueError('No 0.5–1s prefix aligned to the dynamics 30Hz grid')
        actions = batch[:max(valid)].copy()
        ident = uuid.uuid4().hex
        refs = controller.session.snapshot(obs,'imagination_'+ident)
        self.pending = {'id':ident,'actions':actions,'anchor':copy.deepcopy(obs),'snapshot':refs,
                        'phase':controller.phase,'condition':controller.condition_version,
                        'created':time.monotonic(),'fps':fps,'lead':lead,
                        'sha256':hashlib.sha256(actions.tobytes()).hexdigest()}
        self.result = self.error = None
        pending = self.pending
        def work():
            try:
                request = native_request(obs,actions,fps,lead,ident)
                response = rpc(self.host,self.port,request,self.timeout)
                if self.pending is pending:
                    self.result = response
            except Exception as exc:
                if self.pending is pending:
                    self.error = str(exc)
        self.worker = threading.Thread(target=work,daemon=True)
        self.worker.start()
        controller.session.pending = None
        controller.state = 'IMAGINING'
        controller.session.event('imagination_started',candidate_id=ident,actions=actions.tolist(),
                                 candidate_sha256=pending['sha256'],anchor=refs,
                                 fps=fps,lead_time=lead,mode=self.mode)
        controller.status()

    def poll(self, c):
        p = self.pending
        if p is None or c.state != 'IMAGINING':
            return
        c.robot.check()
        if c.phase != p['phase'] or c.condition_version != p['condition'] or c.mode != 'ASTRA':
            raise RuntimeError('Imagination ownership changed')
        if time.monotonic()-p['created'] > self.timeout+5:
            raise TimeoutError('Imagination timed out; no actions published')
        if self.error:
            raise RuntimeError(self.error)
        if self.result is None:
            return
        result = self.result
        for k,expected in (('adapter_sha256',ADAPTER_SHA),('norm_sha256',NORM_SHA)):
            if result.get(k) != expected:
                raise ValueError('Dynamics '+k+' mismatch')
        image = np.asarray(result['future_image'])
        state = np.asarray(result['future_proprio'])
        mask = np.asarray(result['future_proprio_valid_mask'])
        if image.dtype != np.uint8 or image.shape != (368,320,3):
            raise ValueError('invalid dynamics future image')
        if state.shape != (1,346) or mask.shape != (1,346) or mask.dtype != np.bool_ or not np.isfinite(state).all():
            raise ValueError('invalid dynamics future state')
        if result.get('state_space') != 'canonical_normalized':
            raise ValueError('invalid dynamics state space')
        current = np.asarray(result.get('input_image_preview'))
        if current.shape != image.shape or current.dtype != np.uint8:
            raise ValueError('invalid dynamics anchor mosaic')
        directory = c.session.path/'imagination'/p['id'];directory.mkdir(parents=True)
        np.savez_compressed(directory/'prediction.npz',future_proprio=state,future_proprio_valid_mask=mask)
        Image.fromarray(current).save(directory/'current.png')
        from PIL import ImageDraw
        panel = Image.new('RGB',(640,398))
        panel.paste(Image.fromarray(current),(0,30));panel.paste(Image.fromarray(image),(320,30))
        draw=ImageDraw.Draw(panel);draw.text((10,8),'REAL anchor',fill='white')
        draw.text((330,8),'IMAGINED future (not executed)',fill='white')
        panel.save(directory/'panel.png')
        from .session import atomic_json
        file = directory/'future.png';Image.fromarray(image).save(file)
        asset = {'path':str(file.resolve()),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        info = {'candidate_id':p['id'],'candidate_sha256':p['sha256'],
                'synthetic':True,'not_execution_evidence':True,'images':[asset],
                'anchor_timestamp':float(p['anchor']['timestamp']),
                'target_horizon_seconds':p['lead']+len(p['actions'])/p['fps'],
                'future_proprio':state.tolist(),'future_proprio_valid_mask':mask.tolist(),
                'state_space':'canonical_normalized','actions':p['actions'].tolist(),
                'fps':p['fps'],'lead_time':p['lead']}
        # Publish the viewer pointer only after the complete prediction bundle is saved.
        record = {**info, 'session_id':c.session.id, 'phase':p['phase'],
                  'conditioning_version':p['condition'], 'l3':c.l3, 'l2':c.l2,
                  'dynamics_mode':self.mode, 'saved_at':time.time(),
                  'status_at_save':'prediction_only',
                  'target_timestamp':info['anchor_timestamp']+info['target_horizon_seconds'],
                  'target_timestamp_is_hypothetical':True,
                  'files':{name:str((directory/name).resolve()) for name in
                           ('current.png','future.png','panel.png','prediction.npz')}}
        atomic_json(directory/'metadata.json', record)
        entry = {key:record[key] for key in ('candidate_id','phase','l3','l2','saved_at',
                                           'anchor_timestamp','target_timestamp')}
        entry['metadata'] = str((directory/'metadata.json').resolve())
        with (directory.parent/'index.jsonl').open('a') as out:
            out.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n')
            out.flush();os.fsync(out.fileno())
        atomic_json(directory.parent/'latest.json', {'candidate_id':p['id'],
                    'panel':str((directory/'panel.png').resolve()),'synthetic':True,
                    'metadata':entry['metadata']})
        print(f'[V5 IMAGINATION SAVED] {directory}',flush=True)
        p['info'] = info
        c.session.event('imagination_completed',**info)
        if self.mode == 'preview':
            c.pause('V5 preview complete; prediction saved; no candidate executed. R requests a new plan.')
            return
        c.queue = p['actions'].copy()
        c.review('REVIEW_IMAGINATION')

    def approve(self,c,d):
        p = self.pending
        if p is None or 'info' not in p or self.mode != 'review':
            raise ValueError('no reviewed imagination candidate')
        if d.get('candidate_id') != p['id'] or d.get('candidate_sha256') != p['sha256']:
            raise ValueError('approval does not match exact imagined candidate')
        if 'actions' in d or d.get('execute_steps') != len(p['actions']):
            raise ValueError('approval must execute the entire imagined prefix, without changed actions')
        if c.phase != p['phase'] or c.condition_version != p['condition'] or c.mode != 'ASTRA':
            raise ValueError('stale candidate ownership')
        if getattr(c.robot,'fps',20.) != p['fps'] or getattr(c.robot,'lead_time',.1) != p['lead']:
            raise ValueError('candidate execution clock changed')
        if time.monotonic()-p['created'] > 600 or not same_anchor(p['anchor'],c.robot.observe()):
            raise RuntimeError('Imagined scene/state expired or changed; observe and replan')
        if not np.array_equal(c.queue,p['actions']):
            raise ValueError('candidate queue mutated after imagination')
        c.approved_imagination = copy.deepcopy(p["info"])
        c.observation = p['anchor']
        self.pending = None  # One-shot approval consumed before any publication.
        return len(p['actions'])
