"""V8: one real anchor, six directional candidates and one fresh WAM candidate."""
import copy
import hashlib
import json
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .dynamics import ADAPTER_SHA, NORM_SHA, Imagination, current52, native_request, rpc, same_anchor
from .session import atomic_json


def action_hash(actions):
    return hashlib.sha256(np.asarray(actions, dtype=np.float64).tobytes()).hexdigest()


def prefix_steps(available, fps, lead):
    if not np.isfinite([fps, lead]).all() or fps <= 0 or lead < 0:
        raise ValueError('invalid candidate clock')
    valid = [n for n in range(1, available+1) if .5 <= lead+n/fps <= 1.000001
             and abs((lead+n/fps)*30-round((lead+n/fps)*30)) < 1e-6]
    if not valid:
        raise ValueError('No shared 0.5–1s candidate prefix on the Dynamics 30Hz grid')
    return max(valid)


def directional_candidates(anchor, hand, count, distance):
    if hand not in ('left', 'right'):
        raise ValueError('active_hand must be left or right')
    if not np.isfinite(distance) or not 0 < distance <= .10:
        raise ValueError('candidate distance must be >0 and <=0.10 m')
    if count < 1:
        raise ValueError('candidate must contain points')
    state = np.asarray(current52(anchor), dtype=np.float64)
    if state.shape != (52,) or not np.isfinite(state).all():
        raise ValueError('invalid real anchor state')
    ramp = (1-np.cos(np.pi*np.arange(1, count+1)/count))/2
    for axis, name in enumerate('xyz'):
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            actions = np.repeat(state[None, :], count, axis=0)
            actions[:, (0 if hand == 'left' else 6)+axis] += sign*distance*ramp
            yield f'{hand}_{label}_{name}', actions


def validate_prediction(result):
    for key, expected in (('adapter_sha256', ADAPTER_SHA), ('norm_sha256', NORM_SHA)):
        if result.get(key) != expected:
            raise ValueError('Dynamics '+key+' mismatch')
    future = np.asarray(result.get('future_image'))
    current = np.asarray(result.get('input_image_preview'))
    state = np.asarray(result.get('future_proprio'))
    mask = np.asarray(result.get('future_proprio_valid_mask'))
    if any(im.dtype != np.uint8 or im.shape != (368, 320, 3) for im in (future, current)):
        raise ValueError('invalid Dynamics images')
    if (state.shape != (1, 346) or mask.shape != (1, 346) or mask.dtype != np.bool_
            or not np.isfinite(state).all() or result.get('state_space') != 'canonical_normalized'):
        raise ValueError('invalid Dynamics normalized state/mask')
    return current, future, state, mask


class CandidateImagination(Imagination):
    def __init__(self, host='10.128.39.115', port=29642, mode='preview', timeout=30., distance=.10):
        super().__init__(host, port, mode, timeout)
        if not np.isfinite(distance) or not 0 < distance <= .10:
            raise ValueError('candidate distance must be >0 and <=0.10 m')
        self.distance = distance

    def start(self, c, hand):
        if hand not in ('left', 'right'):
            raise ValueError('Astra must choose active_hand left or right')
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError('Previous candidate set still draining; retry after timeout')
        c.robot.check()
        obs = c.robot.observe()
        if c.observation is not None and not same_anchor(c.observation, obs):
            raise RuntimeError('Scene changed since hand selection; observe and choose again')
        # The policy is queried once with this exact anchor and current L3/L2.
        # Its original trajectory is a candidate only, never put in the live queue.
        policy, receipt = c.policy.infer(copy.deepcopy(obs), c.prompt)
        policy = np.asarray(policy, dtype=np.float64)
        if policy.ndim != 2 or policy.shape[1] != 52 or not np.isfinite(policy).all():
            raise ValueError('invalid policy candidate')
        if c.cancelled() or c.switch_pending():
            return
        c.robot.check()
        if not same_anchor(obs, c.robot.observe()):
            raise RuntimeError('Scene changed during policy inference; regenerate all candidates')
        fps, lead = getattr(c.robot, 'fps', 20.), getattr(c.robot, 'lead_time', .1)
        n = prefix_steps(min(len(policy), c.settings.takeover_steps, c.settings.budget-c.steps), fps, lead)
        ident = uuid.uuid4().hex
        entries = []
        proposals = list(directional_candidates(obs, hand, n, self.distance)) + [('policy', policy[:n].copy())]
        for label, actions in proposals:
            entry = dict(candidate_id=uuid.uuid4().hex, label=label,
                         source='policy' if label == 'policy' else 'direction',
                         actions=actions, candidate_sha256=action_hash(actions), status='pending')
            try:
                c.robot.validate(actions, obs)
            except ValueError as error:
                entry.update(status='rejected', error=str(error))
            entries.append(entry)
        c.robot.check()
        if c.cancelled() or c.switch_pending():
            return
        if not same_anchor(obs, c.robot.observe()):
            raise RuntimeError('Scene changed during candidate validation; regenerate all candidates')
        if not any(e['status'] == 'pending' for e in entries):
            raise ValueError('No candidate passed trajectory validation')
        refs = c.session.snapshot(obs, 'candidate_set_'+ident)
        p = dict(id=ident, anchor=copy.deepcopy(obs), snapshot=refs, active_hand=hand,
                 phase=c.phase, condition=c.condition_version, created=time.monotonic(),
                 fps=fps, lead=lead, count=n, candidates=entries, policy_receipt=receipt)
        self.pending = p
        self.result = self.error = None
        directory = c.session.path/'imagination'/ident
        directory.mkdir(parents=True)
        p['directory'] = directory
        atomic_json(directory/'request.json', dict(candidate_set_id=ident, active_hand=hand,
                    frame='chest_link', distance_m=self.distance, anchor=refs, policy_receipt=receipt,
                    fps=fps, lead_time=lead, candidates=[self.describe(e) for e in entries]))

        def work():
            results = []
            for entry in entries:
                if self.pending is not p:
                    return
                out = dict(entry)
                if entry['status'] == 'pending':
                    try:
                        req = native_request(obs, entry['actions'], fps, lead, entry['candidate_id'])
                        res = rpc(self.host, self.port, req, self.timeout)
                        validate_prediction(res)
                        out.update(status='predicted', result=res)
                    except Exception as error:
                        out.update(status='failed', error=str(error))
                results.append(out)
            if self.pending is p:
                self.result = results

        c.session.pending = None
        c.state = 'IMAGINING'
        c.session.event('candidate_set_started', candidate_set_id=ident, active_hand=hand,
                        count=len(entries), anchor=refs, policy_receipt=receipt, mode=self.mode)
        c.status()
        self.worker = threading.Thread(target=work, daemon=True, name='v8-dynamics-candidates')
        self.worker.start()

    @staticmethod
    def describe(entry):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in entry.items()
                if k != 'result'}

    def poll(self, c):
        p = self.pending
        if p is None or c.state != 'IMAGINING':
            return
        c.robot.check()
        if c.phase != p['phase'] or c.condition_version != p['condition'] or c.mode != 'ASTRA':
            raise RuntimeError('candidate set ownership changed')
        if time.monotonic()-p['created'] > 7*(self.timeout+5):
            raise TimeoutError('candidate set prediction timed out')
        if self.result is None:
            return
        rows = self.result
        images = []
        panel = Image.new('RGB', (960, 1200), '#181818')
        draw = ImageDraw.Draw(panel)
        real = None
        predicted = []
        for i, row in enumerate(rows):
            sub = p['directory']/row['candidate_id'];sub.mkdir()
            desc = self.describe(row)
            if row['status'] == 'predicted':
                current, future, state, mask = validate_prediction(row['result'])
                if real is not None and not np.array_equal(current, real):
                    raise ValueError('candidate predictions have different anchor mosaics')
                real = current
                Image.fromarray(current).save(sub/'current.png')
                file = sub/'future.png';Image.fromarray(future).save(file)
                np.savez_compressed(sub/'prediction.npz', future_proprio=state, future_proprio_valid_mask=mask)
                asset = dict(path=str(file.resolve()), sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                             candidate_id=row['candidate_id'], label=row['label'])
                images.append(asset)
                desc['image_sha256'] = asset['sha256']
                predicted.append(row)
                panel.paste(Image.fromarray(future), ((i % 3)*320, (i//3)*400+28))
            draw.text(((i % 3)*320+6, (i//3)*400+6), row['label']+' | '+row['status'], fill='white')
            atomic_json(sub/'metadata.json', dict(desc, candidate_set_id=p['id'],
                        anchor_timestamp=float(p['anchor']['timestamp']), synthetic=True, not_execution_evidence=True))
        if not predicted:
            raise RuntimeError('All Dynamics candidates failed; inspect candidate metadata')
        panel.paste(Image.fromarray(real), (320, 828))
        draw.text((326, 806), 'REAL anchor (all candidates)', fill='cyan')
        panel.save(p['directory']/'panel.png')
        p['candidates'] = rows
        info = dict(candidate_set_id=p['id'], active_hand=p['active_hand'], frame='chest_link',
                    distance_m=self.distance, synthetic=True, not_execution_evidence=True, images=images,
                    candidates=[self.describe(e) for e in rows], anchor_timestamp=float(p['anchor']['timestamp']),
                    fps=p['fps'], lead_time=p['lead'], target_horizon_seconds=p['lead']+p['count']/p['fps'])
        p['info'] = info
        atomic_json(p['directory']/'metadata.json', dict(info, l3=c.l3, l2=c.l2, phase=c.phase,
                    conditioning_version=c.condition_version, status_at_save='prediction_only'))
        atomic_json(p['directory'].parent/'latest.json', dict(candidate_id=p['id'], candidate_set_id=p['id'],
                    panel=str((p['directory']/'panel.png').resolve()), synthetic=True,
                    metadata=str((p['directory']/'metadata.json').resolve())))
        with (p['directory'].parent/'index.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(dict(candidate_set_id=p['id'], metadata=str(p['directory']/'metadata.json')))+'\n')
        c.session.event('candidate_set_predicted', **info)
        if self.mode == 'preview':
            c.pause('V8 candidate preview saved; no candidate executed')
            return
        c.queue = np.empty((0, 52))
        c.review('REVIEW_CANDIDATES')

    def approve(self, c, d):
        p = self.pending
        if (p is None or 'info' not in p or self.mode != 'review'
                or c.state != 'WAITING' or not c.session.pending
                or c.session.pending.get('stage') != 'REVIEW_CANDIDATES'):
            raise ValueError('no reviewed candidate set')
        if d.get('candidate_set_id') != p['id']:
            raise ValueError('wrong candidate set')
        selected = next((e for e in p['candidates'] if e['candidate_id'] == d.get('candidate_id')), None)
        if selected is None or selected['status'] != 'predicted':
            raise ValueError('candidate unavailable or prediction failed')
        if (d.get('candidate_sha256') != selected['candidate_sha256']
                or action_hash(selected['actions']) != selected['candidate_sha256']):
            raise ValueError('candidate hash mismatch')
        if 'actions' in d or type(d.get('execute_steps')) is not int or d['execute_steps'] != len(selected['actions']):
            raise ValueError('select one full predicted prefix without modifying actions')
        if c.phase != p['phase'] or c.condition_version != p['condition'] or c.mode != 'ASTRA':
            raise ValueError('stale candidate set ownership')
        if getattr(c.robot, 'fps', 20.) != p['fps'] or getattr(c.robot, 'lead_time', .1) != p['lead']:
            raise ValueError('candidate execution clock changed')
        if len(c.queue) or c.settings.budget-c.steps < len(selected['actions']):
            raise ValueError('candidate queue or budget changed')
        c.robot.check()
        if time.monotonic()-p['created'] > 600 or not same_anchor(p['anchor'], c.robot.observe()):
            raise RuntimeError('candidate set anchor expired/changed; regenerate the whole set')
        c.robot.validate(selected['actions'], p['anchor'])
        c.queue = selected['actions'].copy()
        c.source = 'V8_'+selected['label']
        c.observation = p['anchor']
        c.approved_imagination = dict(candidate_id=selected['candidate_id'], candidate_set_id=p['id'],
                    candidate_sha256=selected['candidate_sha256'], anchor_timestamp=float(p['anchor']['timestamp']),
                    target_horizon_seconds=p['lead']+p['count']/p['fps'])
        c.session.event('candidate_selected', candidate_set_id=p['id'], candidate_id=selected['candidate_id'],
                        label=selected['label'], candidate_sha256=selected['candidate_sha256'])
        self.pending = None  # All alternatives consumed together, before publishing.
        return len(selected['actions'])
