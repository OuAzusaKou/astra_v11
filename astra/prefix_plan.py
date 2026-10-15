"""V7 bounded execution prefixes. Bounds are not collision/IK certification."""
import numpy as np


def validate_stages(stages, rows):
    if not isinstance(stages, list) or not stages or not 1 <= rows <= 120:
        raise ValueError('V7 requires plan_stages covering 1..120 action rows')
    previous = 0
    for stage in stages:
        if not isinstance(stage, dict) or stage.get('kind') not in ('transport', 'approach', 'contact', 'release'):
            raise ValueError('stage kind must be transport/approach/contact/release')
        end = stage.get('end_step')
        if type(end) is not int or not previous < end <= rows:
            raise ValueError('stage end_step must be increasing exclusive row bounds')
        if not isinstance(stage.get('evidence'), str) or not stage['evidence'].strip():
            raise ValueError('each stage requires grounding evidence')
        previous = end
    if previous != rows:
        raise ValueError('stages must cover the complete plan')


def prefix_limit(actions, anchor, stages, fps, lead, budget):
    validate_stages(stages, len(actions))
    if not np.isfinite([fps, lead]).all() or fps <= 0 or lead < 0:
        raise ValueError('invalid execution clock')
    stage = stages[0]
    transport = stage['kind'] == 'transport'
    seconds, distance = (1.5, .18) if transport else (.5, .04)
    count = min(stage['end_step'], int(np.floor((seconds-lead)*fps+1e-6)), budget)
    reason = 'stage/time/budget boundary'
    path = np.concatenate([np.asarray(anchor)[None], np.asarray(actions)], axis=0)
    travel = np.zeros(2)
    for i in range(count):
        delta = path[i+1]-path[i]
        step = np.array([np.linalg.norm(delta[:3]), np.linalg.norm(delta[6:9])])
        travel += step
        rotation = max(np.linalg.norm(delta[3:6]), np.linalg.norm(delta[9:12]))
        # Conservative rotvec difference; near representation wrap, replan instead of accelerating.
        if step.max()*fps > .15+1e-9 or rotation*fps > .8+1e-9 or travel.max() > distance+1e-9:
            count=i;reason='translation/rotation speed or path-length bound';break
        if transport and np.max(np.abs(path[i+1,12:]-path[0,12:])) > .05:
            count=i;reason='transport must preserve grasp; use a separate contact/release stage';break
    if count < 1:
        raise ValueError('No executable V7 prefix; slow/replan trajectory or refresh observation')
    return {'count':int(count), 'stage':stage['kind'], 'max_seconds_including_lead':seconds,
            'max_path_m_per_arm':distance, 'max_translation_m_s':.15, 'max_rotvec_rad_s':.8,
            'reason':reason, 'collision_free_certified':False, 'ik_certified':False}
