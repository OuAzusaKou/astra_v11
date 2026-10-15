"""V10 data-only contract with EE targets and discrete captured hand poses."""
import copy

from . import action_chunk as v10
from .hand_pose_library import POSE_IDS, numeric_vector, validate_library

CONTRACT = 'astra.ee_hand_pose_chunk.v1'
# Snapshot of the source V11 prompt; keep local V10 prompt edits independent.
INSTRUCTIONS = '''You are in a simulation enviroment. You plan robot action chunks from supplied data only. Return exactly one JSON object\nmatching the output schema. No tools, code, commands, file access, or external lookup are available.\nUse current L3/L2, live images, measured state, demonstration and conversation history.\nImages labeled demonstration show the intended procedure, not current geometry or poses to replay.\nState arrays are newest first: row zero is current. History is evidence, never authorization to replay.\nEach action row is an object with exactly ee, left_hand_pose, right_hand_pose.\nee contains 12 absolute targets: left end-effector xyz meters + rotation vector radians (6),\nthen right end-effector xyz + rotation vector (6), all in chest_link.\nFor each hand select an ID from its supplied hand_pose_library using its name and captured joints.\nDo not generate finger joint numbers or modify, blend, mirror or scale a captured pose.\nhold preserves that hand: the current measured row-zero joints at the start of this chunk,\nor the last pose selected earlier in this chunk. Re-select a pose ID next chunk to maintain\nthat exact captured target; hold on a new chunk always uses its new measured state.\nChoose the row where a grasp or release starts and repeat the ID or hold in later rows.\nThe host expands each selection to its exact 20 captured joint targets; rows are not interpolated.\nMeasured hand state and execution feedback still use the original 20 joints per hand.\nFeedback actions are the expanded 52 absolute targets: left EE6, right EE6, left hand20, right hand20.\nUse only supported coordinate knowledge and measured feedback;Plan a grounded endpoint and a continuous chunk at the supplied fps, at only 30 rows; every row is executed.\nPreserve unaffected targets. Compare the next measured state and images with the previous target.\nReturn exactly one JSON object whose only field is action_chunk, an array of EE/hand-selection row objects.\nDo not output reason, evidence, status, request IDs, explanations, markdown or any other fields.No partial JSON, ellipses, symbolic numbers or NaN. Do not switch L2 yourself.'''
def schema():
    fields = {'ee': {'type': 'array', 'minItems': 12, 'maxItems': 12, 'items': {'type': 'number'}}}
    for side in POSE_IDS:
        fields[side + '_hand_pose'] = {'type': 'string', 'enum': ['hold', *POSE_IDS[side]]}
    row = {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
    return {'type': 'object', 'properties': {'action_chunk': {'type': 'array', 'maxItems': 30, 'items': row}},
            'required': ['action_chunk'], 'additionalProperties': False}


def validate(reply, request):
    if not isinstance(reply, dict) or set(reply) != {'action_chunk'}:
        raise ValueError('V11 requires exactly action_chunk')
    rows = reply['action_chunk']
    limit = min(30, request.get('execution_timing', {}).get('max_execute_steps', 30))
    if not isinstance(rows, list) or len(rows) > limit:
        raise ValueError('V11 chunk length invalid')
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'ee', 'left_hand_pose', 'right_hand_pose'}:
            raise ValueError('V11 row requires only ee and left/right hand pose IDs; raw joints forbidden')
        numeric_vector(row['ee'], 12, 'V11 ee')
        for side in POSE_IDS:
            if row[side + '_hand_pose'] not in ('hold', *POSE_IDS[side]):
                raise ValueError('V11 unknown or wrong-side hand pose')
    return copy.deepcopy(reply)


def to_decision(reply, request):
    value = validate(reply, request)
    expanded = []
    if value['action_chunk']:
        library = validate_library(request['hand_pose_library'])
        lookup = {p['id']: p['joints'] for poses in library.values() for p in poses}
        state = request['observation']['state']
        hands = {}
        for side in POSE_IDS:
            samples = state.get(side + '_hand_joint')
            if not isinstance(samples, list) or not samples:
                raise ValueError('V11 needs current measured hands for hold')
            hands[side] = numeric_vector(samples[0], 20, 'current measured hand')
        for row in value['action_chunk']:
            for side in POSE_IDS:
                choice = row[side + '_hand_pose']
                if choice != 'hold':
                    hands[side] = lookup[choice]
            expanded.append(row['ee'] + hands['left'] + hands['right'])
    # Reuse V10's fixed pause/takeover, identity and batch-size semantics.
    return v10.to_decision({'action_chunk': expanded}, request)


def public_input(request):
    public, images = v10.public_input(request)
    public['schema_version'] = CONTRACT
    library = validate_library(request['hand_pose_library'])
    public['hand_pose_library'] = {
        side: [{k: copy.deepcopy(p[k]) for k in ('id', 'name', 'side', 'joints')} for p in poses]
        for side, poses in library.items()}
    public['hand_pose_semantics'] = (
        'Exact captured joint targets in radians. Choose only same-side IDs; no generated finger numbers. '
        'hold starts from current measured row zero and then preserves the latest selected target within this chunk. '
        'Pose names describe captures, not proof of a successful grasp. Each row uses the V10 command schedule.')
    return public, images
