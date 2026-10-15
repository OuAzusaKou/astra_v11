"""Lossless compatibility for evidence formatting; never change control fields."""


def normalize_decision(decision):
    if not isinstance(decision, dict):
        raise ValueError('decision must be a JSON object')
    result = dict(decision)
    evidence = result.get('evidence')
    if isinstance(evidence, list):
        if not evidence or not all(isinstance(item, str) and item.strip() for item in evidence):
            raise ValueError('evidence list must contain nonempty text items only')
        result['evidence'] = '\n'.join(evidence)
    elif not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('evidence must be nonempty text or a nonempty list of text')
    return result
