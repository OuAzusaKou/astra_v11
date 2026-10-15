"""Assemble complete OpenAI-compatible SSE turns; never expose partial decisions."""
import json

LIMIT = 16_000_000


class IncompleteStream(ConnectionError):
    pass


def merge_delta(target, delta):
    for key, value in delta.items():
        if value is None:
            continue
        if isinstance(value, str):
            if key in ('role', 'type'):
                if key in target and target[key] != value:
                    raise ValueError('stream identity changed')
                target[key] = value
            else:
                target[key] = target.get(key, '') + value
        elif isinstance(value, dict):
            merge_delta(target.setdefault(key, {}), value)
        elif isinstance(value, list):
            dest = target.setdefault(key, [])
            for item in value:
                if isinstance(item, dict) and 'index' in item:
                    index = item['index']
                    if type(index) is not int or not 0 <= index < 1024:
                        raise ValueError('invalid stream item index')
                    while len(dest) <= index:
                        dest.append({})
                    merge_delta(dest[index], {k:v for k,v in item.items() if k != 'index'})
                else:
                    dest.append(item)
        elif key not in target or target[key] == value:
            target[key] = value
        else:
            raise ValueError('inconsistent stream metadata')


def read_completion(response, emit=lambda **kw: None):
    message = {}
    result = {}
    finish = None
    fields = []
    total = events = 0
    done = False

    def consume(data):
        nonlocal finish, events
        if data == '[DONE]':
            return True
        obj = json.loads(data)
        if not isinstance(obj, dict) or 'error' in obj:
            raise RuntimeError('upstream stream error')
        events += 1
        if events == 1:
            emit(event='stream_first_event', stage='read_stream_body')
        for key in ('id', 'model', 'created', 'usage', 'system_fingerprint'):
            if key in obj:
                result[key] = obj[key]
        choices = obj.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError('expected one streamed choice')
        if choices:
            choice = choices[0]
            if choice.get('index', 0) != 0:
                raise ValueError('unexpected stream choice index')
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                raise ValueError('invalid stream delta')
            if finish is not None and delta:
                raise ValueError('stream continued after finish')
            merge_delta(message, delta)
            if choice.get('finish_reason') is not None:
                if finish is not None and finish != choice['finish_reason']:
                    raise ValueError('stream finish reason changed')
                finish = choice['finish_reason']
        return False

    while True:
        raw = response.readline(LIMIT + 1)
        if not raw:
            if fields:
                done = consume('\n'.join(fields))
            break
        total += len(raw)
        if total > LIMIT:
            raise ValueError('stream response exceeds transport limit')
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if fields:
                done = consume('\n'.join(fields))
                fields = []
                if done:
                    break
        elif line.startswith('data:'):
            fields.append(line[5:].removeprefix(' '))
        # SSE comments/keepalives and event/id fields do not carry model text.
    if not done or finish is None:
        raise IncompleteStream('stream ended before finish and DONE')
    if message.get('role') != 'assistant':
        raise ValueError('invalid streamed assistant role')
    emit(event='stream_complete', stage='read_stream_body', response_bytes=total, stream_events=events)
    return {**result, 'choices':[{'index':0, 'message':message, 'finish_reason':finish}]}
