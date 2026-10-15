"""V11 data-only planning sessions, independent of the V4/V8 source-reading prompts."""
import copy
import json
import threading
import time
import uuid

from .action_chunk_v11 import CONTRACT, INSTRUCTIONS, public_input, schema, validate
from .api_supervisor import QuickRouterSupervisor, parse_decision
from .v10_images import image_block
from .session import atomic_json
from .supervisor import CodexSupervisor


class ChunkMixin:
    def log(self, message):
        if message.startswith('STARTING:'):
            message = 'STARTING: V11 data-only session; model tools and source context excluded'
        super().log(message)
        path = self.session.path/'supervisor.json'
        info = json.loads(path.read_text(encoding='utf-8'))
        info.update(control_version=11, action_chunk_contract=CONTRACT, inspection_mode='data_only', model_tools_enabled=False)
        if 'requested_service_tier' in info:
            info['requested_service_tier'] = 'default'
        atomic_json(path, info)

    def bootstrap(self, stop):
        self.data_history = []
        # A fresh session never inherits a V4/V8 tool conversation or control source.
        content = [{'type': 'text', 'text': 'Initialization only. Return {"action_chunk": []}. No actions.'}]
        ready_schema = schema()
        ready_schema['properties']['action_chunk']['maxItems'] = 0
        reply = self.exchange(content, ready_schema, stop, 'bootstrap')
        if reply != {'action_chunk': []}:
            raise ValueError('V11 data-only handshake failed')
        self.commit(content, reply)
        self.log('READY: V11 data-only action chunks; no model tools or controller source; conversation='+self.thread_id)

    def commit(self, content, reply):
        self.data_history.extend([{'role': 'user', 'content': content},
                                  {'role': 'assistant', 'content': json.dumps(reply, ensure_ascii=False)}])
        atomic_json(self.path/'data_history.json', dict(conversation_id=self.thread_id,
                                                       messages=self.data_history))

    def recent_history(self):
        """Keep two turns; demo and prior execution sequences appear only in the current request."""
        history = copy.deepcopy(self.data_history[-4:])
        for message in history:
            blocks = message['content']
            if message['role'] != 'user' or not isinstance(blocks, list) or not blocks:
                continue
            if blocks[0].get('type') != 'text':
                continue
            try:
                public = json.loads(blocks[0]['text'])
            except (ValueError, TypeError):
                continue  # Initialization has no observation metadata or demo frames.
            if not isinstance(public, dict):
                continue
            labels = public.get('image_order', [])
            demo = {i for i, label in enumerate(labels) if label.startswith(('DEMONSTRATION ', 'EXECUTION '))}
            if 'execution_feedback' in public:
                public.pop('execution_feedback')
                blocks[0]['text'] = json.dumps(public, ensure_ascii=False, allow_nan=False)
            if not demo:
                continue
            if sum(block.get('type') == 'image_url' for block in blocks) != len(labels):
                raise ValueError('V11 historical image labels do not match attachments')
            kept, image_index = [], 0
            for block in blocks:
                if block.get('type') == 'image_url':
                    include = image_index not in demo
                    image_index += 1
                    if not include:
                        continue
                kept.append(block)
            public['image_order'] = [label for i, label in enumerate(labels) if i not in demo]
            public['demonstration'] = []
            kept[0]['text'] = json.dumps(public, ensure_ascii=False, allow_nan=False)
            message['content'] = kept
        return history

    def request_messages(self, content):
        """Reuse identical trajectory/camera images, preferring the latest request."""
        messages = self.recent_history() + [{'role': 'user', 'content': copy.deepcopy(content)}]
        sources = {}
        for message in reversed(messages):
            blocks = message['content']
            if message['role'] != 'user' or not isinstance(blocks, list) or not blocks:
                continue
            if blocks[0].get('type') != 'text':
                continue
            try:
                public = json.loads(blocks[0]['text'])
            except (ValueError, TypeError):
                continue
            if not isinstance(public, dict) or not public.get('request_id'):
                continue
            labels = public.get('image_order', [])
            if not any(label.startswith('HISTORICAL ') for label in labels):
                continue
            if sum(b.get('type') == 'image_url' for b in blocks) != len(labels):
                raise ValueError('V11 WAM image labels do not match attachments')
            kept, kept_labels, references, image_index = [], [], [], 0
            for block in blocks:
                if block.get('type') == 'image_url':
                    label = labels[image_index]
                    image_index += 1
                    if label.startswith('HISTORICAL '):
                        # Same trajectory and camera AND exact bytes; never infer visual similarity.
                        key = (label, block['image_url']['url'])
                        if key in sources:
                            references.append({'image_label': label, **sources[key]})
                            continue
                        sources[key] = {'source_request_id': public['request_id'],
                                        'source_image_label': label}
                    kept_labels.append(label)
                kept.append(block)
            if references:
                public['image_order'] = kept_labels
                public['reused_wam_images'] = references
                public['wam_image_reuse_semantics'] = (
                    'These historical WAM images are attached once in the referenced request in this input. '
                    'They are the same trajectory/camera and exact image bytes, not fresh observations. '
                    'State, actions and publication status remain specific to each original record.')
                kept[0]['text'] = json.dumps(public, ensure_ascii=False, allow_nan=False)
                message['content'] = kept
        return messages

    def start(self, request):
        if request.get('version') != 11:
            raise ValueError('data-only supervisor requires V11')
        if self.worker is not None:
            raise RuntimeError('previous V11 turn has not retired')
        self.request_id = request['request_id']
        self.cancel = threading.Event()
        self.result = self.error = None
        generation, cancel, request = self.generation, self.cancel, copy.deepcopy(request)
        public, images = public_input(request)
        content = [{'type': 'text', 'text': json.dumps(public, ensure_ascii=False, allow_nan=False)}]
        content += [image_block(path) for path in images]
        atomic_json(self.path/(self.request_id+'.public.json'), public)
        self.log(f'WAITING V11 request={self.request_id}; images={len(images)}; tools=0')
        def work():
            try:
                reply = self.exchange(content, schema(), cancel, request['request_id'])
                reply = validate(reply, request)
                if generation == self.generation and not cancel.is_set():
                    self.commit(content, reply)
                    self.result = {k: request[k] for k in ('session_id', 'owner', 'request_id')}
                    self.result.update(operation='v11_action_chunk', chunk_reply=reply,
                                       reason=f"Host: received {len(reply['action_chunk'])} action rows",
                                       evidence='Host-bound response; no model explanation requested')
            except InterruptedError:
                pass
            except Exception as error:
                if generation == self.generation:
                    self.error = str(error)
        self.worker = threading.Thread(target=work, name='astra-v11-data-only', daemon=True)
        self.worker.start()


class ChunkApiSupervisor(ChunkMixin, QuickRouterSupervisor):
    def exchange(self, content, output_schema, cancel, tag):
        messages = [{'role': 'system', 'content': INSTRUCTIONS}]
        messages += self.request_messages(content)
        payload = dict(model=self.model, messages=messages, tools=[], tool_choice='none',
                       stream=False, max_tokens=16384,
                       response_format={'type': 'json_schema', 'json_schema': {
                           'name': 'action_chunk', 'strict': True, 'schema': output_schema}})
        response = self._http(payload, cancel, time.monotonic()+self.timeout)
        atomic_json(self.path/(tag+'.raw.json'), response)
        choices = response.get('choices', [])
        if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
            raise ValueError('V11 incomplete model response')
        message = choices[0].get('message', {})
        if message.get('role') != 'assistant' or message.get('tool_calls') or message.get('function_call'):
            raise ValueError('V11 tool calls are forbidden; no action released')
        reply = parse_decision(message.get('content'))
        atomic_json(self.path/(tag+'.reply.json'), reply)
        return reply


class ChunkCodexSupervisor(ChunkMixin, CodexSupervisor):
    def exchange(self, content, output_schema, cancel, tag):
        from .data_only_relay import DataOnlyRelay, response_payload
        messages = []
        for message in self.request_messages(content):
            blocks = message['content']
            if isinstance(blocks, str):
                blocks = [{'type': 'text', 'text': blocks}]
            converted = []
            for block in blocks:
                if block['type'] == 'text':
                    converted.append({'type': 'output_text' if message['role'] == 'assistant' else 'input_text',
                                      'text': block['text']})
                else:
                    converted.append({'type': 'input_image', 'image_url': block['image_url']['url']})
            messages.append({'role': message['role'], 'content': converted})
        payload = response_payload(INSTRUCTIONS, messages, output_schema)
        # Log exactly the data sent, not the CLI's hidden agent context or auth headers.
        atomic_json(self.path/(tag+'.model_input.json'), payload)
        with DataOnlyRelay(payload, self.proxy, cancel, self.timeout,
                           log=lambda message: self.log(f'{message}; request={tag}')) as relay:
            self.data_only_url = relay.url
            try:
                reply = CodexSupervisor._call(self, 'Return the data-only JSON response.', [], cancel, tag)
            except Exception:
                if relay.error:
                    raise RuntimeError('V11 transport boundary: '+relay.error) from None
                raise
            finally:
                self.data_only_url = None
        atomic_json(self.path/(tag+'.reply.json'), reply)
        return reply
