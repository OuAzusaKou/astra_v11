"""Persistent Codex conversation; asynchronous turns, no direct robot access.

Only the runner accepts a completed, current request-bound decision. The worker
never writes response.json or publishes ROS actions. Switching cancels its turn.
"""
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import shlex
import subprocess
import sys
import threading
import time
import uuid

from .session import atomic_json
from .decision import normalize_decision

ROOT = Path(__file__).resolve().parents[1]


class CodexSupervisor:
    def __init__(self, session, timeout=600., binary=None, proxy=None, readonly_remote=None):
        self.session = session
        self.path = session.path / 'supervisor'
        self.path.mkdir(exist_ok=False)
        self.binary = binary or shutil.which('codex') or str(Path.home() / '.local/bin/codex')
        if not Path(self.binary).is_file():
            raise FileNotFoundError('Codex CLI missing; automatic supervisor unavailable')
        self.timeout = timeout
        self.proxy = proxy
        self.readonly_remote = readonly_remote
        self.thread_id = None
        self.worker = None
        self.cancel = threading.Event()
        self.result = None
        self.error = None
        self.request_id = None
        self.generation = 0
        self.log('STARTING: creating dedicated Astra conversation; medium; requested service_tier=fast; ' +
                 ('OpenAI transport=HTTPS' if platform.system() in ('Darwin', 'Windows') else 'transport=provider default'))

    def log(self, message):
        line = f'[{time.strftime("%H:%M:%S")}] {message}'
        print('[SUPERVISOR] ' + line, flush=True)
        with (self.session.path / 'supervisor.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')
        atomic_json(self.session.path / 'supervisor.json', {
            'thread_id': self.thread_id, 'owner': self.session.owner,
            'model_reasoning_effort': 'medium', 'requested_service_tier': 'fast', 'timeout_seconds': self.timeout,
            'openai_transport': 'https' if platform.system() in ('Darwin', 'Windows') else 'provider_default',
            'runner_session': str(self.session.path), 'request_id': self.request_id,
            'status': message, 'updated_at': time.time(),
            'resume_command': f'codex resume {self.thread_id}' if self.thread_id else None,
            'note': 'Do not concurrently resume while runner owns this conversation.'})

    def _report_errors(self, events, offset):
        # Read complete JSONL records only; never expose reasoning or tool output.
        with events.open(encoding='utf-8') as stream:
            stream.seek(offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line.endswith('\n'):
                    return position
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                error = event if event.get('type') == 'error' else event.get('item', {})
                if error.get('type') == 'error':
                    message = str(error.get('message', 'unknown transport error'))
                    # Codex emits this advisory as an error item, although the turn continues.
                    # Match narrowly so actual transport/compaction failures remain errors.
                    compaction_advisory = message.startswith(
                        'Heads up: Long threads and multiple compactions can cause the model to be less accurate.')
                    level = 'WARNING' if compaction_advisory else 'CALL_ERROR'
                    self.log(level + ': ' + message[:600].replace('\n', ' '))

    def _call(self, prompt, images, cancel, tag, bootstrap=False):
        output = self.path / (tag + '.json')
        events = self.path / (tag + '.events.jsonl')
        stderr = self.path / (tag + '.stderr.log')
        args = [self.binary, 'exec', '-c', 'approval_policy="never"',
                '-c', 'default_permissions="astra_observe"',
                '-c', 'permissions.astra_observe.extends=":read-only"',
                '-c', 'permissions.astra_observe.network.enabled=true',
                '-c', 'mcp_servers.node_repl.enabled=false',
                '-c', 'model_reasoning_effort="medium"',
                '-c', 'features.fast_mode=true', '-c', 'service_tier="fast"',
                '-m', 'gpt-6-astra']
        if platform.system() == 'Windows' and self.readonly_remote and not getattr(self, 'data_only_url', None):
            host, repo = self.readonly_remote
            # MCP runs as the authenticated host user. Tools are constrained by
            # readonly_query; do not grant the sandbox SSH key access.
            args += ['-c', 'mcp_servers.astra_readonly.command='+json.dumps(sys.executable),
                     '-c', 'mcp_servers.astra_readonly.args='+json.dumps([
                         '-u','-B',str(ROOT/'astra/readonly_mcp.py'),'--host',host,'--repo',repo]),
                     '-c', 'mcp_servers.astra_readonly.startup_timeout_sec=15',
                     '-c', 'mcp_servers.astra_readonly.tool_timeout_sec=20']
        if platform.system() in ("Darwin", "Windows"):
            # Avoid repeated WebSocket failures on the laptop OpenAI proxy path.
            # Use the existing OpenAI login with a per-call HTTP-only provider.
            args += ['-c', 'model_provider="astra_https"',
                     '-c', 'model_providers.astra_https.name="OpenAI HTTPS for Astra"',
                     '-c', 'model_providers.astra_https.wire_api="responses"',
                     '-c', 'model_providers.astra_https.requires_openai_auth=true',
                     '-c', 'model_providers.astra_https.supports_websockets=false']
        if getattr(self, 'data_only_url', None):
            # V10's relay replaces all model-visible input and rejects tool outputs.
            # Never attach robot inspection MCP to this private CLI process.
            index = args.index('mcp_servers.node_repl.enabled=false')
            del args[index-1:index+1]  # No incomplete MCP table under --ignore-user-config.
            args += ['--ignore-user-config', '--ignore-rules',
                     '-c', 'model_provider="astra_data_only"',
                     '-c', 'model_providers.astra_data_only.name="Astra data-only"',
                     '-c', 'model_providers.astra_data_only.wire_api="responses"',
                     '-c', 'model_providers.astra_data_only.requires_openai_auth=true',
                     '-c', 'model_providers.astra_data_only.supports_websockets=false',
                     '-c', 'model_providers.astra_data_only.base_url='+json.dumps(self.data_only_url),
                     '-c', 'model_providers.astra_data_only.request_max_retries=0',
                     '-c', 'model_providers.astra_data_only.stream_max_retries=0',
                     '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
                     '-c', 'features.shell_tool=false', '-c', 'features.plugins=false',
                     '-c', 'features.apps=false', '-c', 'features.hooks=false']
        if self.thread_id:
            args += ['resume', self.thread_id]
        args += ['--skip-git-repo-check', '--json', '-o', str(output)]
        if bootstrap:
            schema = self.path / 'bootstrap.schema.json'
            atomic_json(schema, {'type': 'object', 'properties': {'ready': {'type': 'boolean'}},
                                'required': ['ready'], 'additionalProperties': False})
            args += ['--output-schema', str(schema)]
        for path in images:
            args += ['--image', str(path)]
        args += ['-']
        # Interactive bash loads the same API environment as the operator's CLI.
        # Pass arguments without interpolation; do not source arbitrary prompt text.
        windows = platform.system() == 'Windows'
        job = None
        if windows:
            from .windows_process import native_codex, ProcessJob
            args[0] = native_codex(self.binary)
            command = args
        else:
            command = args if platform.system() == "Darwin" else ['bash', '-ic', 'exec "$@"', 'astra-supervisor', *args]
        env = os.environ.copy()
        if self.proxy:
            env.update(HTTPS_PROXY=self.proxy, HTTP_PROXY=self.proxy,
                       https_proxy=self.proxy, http_proxy=self.proxy)
        if getattr(self, 'data_only_url', None):
            env.update(NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
        with events.open('w', encoding='utf-8') as out, stderr.open('w', encoding='utf-8') as err:
            process = subprocess.Popen(command, cwd=self.session.path, stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, text=True, encoding='utf-8',
                                       start_new_session=not windows, env=env,
                                       **({'creationflags': subprocess.CREATE_NO_WINDOW} if windows else {}))
            try:
                if windows:
                    # Attach before supplying the prompt: no model/tool work may
                    # begin unless this call's process tree can be cancelled.
                    job = ProcessJob()
                    job.attach(process)
                process.stdin.write(prompt)
                process.stdin.close()
                deadline = time.monotonic() + self.timeout
                offset = 0
                while process.poll() is None:
                    offset = self._report_errors(events, offset)
                    if cancel.wait(.1):
                        raise InterruptedError('superseded supervisor turn')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Astra response timeout')
                self._report_errors(events, offset)
                if cancel.is_set():
                    raise InterruptedError('superseded supervisor turn')
                if process.returncode:
                    raise RuntimeError(f'Codex exit {process.returncode}; see {stderr}')
            finally:
                if windows:
                    if job is not None:
                        job.close()
                    if process.poll() is None:
                        process.kill()  # Also covers failed job creation/assignment.
                    process.wait(timeout=3)
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                elif process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
        for line in events.read_text(encoding='utf-8').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'thread.started':
                thread_id = str(uuid.UUID(event['thread_id']))
                if self.thread_id is not None and thread_id != self.thread_id:
                    raise RuntimeError('Codex resumed a different conversation')
                self.thread_id = thread_id
        if self.thread_id is None:
            raise RuntimeError('Codex did not report a conversation ID')
        return json.loads(output.read_text(encoding='utf-8'))

    def bootstrap(self, stop):
        prompt = ('This is a dedicated Astra robot supervisor conversation. '
            'Initialization handshake only: return ready without robot actions. Read-only inspection tools are allowed when needed; never modify files or robot state. '
            'The runner will send individual requests in later turns. Return {"ready":true}.')
        if platform.system() == 'Windows' and self.readonly_remote:
            prompt += (' Before reporting ready, use astra_readonly read_file to read lines 1-12 of '
                       + self.readonly_remote[1]+'/envs/leo_wuji_env_infer.py. '
                       'If the read fails return {"ready":false}. Use this MCP for remote inspection; '
                       'direct sandbox SSH cannot access the host user credentials.')
        result = self._call(prompt, [], stop, 'bootstrap', bootstrap=True)
        if result != {'ready': True}:
            raise RuntimeError('Astra handshake failed')
        if platform.system() == 'Windows' and self.readonly_remote:
            if not self._remote_read_verified():
                raise RuntimeError('Astra reported ready without a successful remote inspection; startup refused')
        self.log(f'READY thread_id={self.thread_id}; same conversation will handle every request')

    def _remote_read_verified(self):
        expected = self.readonly_remote[1]+'/envs/leo_wuji_env_infer.py'
        for line in (self.path/'bootstrap.events.jsonl').read_text(encoding='utf-8').splitlines():
            try:
                item=json.loads(line).get('item',{})
                if (item.get('type')!='mcp_tool_call' or item.get('server')!='astra_readonly'
                        or item.get('tool')!='read_file' or item.get('status')!='completed'
                        or item.get('error') or item.get('arguments',{}).get('path')!=expected):
                    continue
                for block in (item.get('result') or {}).get('content',[]):
                    if block.get('type')=='text':
                        payload=json.loads(block['text'])
                        if payload.get('path')==expected and payload.get('text') and not payload.get('error'):
                            return True
            except (ValueError,TypeError,KeyError):
                continue
        return False

    def start(self, request):
        if request.get('version') == 10:
            raise ValueError('V10 requires the dedicated data-only supervisor; legacy source-reading path refused')
        if self.worker is not None:
            raise RuntimeError('previous supervisor turn has not retired')
        self.request_id = request['request_id']
        self.cancel = threading.Event()
        self.result = self.error = None
        generation = self.generation
        self.log(f'WAITING request={self.request_id} stage={request["stage"]}')
        code_handoff = request.get('version') in (4, 8)
        model_request = dict(request)
        if code_handoff:
            model_request.pop('astra_pause_handback', None)
        instructions = 'astra/TAKEOVER_V4_V8.md' if code_handoff else 'astra/SUPERVISOR.md'
        prompt = (ROOT / instructions).read_text(encoding='utf-8') + '\n' + (
            'AUTOMATED BRIDGE TURN: inspect the attached current camera images and the exact state below, including optional explicitly labeled RealSense RGB/depth. '
            'You may use read-only tools before answering: inspect local files, robot URDF/configuration and logs, SSH to leo3 to read deployment files, and query ROS state, TF and parameters. '
            'The robot deployment is /home/liberai/Liberai_Deploy_astra and ROS workspace is /home/liberai/Desktop/colcon_ws on leo3; on Mac use SSH leo3, on leo3 read locally. '
            'Preserve all current request data and use it directly; do not reread whole request/event files to duplicate it. '
            'For large JSONL logs, parse and select event metadata or the specific needed fields/trajectory rows rather than raw tail/cat; a single line may contain huge arrays. '
            'Prefer focused queries and paginate as needed, with source paths, event/trajectory IDs and row ranges. No hard information budget: retrieve additional evidence whenever planning needs it. '
            'Reuse verified static configuration facts from this conversation with their source and frame/unit conventions; recheck when changed, contradicted or unavailable after compaction. Never reuse historical robot state as current state. '
            'Use bounded read commands and existing authentication; SSH must not update known_hosts (StrictHostKeyChecking=yes, UpdateHostKeys=no). Do not expose credentials. '
            'Do not create, modify or delete files; do not publish ROS messages, invoke state-changing services/actions, reset, launch/restart/kill robot processes, or write response files. '
            'Network access authorizes read-only inspection, not remote mutations. Do not wait indefinitely for observations. '
            'Final answer must be ONE JSON decision object, no markdown or shell commands in the final answer. '
            'The bridge alone submits this return value. All previous requests/actions are stale. '
            'Copy session_id, owner and request_id from CURRENT_REQUEST. Include operation, reason, evidence. '
            'reason and evidence MUST be nonempty JSON strings, not arrays or objects. Put multiple evidence points into one string separated by newline escapes. '
            'Allowed operations follow Controller.decide: takeover needs actions [T,52], new_attempt, '
            'change_reason for new attempts, short_batch_reason for fewer than 30 rows; takeover execute_steps <=30; WAM review/correct <=15. '
            '52D absolute target order: left EE xyz(m)+rotvec(rad), right EE xyz+rotvec in chest_link, '
            'left hand20 then right hand20 radians. Do not infer metric displacement from pixels alone. '
            'State arrays are newest-first: row 0 is current, later rows are historical samples. Differences within one array do not prove motion continues after the batch. '
            'Assess settling using the scheduled batch end and current row 0 versus targets and previous requests\' row 0; another observe must answer a specific unresolved settling question. '
            'In V2/V3 ASTRA mode independently plan new takeover actions from the current images, measured state and task. Do not use correct/edit. '
            'wam_context is control-learning history: relate commanded numbers to paired observations and subsequent measured feedback. It is NOT a future execution plan or a required trajectory to extend. '
            'WAM history may be empty or unsuitable for the current goal; neither alone is a pause reason. A calibration file or demonstration is not mandatory. '
            'Published rows are not proof of execution; unpublished/discarded rows provide no measured motion effects and no replay authorization. '
            'Compare age, conditioning versions, historical and current states before drawing conclusions about control effects. Cite trajectory IDs/rows when using this evidence. '
            'After each short batch, compare actual feedback with intended progress and independently plan the next takeover. Small progress alone does not require pause. '
            'Exact bowl coordinates and zero geometric uncertainty are not prerequisites for every motion. Relative image direction plus verified frame conventions, measured state and observed control effects can ground an intermediate action; do not invent exact metric geometry from pixels. '
            'An uncertain final bowl placement pose does not mean all movement is impossible. First assess grounded intermediate transport, clearance or view-improving poses using current state and observed control effects. Never force motion if no such path can be justified. '
            'ASTRA_RETRY is a human request for reassessment using fresh observations; compare the previous pause reason in retry_context. It is not confirmation that a hazard is cleared. '
            'Choose a grounded batch endpoint first, then use up to 30 coordinated waypoints to span the path from current pose to that endpoint. The final executed point must reach the stated batch endpoint, not leave it in unexecuted rows. '
            'Default to direct goal-directed coordinated x/y/z and justified orientation changes, integrating clearance into a continuous path. Do not split lift and individual coordinate axes into separate turns unless an obstacle, contact sequence, uncertainty or timing constraint requires it; explain that constraint. '
            'State the batch endpoint, expected total displacement and scheduled duration in reason/evidence strings. Do not invent metric goal coordinates from pixels or choose a tiny increment before choosing the endpoint. '
            'Use execution_timing for actual fps (default 20) and batch limit. The current interface has no custom duration or keyframe resampling: choose an endpoint achievable within this actual timing, and do not compress a long path into 1.5 seconds. '
            'When measured tracking, direction and grasp match the previous plan and current views support it, increase the next batch total travel; explain feedback and clearance evidence. More waypoints alone do not increase travel. '
            'Intended task contact is not an unintended collision. Use the current L2, live views, measured state and available demonstration to identify the intended contact part, purpose and object outcome. '
            'When observations and verified control mapping support a path, proactively execute task-directed approach, insertion, grasp, pull or push; do not pause solely because contact is needed, exact coordinates are unavailable, or uncertainty is nonzero. '
            'Plan meaningful continuous local goals rather than defaulting to hovering, repeated static observation or negligible probes. Adapt travel and feedback frequency to the required contact precision, available duration and observed response; never force larger jumps. '
            'Plan around completing an operation stage, not a fixed centimetre increment. For a drawer, when observations, verified control mapping and batch duration support it, prefer one coordinated batch to approach and insert into a graspable position. '
            'After fresh feedback confirms the required contact or grasp, prefer a continuous pull to the opening required by the current L2 in the next batch. Published insertion targets alone do not confirm contact. '
            'Do not default to repeated 2-3 cm probes or separate-axis turns. Describe the operation stage, intended object/hand endpoint state and next feedback check. '
            'If duration or a necessary feedback boundary requires more batches, continue toward the same stage endpoint without forcing speed or skipping contact confirmation. This is not a minimum-distance rule or a mandatory two-batch budget. '
            'A pause must identify a specific unresolved hazard, abnormal state, or essential missing action evidence after reassessment, and explain which part of the next path it blocks. Generic possible-collision language alone is insufficient. '
            'First assess a grounded task-progress or information-gathering action. Do not invent clearance, force feedback or target geometry; operator emergency-stop availability is not evidence that a path is feasible. '
            'Use up to 30 executed takeover waypoints and explain shorter batches. '
            'If a plan is not ready, use observe with the specific question to reassess; the bridge obtains fresh observations and calls you again, retaining ASTRA ownership without motion. '
            'Do not loop identical observations without a reason; report concrete missing information if it remains unresolved. '
            'Pause for a concrete unresolved physical hazard, hardware/sensor failure or a specific remaining inability to ground motion after reassessment, not merely because no WAM continuation fits. '
            'Never generate arbitrary targets just to make the bridge appear functional. '
            'A pause decision has no actions. Use complete only with visual success evidence. '
            'RETAIN TAKEOVER OWNERSHIP by default: independently attempt grounded strategies, execute bounded version-appropriate actions, inspect real feedback and revise ineffective strategies. '
            'Keep working on the current L2 when a grounded next step exists, using real feedback to revise the plan. '
            'Use targeted queries or observe for a specific unresolved question; reuse verified static facts instead of repeatedly searching the whole repository or other experiment decisions. '
            'Before a planning pause, state actual independent attempts and observed results, alternatives assessed and the specific blocker; if no attempt was grounded, say so honestly. '
            'No forced attempt count: immediately pause for a concrete unmanageable hazard or hardware fault; never invent actions or clearance just to keep trying. '
            'If recovery is incomplete continue version-appropriate takeover/observe (V8: new candidate set and Dynamics review); if the whole L3 is done use complete. '
            'For V1 condition includes demo_context as required by the supervisor instructions. '
            '\nCURRENT_REQUEST:\n' + json.dumps(model_request, ensure_ascii=False, separators=(',', ':')))
        if platform.system() == 'Windows' and self.readonly_remote:
            prompt = ('WINDOWS REMOTE INSPECTION: use the astra_readonly MCP tools for all Leo3 reads. '
                      'The shell sandbox cannot read the user SSH credentials, so do not run ssh there '
                      'or try to read .ssh/config or private keys. This is not an unavailable robot: '
                      'the bounded MCP uses the authenticated bridge user. Use list_directory, search_text, '
                      'read_file and ros_query. The repo directory is config (singular); Leo3 has no rg. '
                      'Control definitions: envs/leo_wuji_env_infer.py (chest_link to left_link_ee/right_link_ee, '
                      'finger order and publishers), astra/robot.py, utils/liber_teleop_v22_client.py; '
                      'inspect ROS workspace URDF origins/axes and live TF as needed. These are source hints, '
                      'not an assertion of axis directions, grasp or clearance.\n') + prompt
        images = [v['path'] for v in request['observation']['images'].values()]
        image_labels = ['CURRENT ' + key for key in request['observation']['images']]
        depth = request['observation'].get('depth')
        if depth is not None:
            prompt += ('\nDEPTH-JUDGMENT MODE: Use depth only for visible-surface optical near/far judgments. '
                       'No XYZ/deprojection, extrinsic calibration, robot target conversion, or interpreting depth delta as arm travel/Euclidean clearance. '
                       'Query ROIs only in realsense_rgb pixel coordinates, not in the original three cameras. '
                       'Black/zero depth is unknown. Small depth difference does not prove contact or grasp. '
                       'Cross-frame comparison requires the same camera, stable viewpoint/background and matched surfaces. '
                       'Unavailable or mixed depth is a local evidence gap, not an automatic pause of all task motion.\n')
            if depth.get('status') == 'available':
                command = [sys.executable, '-B', str(ROOT / 'astra/depth_query.py'),
                           '--metadata', depth['metadata_path']]
                prompt += ('READ-ONLY DEPTH QUERY (use before claiming metric depth or pausing over unresolved near/far): '
                           + shlex.join(command) + ' --roi X0 Y0 X1 Y1 [--roi X0 Y0 X1 Y1]\n'
                           'Replace each ROI with integer pixels from the attached RealSense RGB; bounds are end-exclusive. '
                           'The query reads the immutable current observation and prints depth percentiles/validity, no files or robot changes. '
                           'Preview colors are fixed red-near/blue-far, black-invalid; numerical evidence comes from the query.\n')
        for record in request.get('wam_context', {}).get('trajectories', []):
            for key, image in record['observation']['images'].items():
                images.append(image['path'])
                image_labels.append('WAM ' + record['trajectory_id'] + ' ' + key)
        imagined = request.get('imagination') or {}
        if request.get('version') == 8:
            prompt += ('\nV8 OVERRIDES the earlier trajectory-generation, coordinated-axis and takeover instructions. '
                       'First choose the operating hand from CURRENT real images, measured state and current L2. '
                       'Return operation=propose_candidates, active_hand=left or right, reason and evidence; NO actions. '
                       'The runner creates six chest_link +/-X +/-Y +/-Z translations for that hand, preserving '
                       'orientation, both finger configurations and the other arm, plus ONE fresh full-body policy '
                       'candidate from the SAME real anchor. Policy may move both hands and fingers. '
                       'Runner validates and predicts candidates serially through Dynamics; none is executed yet. '
                       'At REVIEW_CANDIDATES compare every available labeled SYNTHETIC FUTURE with real views and '
                       'current task, considering task progress, grasp/contact and visible hazards. Failed/rejected '
                       'candidates cannot be selected. Choose exactly ONE status=predicted candidate with operation=approve, '
                       'candidate_set_id, candidate_id and candidate_sha256 copied exactly, execute_steps equal to '
                       'len(selected.actions), reason and evidence explaining this choice relative to alternatives. '
                       'No actions, coordinate edits, concatenation or shortened prefixes; takeover/correct are forbidden. '
                       'If no candidate is grounded, use observe, propose_candidates to select the other hand/rebuild, '
                       'or pause with a concrete reason. Predictions do not certify collision safety, measured contact '
                       'or task completion. Canonical normalized predicted state is not a metric robot command. '
                       'After one execution, VERIFY_RECOVERY provides REAL feedback; choose a hand and build a NEW set '
                       'if recovery should continue. All unselected candidates are discarded. '
                       'Use complete only with real evidence that the entire L3 task is finished.\n')
        if request.get('version') == 5:
            prompt += ('\nV5: takeover actions are candidates, not published until imagination review. '
                       'Prefer a trajectory prefix lasting 0.5–1 second including lead time; runner may truncate longer proposals. '
                       'REVIEW_IMAGINATION: attached SYNTHETIC FUTURE is a prediction, never evidence of physical completion. '
                       'Compare it with the current REAL views and target L2. To accept, return operation=approve, '
                       'candidate_id and candidate_sha256 copied exactly, execute_steps equal to the candidate action count, '
                       'reason and evidence. Do not include actions. To revise, return a new takeover trajectory; '
                       'to refresh real observations, return observe. hand_back/complete must rely on REAL evidence only. '
                       'Future state is normalized canonical data, not a robot command or metric pose.\n')
        if request.get('version') == 6:
            prompt += ('\nV6: Astra candidates first animate URDF-sampled hand clouds against a frozen depth scene. '
                       'Propose up to 1 second including lead; longer batches may be truncated. '
                       'Attached top/front/side projection sheets show ordered candidate steps, cyan anchor hands and orange imagined hands. '
                       'These are geometric projections of the SAME points shown in RViz, not camera images or dynamics predictions. '
                       'Scene objects remain frozen, including observed robot surfaces. Occlusion and calibration uncertainty remain; '
                       'do not infer transported objects, successful grasp, collision-free motion, or actual completion from these images. '
                       'Review displacement, hand shape, approach direction and visible scene clearance with the REAL camera views. '
                       'At REVIEW_IMAGINATION accept with operation=approve, exact candidate_id and candidate_sha256, '
                       'execute_steps equal to the entire candidate length, reason and evidence, and NO actions. '
                       'Revise via a new takeover (which is imagined again), or observe to refresh. '
                       'Only REAL post-execution observations justify hand_back or completion.\n')
        if request.get('version') == 7:
            prompt += ('\nV7 OVERRIDES earlier 30-row and endpoint-execution instructions: plan a continuous remaining-current-L2 route '
                       'to the next grounded observation milestone (e.g. above bowl), up to 120 dense 52D rows at execution_timing.fps. '
                       'Do not assume grasp succeeded. Ground current grasp, targets and clearance in real observations. '
                       'takeover execute_steps must equal full submitted rows; this only requests imagination, never direct motion. '
                       'Include plan_stages=[{"kind":"transport","end_step":60,"evidence":"real grasp and free-space evidence"}]. '
                       'Kinds: transport, approach, contact, release. end_step is exclusive, strictly increasing and covers all rows. '
                       'Separate contact/release boundaries. All rows obey joint/jump guards. Use <=0.15 m/s translation and <=0.8 rad/s rotvec changes. '
                       'Transport preserves fingers within 0.05 rad. Plan fewer than 120 rows when justified with short_batch_reason. '
                       'Runner previews the WHOLE plan; REVIEW_IMAGINATION approves only 1..imagination.max_execute_steps rows. '
                       'Prefix is bounded by first stage, remaining budget, 1.5s/0.18m transport or 0.5s/0.04m otherwise, including lead. '
                       'Review attached full-plan and executable-prefix geometry and REAL views. These are frozen-scene URDF hand projections, '
                       'not object dynamics, full-arm IK or collision certification. Held objects are not modeled as swept volumes. '
                       'approve requires exact candidate_id and candidate_sha256, execute_steps, reason/evidence and NO actions. '
                       'Shorten the prefix if uncertain, observe or replan if no grounded prefix exists. '
                       'After execution the tail is discarded; VERIFY_RECOVERY uses fresh measurements to replan or hand_back early '
                       'when WAM can continue current L2. Do not replay the tail or use synthetic progress as completion evidence.\n')
        for frame in imagined.get('images', []):
            images.append(frame['path'])
            image_labels.append(('EXECUTABLE PREFIX' if Path(frame['path']).name.startswith('prefix_') else 'FULL CANDIDATE')
                                + ' / SYNTHETIC, NOT OBSERVED / candidate='
                                + frame.get('candidate_id', imagined.get('candidate_id', ''))
                                + ' ' + frame.get('label', ''))
        reference = request.get('reference_video', {})
        if reference.get('images'):
            self.log(f'REFERENCE VIDEO request={self.request_id} frames={len(reference["images"])} source={reference.get("source_path", "unknown")}')
            prompt += ('\nREFERENCE VIDEO CONTEXT: The final attached images are chronological frames '
                       'from an operator-provided demonstration, not live observations or WAM predictions. '
                       'Use them to understand task sequence, manipulation and intended object outcome, '
                       'in conjunction with the current L3/L2. Uniform samples are not labeled L2 boundaries. '
                       'Do not treat demonstrated poses as current robot state or directly replay them.\n')
            for frame in reference['images']:
                images.append(frame['path'])
                image_labels.append(f"DEMONSTRATION frame={frame['frame_index']} time_seconds={frame['time_seconds']}")
        if request.get('astra_pause_handback') and not code_handoff:
            prompt += ('\nLAST-RESORT OPERATOR HANDOFF POLICY FOR THIS RUN: Prioritize independent recovery and retain ownership as described above. '
                       'This fallback is NOT a preferred strategy, and is not a shortcut for asking WAM to solve the same stagnation. '
                       'First assess grounded independent alternatives and relevant feedback; do not force risky attempts or invent evidence. '
                       'Only when you cannot reasonably continue and choose to pause your own planning, '
                       'including for collision risk or missing geometric evidence, return operation=pause with '
                       'pause_kind=planning, collision_risk, or observation_uncertainty as appropriate. '
                       'The runner records your unresolved concern, discards Astra actions, and immediately hands '
                       'control to low-policy/WAM using fresh observations, keeping L2 and budget. '
                       'This is an operator-requested fallback, NOT evidence of recovery or safety; do not claim either. '
                       'For hardware faults or ESTOP use pause_kind=hardware_fault; those remain paused and never hand back. '
                       'Do not put actions in a pause. Explain actual attempts/results, or the concrete reason no attempt was grounded. '
                       'This changes only what the runner does AFTER a justified nonhardware pause; it does not weaken the preference to keep trying independently. '
                       'Normal recovered hand_back still requires true recovery_evidence and a reason WAM should take over the remaining work.\n')
        prompt += '\nATTACHED IMAGE ORDER:\n' + json.dumps(image_labels)
        def work():
            try:
                result = self._call(prompt, images, self.cancel, self.request_id)
                if generation == self.generation and not self.cancel.is_set():
                    for key in ('session_id', 'owner', 'request_id'):
                        if result.get(key) != request[key]:
                            raise ValueError(f'Astra returned mismatched {key}')
                    self.result = normalize_decision(result)
            except InterruptedError:
                pass
            except Exception as error:
                if generation == self.generation:
                    self.error = str(error)
        self.worker = threading.Thread(target=work, name='astra-codex-supervisor', daemon=True)
        self.worker.start()

    def invalidate(self):
        self.generation += 1
        self.cancel.set()
        self.result = self.error = None
        self.request_id = None

    def poll(self, request):
        """Return results only for current request; never block keyboard handling."""
        wanted = request['request_id'] if request else None
        if self.request_id is not None and self.request_id != wanted:
            self.invalidate()
            self.log('CANCELLED: ownership/request changed; stale response discarded')
        if self.worker is not None and not self.worker.is_alive():
            self.worker.join()
            self.worker = None
            result, error = self.result, self.error
            self.result = self.error = None
            if error:
                self.log('FAILED: ' + error)
                raise RuntimeError(error)
            if result is not None and wanted == self.request_id:
                self.request_id = None
                self.log('RESPONSE: ' + str(result.get('operation')) + ' | ' + str(result.get('reason')))
                return result
        if request and self.worker is None and self.request_id is None:
            self.start(request)
        return None

    def close(self):
        self.invalidate()
        if self.worker:
            self.worker.join(timeout=8)
        self.log('STOPPED: runner no longer accepts decisions')
