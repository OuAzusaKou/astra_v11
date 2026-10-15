"""Synchronous review boundaries above the existing Leo 52D chunk publishers."""
from dataclasses import asdict, dataclass
import time
import copy
import uuid

import numpy as np

from .session import atomic_json
from .decision import normalize_decision


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def actions52(value):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 52 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("actions must be finite nonempty [T,52] absolute robot targets")
    return arr.copy()


@dataclass(frozen=True)
class Settings:
    original_step_limit: int
    step_limit_multiplier: int = 3
    monitor_steps: int = 15
    takeover_steps: int = 30
    stagnation_steps: int = 128
    review_timeout: float = 900.
    version: int = 1
    astra_pause_handback: bool = False
    no_step_limit: bool = False

    def __post_init__(self):
        if type(self.version) is not int or self.version not in (1, 2, 3, 4, 5, 6, 7, 8, 10):
            raise ValueError("version must be 1 (review), 2 (keyboard), 3 (Astra hand-back) 4 (VLM L2) 5 (dynamics) 6 (pointcloud) 7 (bounded prefix) or 8 (direction + policy candidates)")
        for key in ("original_step_limit", "step_limit_multiplier", "monitor_steps", "takeover_steps", "stagnation_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.no_step_limit and self.version not in (4, 10):
            raise ValueError("no_step_limit currently requires V4 or V10")
        if not np.isfinite(self.review_timeout) or self.review_timeout <= 0:
            raise ValueError("review_timeout must be finite and positive")

    @property
    def budget(self):
        return None if self.no_step_limit else self.original_step_limit * self.step_limit_multiplier


class Controller:
    def __init__(self, robot, policy, session, settings, instruction, initial_l2=None, next_subtasks=()):
        self.robot, self.policy, self.session, self.settings = robot, policy, session, settings
        self.l3 = text(instruction, "instruction")
        self.initial_l2 = text(initial_l2, "initial_l2") if initial_l2 is not None else None
        self.subtasks = [self.initial_l2] + [text(v, "next subtask") for v in next_subtasks]
        self.subtask_index = 0
        self.l2 = None
        self.condition_version = 0
        self.demo = None
        self.state, self.mode = "STARTING", "WAM"
        self.wam_history = []
        self.latest_execution = None
        self.wam_active = None
        self.wam_offset = 0
        self.queue = np.empty((0, 52))
        self.source = None
        self.steps = self.takeovers = self.attempts = self.corrections = self.replans = 0
        self.published = 0
        self.last_progress = 0
        self.pause_id = None
        self.last_pause_reason = None
        self.supervisor_hardware_hold = False
        self.retry_context = None
        self.takeover_trigger = None
        self.result = None
        self.observation = None
        self.cancelled = lambda: False
        self.switch_pending = lambda: False
        self.vlm_takeover = None
        self.vlm_gate = None
        self.imagination = None
        self.approved_imagination = None
        self.plan_stages = None
        self.phase = 0
        self.session.event("settings", **asdict(settings), budget=settings.budget)

    @property
    def prompt(self):
        return f"Task: {self.l3} Subtask: {self.l2}"

    def discard(self):
        self.approved_imagination = None
        if self.imagination is not None:
            self.imagination.cancel()
        if self.wam_active is not None:
            self.wam_active["row_status"] = [
                "not_published_discarded" if v == "not_published" else v
                for v in self.wam_active["row_status"]]
        self.wam_active = None
        self.wam_offset = 0
        self.queue = np.empty((0, 52))
        self.source = None

    def wam_context(self):
        records = copy.deepcopy(self.wam_history)
        now = time.time()
        for record in records:
            record["age_seconds"] = max(0., now - record["inferred_at"])
            record["matches_current_condition"] = record["conditioning_version"] == self.condition_version
        return {"reference_only": True, "physical_execution_confirmed": False,
                "purpose": "control_learning_history_not_future_plan",
                "effect_evidence": "relate published targets to subsequent measured observations; unpublished rows have no observed execution effect",
                "row_status_semantics": "published means publisher returned, not measured execution; publication_attempted may be partially sent; not_published/discarded never authorized for replay",
                "trajectories": records}

    def status(self):
        self.session.status(state=self.state, control_mode=self.mode, steps=self.steps,
                            pause_id=self.pause_id, condition_version=self.condition_version,
                            version=self.settings.version, phase=self.phase, l2_instruction=self.l2,
                            subtask_index=self.subtask_index, subtasks=self.subtasks,
                            switch_modes=getattr(self, 'switch_modes', None),
                            vlm_auto_l2=self.vlm_gate.status if self.vlm_gate else None,
                            vlm_auto_takeover=self.vlm_takeover.status if self.vlm_takeover else None)

    def transition(self, target, reason):
        previous = self.mode if self.phase else "START"
        self.mode = target
        self.phase += 1
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] [PHASE {self.phase:03d}] {previous} -> {target} | step={self.steps} | {reason}", flush=True)
        self.session.event("control_transition", phase=self.phase, previous=previous,
                           current=target, reason=reason, steps=self.steps, l2_instruction=self.l2)

    def keyboard_toggle(self, source="keyboard", evidence=None):
        if self.settings.version not in (2, 3, 4, 5, 6, 7, 8, 10):
            raise ValueError("keyboard toggle requires version 2, 3 or 4")
        if self.state in ("FINISHED", "EXITED", "RESETTING", "WAITING_FOR_START") or (
                self.state == "PAUSED_FOR_OPERATOR" and self.mode != "ASTRA"):
            self.session.event("toggle_ignored", state=self.state)
            print(f"[CONTROL] switch ignored in {self.state}; it cannot clear a fault", flush=True)
            return
        self.robot.check()
        self.discard()
        self.session.pending = None
        if self.mode == "WAM":
            self.takeovers += 1
            self.takeover_trigger = {"source": source, "evidence": evidence}
            self.transition("ASTRA", f"{source} takeover; WAM queue discarded")
            self.review("ASTRA_TAKEOVER")
            print(f"[ASTRA WAITING] Automatic supervisor processing | session={self.session.path}", flush=True)
        else:
            self.pause_id = None
            self.transition("WAM", "keyboard hand-back; fresh inference required")
            self.state = "AUTO"
            self.status()

    def retry_astra(self, source="operator"):
        if self.settings.version not in (2, 3, 4, 5, 6, 7, 8, 10) or self.mode != "ASTRA" or self.state != "PAUSED_FOR_OPERATOR":
            self.session.event("astra_retry_ignored", state=self.state, control_mode=self.mode)
            print("[CONTROL] R requires a paused V2/V3/V4 Astra session", flush=True)
            return False
        # Do not rearm or bypass the hardware fault latch.
        self.robot.check()
        self.retry_context = {"operator_requested": source == "operator", "retry_source": source,
                              "previous_pause_reason": self.last_pause_reason,
                              "instruction": "Reassess fresh observations; retry is not proof that the previous hazard is resolved."}
        self.discard()
        self.session.pending = None
        self.pause_id = None
        self.session.event("operator_astra_retry" if source == "operator" else "transport_astra_retry",
                           previous_pause_reason=self.last_pause_reason, steps=self.steps)
        self.review("ASTRA_RETRY")
        print(f"[ASTRA RETRY] source={source}; fresh observation requested; ASTRA retains control; no immediate motion", flush=True)
        return True

    def next_subtask(self, source="operator"):
        if self.settings.version not in (2, 3, 4, 5, 6, 7, 8, 10) or self.mode != "WAM" or self.state != "AUTO":
            print("[SUBTASK] P requires V2/V3/V4 WAM execution; return with T first", flush=True)
            return False
        if self.subtask_index + 1 >= len(self.subtasks):
            print("[SUBTASK] Already at the final subtask; no change", flush=True)
            return False
        self.robot.check()
        self.discard()
        self.session.pending = None
        self.subtask_index += 1
        self.l2 = self.subtasks[self.subtask_index]
        self.condition_version += 1
        self.observation = None
        self.state = "WAITING_FOR_START"
        self.session.event("vlm_subtask_selected" if source == "vlm" else "operator_subtask_selected", index=self.subtask_index,
                           l2_instruction=self.l2, steps=self.steps)
        self.status()
        confirmation = "VLM confirmed; starting next L2" if source == "vlm" else "Press Enter to confirm"
        print(f"[SUBTASK {self.subtask_index+1}/{len(self.subtasks)}] {self.l2} | {confirmation}", flush=True)
        return True

    def try_vlm_subtask(self):
        """Called on the runner thread after drain; never send keyboard input or motion."""
        if self.vlm_gate is None:
            return False
        before = self.vlm_gate.status
        evidence = self.vlm_gate.poll(self)
        if before != self.vlm_gate.status:
            self.status()
        if evidence is None or self.switch_pending() or self.cancelled():
            return False
        self.robot.check()
        previous = self.l2
        if not self.next_subtask(source="vlm"):
            return False
        self.session.event("vlm_auto_subtask", previous_l2=previous, next_l2=self.l2,
                           steps=self.steps, evidence=evidence, budget_preserved=True)
        print(f"[VLM AUTO L2] {previous} -> {self.l2} | confirmed {evidence['progress']}", flush=True)
        self.start()  # Same start validation as Enter; old actions were discarded by next_subtask.
        return True

    def try_vlm_takeover(self):
        if self.vlm_takeover is None:
            return False
        before = self.vlm_takeover.status
        evidence = self.vlm_takeover.poll(self)
        if before != self.vlm_takeover.status:
            self.status()
        if evidence is None or self.switch_pending() or self.cancelled():
            return False
        self.robot.check()
        self.session.event("vlm_auto_takeover", evidence=evidence, l2_instruction=self.l2, steps=self.steps)
        print(f"[VLM AUTO TAKEOVER] {self.l2} | {evidence}", flush=True)
        self.keyboard_toggle(source="vlm", evidence=evidence)  # Drain already completed.
        self.vlm_takeover.was_astra = True
        self.vlm_takeover.suspend("Astra owns control")
        return True

    def operator_reset(self):
        """Only the physical keyboard invokes this; never a supervisor decision."""
        self.discard()
        self.session.pending = None
        self.wam_history.clear()
        self.latest_execution = None
        self.state = "RESETTING"
        self.transition("RESET", "operator B; discard all pending actions and supervision")
        self.status()
        try:
            receipt = self.robot.reset_to_task_pose()
            self.pause_id = None
            self.subtask_index = 0
            self.l2 = self.initial_l2
            self.condition_version += 1
            self.demo = None
            self.observation = None
            self.state = "WAITING_FOR_START"
            self.session.event("operator_reset_complete", receipt=receipt,
                               budget_preserved=True, steps=self.steps)
            self.status()
            print("[RESET COMPLETE] Task initial pose verified. Press Enter to continue; budget is preserved.", flush=True)
        except Exception as error:
            self.pause(error)
            print(f"[RESET FAILED] {error}; no automatic restart", flush=True)

    def auto_step(self):
        if self.settings.version not in (2, 3, 4, 5, 6, 7, 8, 10) or self.mode != "WAM" or self.state != "AUTO":
            return
        if self.switch_pending() or self.cancelled():
            return
        if not len(self.queue):
            self.infer(review=False)
        else:
            self.observation = self.robot.observe()
            self.execute(min(len(self.queue), self.settings.monitor_steps), review=False)

    def supervisor_stopped(self, reason, *, pause_kind=None, source='supervisor_error'):
        """Runner-owned terminal fallback; never publishes or clears a hardware fault."""
        if self.state in ('FINISHED', 'EXITED'):
            return False
        if pause_kind == 'hardware_fault':
            self.supervisor_hardware_hold = True
        if (self.settings.version not in (4, 8, 10) or not self.settings.astra_pause_handback
                or self.mode != 'ASTRA' or self.state not in ('WAITING', 'PAUSED_FOR_OPERATOR')
                or self.cancelled() or self.switch_pending() or self.supervisor_hardware_hold):
            self.pause(reason)
            return False
        try:
            self.robot.check()
        except Exception as error:
            self.supervisor_hardware_hold = True
            self.pause(error)
            return False
        self.discard()
        self.session.pending = None
        self.pause_id = None
        self.observation = None
        self.session.event('astra_pause_handback', reason=str(reason), source=source,
                           pause_kind=pause_kind, recovery_confirmed=False, steps=self.steps)
        self.transition('WAM', 'Supervisor stopped; code fallback, recovery unconfirmed: '+str(reason))
        self.state = 'AUTO'
        self.status()
        return True

    def pause(self, error):
        if self.vlm_takeover:
            self.vlm_takeover.suspend("Runner paused")
        if self.vlm_gate:
            self.vlm_gate.suspend("Runner paused; require fresh confirmation")
        # This is a publication pause, NOT an invented emergency-stop action.
        # Already issued short batches may finish; hardware ESTOP remains ROS-owned.
        self.discard()
        self.session.pending = None
        self.last_pause_reason = str(error)
        print(f"[PAUSED] {self.mode}: {error}", flush=True)
        self.state = "PAUSED_FOR_OPERATOR"
        self.pause_id = uuid.uuid4().hex
        self.session.event("pause", pause_id=self.pause_id, reason=str(error), steps=self.steps,
                           hardware=self.robot.status(), pending_motion_may_finish=True)
        self.status()

    def finish(self, source, evidence, outcome="success"):
        text(evidence, "completion evidence")
        self.discard()
        self.session.pending = None
        self.state = "FINISHED"
        self.result = {"outcome": outcome, "completion_source": source, "evidence": evidence,
                       "published_waypoints": self.published, "charged_waypoints": self.steps,
                       "hardware_execution_confirmed": False,
                       "takeovers": self.takeovers, "recovery_attempts": self.attempts,
                       "corrections": self.corrections, "replans": self.replans,
                       "high_level_assisted": True, "low_policy_independent": False,
                       "original_step_limit": self.settings.original_step_limit,
                       "step_limit": self.settings.budget}
        atomic_json(self.session.path / "summary.json", self.result)
        self.session.event("finished", **self.result)
        self.status()

    def start(self):
        try:
            self.robot.check()
            if self.settings.version in (2, 3, 4, 5, 6, 7, 8, 10):
                self.l2 = text(self.subtasks[self.subtask_index], "operator L2")
                self.condition_version = max(1, self.condition_version)
                self.transition("WAM", f"version {self.settings.version} started; model executes without Astra review")
                self.state = "AUTO"
                self.status()
            else:
                self.transition("WAM", "version 1 started; Astra reviews each batch")
                self.review("PREPARE_SUBGOAL")
        except Exception as error:
            self.pause(error)

    def review(self, stage):
        self.robot.check()
        self.observation = self.robot.observe()
        snapshot = self.session.snapshot(self.observation, str(self.steps), include_depth=True)
        self.session.request(stage=stage, control_mode=self.mode, steps=self.steps,
            observation=snapshot, wam_context={} if self.settings.version == 10 else self.wam_context(), instruction=self.l3, l2_instruction=self.l2,
            retry_context=self.retry_context, takeover_trigger=self.takeover_trigger,
            execution_timing={"fps": getattr(self.robot, "fps", 20.),
                              "max_plan_steps": 120 if self.settings.version == 7 else self.settings.takeover_steps,
                              "max_execute_steps": self.settings.takeover_steps if self.mode == "ASTRA" else self.settings.monitor_steps},
            initial_l2_instruction=self.initial_l2,
            conditioning_version=self.condition_version, demo_context=self.demo,
            proposed_actions=self.queue[:self.settings.takeover_steps if self.mode == "ASTRA" else self.settings.monitor_steps].tolist(),
            remaining_actions=len(self.queue), action_source=self.source,
            stagnation_due=self.steps - self.last_progress >= self.settings.stagnation_steps,
            takeovers=self.takeovers, recovery_attempts=self.attempts,
            version=self.settings.version, phase=self.phase,
            astra_pause_handback=self.settings.astra_pause_handback,
            imagination=(self.imagination.pending.get("info") if self.imagination and self.imagination.pending else None),
            **({'execution_feedback': copy.deepcopy(self.latest_execution)} if self.settings.version == 10 else {}))
        self.retry_context = None
        self.wait_started = time.monotonic()
        self.state = "WAITING"
        self.status()

    def tick(self):
        if self.state == "IMAGINING":
            try:
                self.imagination.poll(self)
            except Exception as error:
                self.pause(error)
            return
        if self.state != "WAITING":
            return
        try:
            self.robot.check()
        except Exception as error:
            self.pause(error)
            return
        if time.monotonic() - self.wait_started > self.settings.review_timeout:
            self.supervisor_stopped('Astra review timed out', source='review_timeout')

    def _subgoal(self, decision):
        if "l2_instruction" not in decision:
            if self.l2 is None or decision.get("conditioning_version") != self.condition_version:
                raise ValueError("new l2_instruction or explicit current conditioning_version required")
            return None
        l2 = text(decision["l2_instruction"], "l2_instruction")
        if "\n" in l2 or len(l2.split()) > 30 or "Task:" in l2 or "Subtask:" in l2:
            raise ValueError("L2 must be one short subtask, without Task/Subtask wrappers")
        if self.l2 is None and self.initial_l2 is not None and l2 != self.initial_l2:
            raise ValueError("initial L2 must match the operator-supplied subtask")
        demo = decision.get("demo_context")
        if not isinstance(demo, dict) or not demo.get("viewed_frames"):
            raise ValueError("new subgoal requires demo_context and actually viewed_frames")
        for key in ("video_path", "review_evidence", "scene_differences"):
            text(demo.get(key), "demo_context." + key)
        return l2, demo

    def _apply_subgoal(self, new):
        if new:
            self.l2, self.demo = new
            self.condition_version += 1
            self.session.event("subgoal", version=self.condition_version, instruction=self.prompt,
                               demo_context=self.demo, steps=self.steps)

    def infer(self, review=True):
        self.discard()
        self.state = "INFERRING"
        self.session.pending = None
        self.status()
        from .policy import PolicyTransportError
        for attempt in range(3):
            if self.cancelled():
                raise InterruptedError("operator stopped during WAM inference recovery")
            if self.switch_pending():
                self.state = "AUTO"
                self.status()
                return
            self.robot.check()
            obs = self.robot.observe()
            refs = self.session.snapshot(obs, "inference")
            try:
                actions, metadata = self.policy.infer(obs, self.prompt)
                break
            except PolicyTransportError as error:
                if attempt == 2:
                    self.session.event("wam_reconnect_exhausted", attempts=3, reason=str(error))
                    raise PolicyTransportError(f"WAM connection failed after 3 attempts: {error}") from error
                self.session.event("wam_reconnect", attempt=attempt+1, reason=str(error),
                                   fresh_observation_required=True)
                print(f"[WAM RECONNECT] {error}; retry {attempt+1}/2 with fresh observations", flush=True)
                deadline = time.monotonic() + .5 * (attempt + 1)
                while time.monotonic() < deadline and not self.cancelled() and not self.switch_pending():
                    time.sleep(.05)
        if self.cancelled():
            raise InterruptedError("operator stopped after WAM inference")
        if self.switch_pending():
            self.state = "AUTO"
            self.status()
            return
        if attempt:
            self.session.event("wam_reconnected", attempts=attempt+1)
        self.robot.check()
        self.queue = actions52(actions)
        self.source = "WAM"
        self.wam_active = {
            "trajectory_id": uuid.uuid4().hex, "inferred_at": time.time(),
            "observation": refs, "instruction": self.l3, "l2_instruction": self.l2,
            "conditioning_version": self.condition_version, "phase": self.phase,
            "first_budget_step": self.steps, "actions": self.queue.tolist(),
            "row_status": ["not_published"] * len(self.queue), "publications": [],
            "model_metadata": copy.deepcopy(metadata)}
        self.wam_history.append(self.wam_active)
        self.wam_history = self.wam_history[-3:]
        self.session.event("inference", observation=refs, instruction=self.prompt,
            conditioning_version=self.condition_version, metadata=metadata, actions=self.queue.tolist())
        if review:
            self.review("REVIEW_ACTIONS")
        else:
            self.state = "AUTO"
            self.status()

    def decide(self, d):
        self.tick()
        if self.state != "WAITING":
            raise ValueError("no active review")
        self.session.validate_identity(d)
        if self.settings.version == 10:
            if d.get('operation') == 'v10_action_chunk':
                from .action_chunk import to_decision
                self.session.event('v10_chunk_received', reply=d.get('chunk_reply'), request_id=d['request_id'])
                d = to_decision(d.get('chunk_reply'), self.session.pending)
            elif not (d.get('operation') == 'pause' and d.get('pause_kind') == 'technical_error'):
                raise ValueError('V10 accepts only validated action_chunk envelopes')
        normalized = normalize_decision(d)
        if isinstance(d.get("evidence"), list):
            self.session.event("decision_format_normalized", request_id=d["request_id"],
                               field="evidence", original=d["evidence"], normalized=normalized["evidence"])
        d = normalized
        text(d.get("evidence"), "evidence")
        text(d.get("reason"), "reason")
        op = d.get("operation")
        if self.settings.version in (4, 8, 10) and op == 'hand_back':
            raise ValueError('model-selected hand_back is disabled for this version')
        if self.settings.version == 8:
            if self.mode != "ASTRA":
                raise ValueError("V8 decisions require Astra ownership")
            if op in ("takeover", "correct") or "actions" in d:
                raise ValueError("V8 requires hand selection then candidate approval; no raw actions")
            if op == "propose_candidates":
                if d.get("active_hand") not in ("left", "right") or self.l2 is None:
                    raise ValueError("V8 requires active_hand left/right and current L2")
                if self.imagination is None:
                    raise RuntimeError("V8 requires candidate imagination backend")
                self.session.event("decision", **d)
                self.discard()
                self.attempts += 1
                self.imagination.start(self, d["active_hand"])
                return
            if op == "approve":
                if not self.session.pending or self.session.pending.get("stage") != "REVIEW_CANDIDATES":
                    raise ValueError("V8 approve requires reviewed candidate set")
                count = self.imagination.approve(self, d)
                self.session.event("decision", **d)
                self.execute(count, _imagined=True)
                return
        if self.settings.version in (5, 6, 7) and op == "approve":
            if not self.session.pending or self.session.pending.get("stage") != "REVIEW_IMAGINATION":
                raise ValueError("V5/V6/V7 approve requires reviewed imagination")
            count = self.imagination.approve(self, d)
            self.session.event("decision", **d)
            self.execute(count, _imagined=True)
            return
        if self.settings.version in (2, 3, 4, 5, 6, 7, 8, 10):
            forbidden = ("condition", "replan", "hand_back") if self.settings.version == 2 else ("condition", "replan")
            if self.mode != "ASTRA" or op in forbidden:
                raise ValueError(f"version {self.settings.version} forbids this supervisor ownership/subtask operation")
        if self.settings.version in (3, 4, 5, 6, 7, 8, 10) and op == "hand_back":
            # V3 keeps the human's L2 and bypasses V1's subgoal/demo contract.
            text(d.get("recovery_evidence"), "recovery_evidence")
            if any(key in d for key in ("actions", "l2_instruction", "demo_context")):
                raise ValueError("V3 hand_back must not contain actions or change the subtask")
            self.robot.check()
            self.session.event("decision", **d)
            self.discard()
            self.session.pending = None
            self.pause_id = None
            self.observation = None
            self.transition("WAM", "Astra automatic hand-back; fresh inference required: " + d["reason"])
            self.state = "AUTO"
            self.status()
            return
        if op not in ("condition", "approve", "replan", "correct", "takeover", "hand_back", "pause", "complete", "observe"):
            raise ValueError("unsupported operation; no reset or abort via supervisor")
        if op == "complete":
            self.session.event("decision", **d)
            self.finish("astra_visual", d["evidence"])
            return
        if op == "pause":
            self.session.event("decision", **d)
            kind = d.get('pause_kind')
            if self.settings.version in (4, 8, 10):
                self.supervisor_stopped(d['reason'], pause_kind=kind, source='model_pause')
                return
            if (self.settings.astra_pause_handback and self.settings.version in (3, 4, 5, 6, 7, 8, 10)
                    and self.mode == 'ASTRA' and kind in ('planning', 'collision_risk', 'observation_uncertainty')):
                self.robot.check()  # Do not rearm/clear hardware incidents.
                self.discard()
                self.session.pending = None
                self.pause_id = None
                self.observation = None
                self.session.event('astra_pause_handback', reason=d['reason'], evidence=d['evidence'],
                                   pause_kind=kind, recovery_confirmed=False, steps=self.steps)
                self.transition('WAM', 'Operator-configured Astra pause hand-back; recovery unconfirmed: '+d['reason'])
                self.state = 'AUTO'
                self.status()
                return
            self.pause(d["reason"])
            return
        if op == "condition" and (self.l2 is not None or self.mode != "WAM"):
            raise ValueError("condition is only for initial subgoal; use replan/hand_back")
        if op not in ("condition", "observe") and self.l2 is None:
            raise ValueError("prepare initial subgoal before motion")
        new = None
        if op in ("condition", "replan", "hand_back"):
            if op == "replan" and self.mode == "ASTRA":
                raise ValueError("explicit hand_back required during takeover")
            if op == "hand_back":
                if self.mode != "ASTRA":
                    raise ValueError("not in takeover")
                text(d.get("recovery_evidence"), "recovery_evidence")
            new = self._subgoal(d)
        actions = None
        if op in ("correct", "takeover"):
            if op == "correct" and self.mode == "ASTRA":
                raise ValueError("use takeover to continue recovery")
            actions = actions52(d.get("actions"))
            limit = 120 if self.settings.version == 7 and op == "takeover" else (self.settings.takeover_steps if op == "takeover" else self.settings.monitor_steps)
            if len(actions) > limit:
                raise ValueError(f"submission exceeds {limit} rows")
            if len(actions) < limit:
                text(d.get("short_batch_reason"), "short_batch_reason")
            if op == "takeover":
                if type(d.get("new_attempt")) is not bool:
                    raise ValueError("new_attempt must be an explicit boolean")
                if self.mode != "ASTRA" and not d["new_attempt"]:
                    raise ValueError("first takeover starts a new attempt")
                if d["new_attempt"]:
                    text(d.get("change_reason"), "change_reason")
        count = None
        if op in ("approve", "correct", "takeover"):
            available = len(self.queue) if actions is None else len(actions)
            if not available and self.mode == "ASTRA":
                raise ValueError("submit new recovery actions or hand_back")
            execute_limit = self.settings.takeover_steps if op == "takeover" or self.mode == "ASTRA" else self.settings.monitor_steps
            if self.settings.version == 7 and op == "takeover":
                from .prefix_plan import validate_stages
                validate_stages(d.get("plan_stages"), available)
                execute_limit = available
                if d.get("execute_steps", available) != available:
                    raise ValueError("V7 takeover must imagine the whole submitted plan")
            count = d.get("execute_steps", min(available, execute_limit))
            if type(count) is not int or not 0 <= count <= min(available, execute_limit):
                raise ValueError("invalid execute_steps")
            if available and count == 0:
                raise ValueError("execute_steps must be positive")
            if count < min(available, execute_limit):
                text(d.get("short_batch_reason"), "short_batch_reason")
            if d.get("progress_evidence"):
                text(d["progress_evidence"], "progress_evidence")
            if op == "approve" and self.steps - self.last_progress >= self.settings.stagnation_steps and not d.get("progress_evidence"):
                raise ValueError("stagnation requires diagnosis/replan or observed progress evidence")
        self.session.event("decision", **d)
        try:
            if op == "observe":
                if self.mode == "ASTRA":
                    self.discard()
                    self.session.event("astra_reobserve", reason=d["reason"], evidence=d["evidence"])
                    print("[ASTRA] Re-observing and planning; ASTRA retains control, no actions published", flush=True)
                self.review("OBSERVE")
            elif op in ("condition", "replan", "hand_back"):
                self.discard()
                self._apply_subgoal(new)
                if op != "condition":
                    self.replans += 1
                if self.mode != "WAM":
                    self.transition("WAM", "Astra hand-back")
                # This deployed server is request-based: no reset RPC exists.
                self.infer()
            else:
                if actions is not None:
                    self.session.event("intervention", original_actions=self.queue.tolist(),
                                       final_actions=actions.tolist(), decision=d)
                    self.discard()
                    self.queue, self.source = actions, op.upper()
                    if self.settings.version == 7:
                        self.plan_stages = copy.deepcopy(d.get("plan_stages"))
                    if op == "takeover":
                        self.takeovers += int(self.mode != "ASTRA")
                        self.attempts += int(d["new_attempt"])
                        if self.mode != "ASTRA":
                            self.transition("ASTRA", "Astra takeover decision")
                    else:
                        self.corrections += 1
                if d.get("progress_evidence"):
                    self.last_progress = self.steps
                if len(self.queue):
                    if self.mode == "ASTRA":
                        print(f"[ASTRA EXECUTING] validated decision; up to {count} waypoints", flush=True)
                    self.execute(count)
                else:
                    self.infer()  # New chunk still needs a separate approval.
        except Exception as error:
            self.pause(error)

    def execute(self, count, review=True, *, _imagined=False):
        if self.switch_pending():
            return
        if self.cancelled():
            raise InterruptedError("operator stopped runner")
        if self.settings.budget is not None and not (self.settings.version == 7 and self.mode == "ASTRA" and not _imagined):
            count = min(count, self.settings.budget - self.steps)
        if count <= 0:
            self.finish("budget", "configured published-waypoint budget exhausted", "budget_exhausted")
            return
        self.robot.check()
        if self.settings.version == 8 and self.mode == "ASTRA":
            if not _imagined or self.approved_imagination is None:
                raise ValueError("V8 execution requires an exact approved candidate")
            from .dynamics_candidates import action_hash
            if action_hash(self.queue[:count]) != self.approved_imagination['candidate_sha256']:
                raise ValueError("V8 approved candidate changed before execution")
        batch = self.queue[:count].copy()
        self.robot.validate(batch, self.observation)
        if self.settings.version in (5, 6, 7) and self.mode == "ASTRA" and not _imagined:
            if self.imagination is None:
                raise RuntimeError("V5/V6/V7 requires a configured imagination backend")
            self.imagination.start(self, batch)
            return
        if self.switch_pending():
            return
        self.state = "EXECUTING"
        execution_record = None
        if self.settings.version == 10 and self.mode == 'ASTRA' and self.source == 'TAKEOVER':
            execution_record = {'chunk_id': uuid.uuid4().hex,
                'source_request_id': (self.session.pending or {}).get('request_id'),
                'l2_instruction': self.l2, 'conditioning_version': self.condition_version,
                'phase': self.phase, 'fps': getattr(self.robot, 'fps', 20.),
                'actions': batch.tolist(), 'publication_status': 'publication_attempted',
                'capture_status': 'unavailable', 'samples': [], 'physical_execution_confirmed': False}
            self.latest_execution = execution_record
        self.session.pending = None
        self.status()
        first = self.steps
        self.session.event("batch_intent", first_step=first, source=self.source, actions=batch.tolist())
        # Charge before publishing: partial arm/hand failures must not free the
        # budget to replay potentially issued targets. This is conservative.
        self.steps += count
        publication = None
        if self.source == "WAM" and self.wam_active is not None:
            publication = {"row_start": self.wam_offset, "row_stop": self.wam_offset + count,
                           "attempted_at": time.time(), "observed_state": {k: np.asarray(self.observation[k]).tolist() for k in
                               ("timestamp", "left_ee_pose", "right_ee_pose", "left_hand_joint", "right_hand_joint")},
                           "status": "publication_attempted", "physical_execution_confirmed": False}
            self.wam_active["publications"].append(publication)
            self.wam_active["row_status"][self.wam_offset:self.wam_offset+count] = ["publication_attempted"] * count
        imagined = self.approved_imagination if _imagined else None
        self.approved_imagination = None
        receipt = self.robot.publish(batch)
        capture = None
        if execution_record is not None:
            from .execution_feedback import ExecutionCapture
            execution_record['publication_status'] = 'published'
            capture = ExecutionCapture(self.robot, receipt, batch, execution_record, self.cancelled).start()
        if imagined is not None:
            self.session.event("imagination_execution", candidate_id=imagined["candidate_id"],
                               candidate_sha256=imagined["candidate_sha256"], receipt=receipt,
                               anchor_timestamp=imagined["anchor_timestamp"],
                               model_target_timestamp=imagined["anchor_timestamp"]+imagined["target_horizon_seconds"],
                               hypothetical_target_not_wall_clock_execution=True)
        if publication is not None:
            publication.update(status="published", published_at=time.time(), receipt=copy.deepcopy(receipt))
            self.wam_active["row_status"][self.wam_offset:self.wam_offset+count] = ["published"] * count
            self.wam_offset += count
        self.published += count
        self.session.event("batch_published", first_step=first, source=self.source,
                           actions=batch.tolist(), receipt=receipt,
                           conditioning_version=self.condition_version, attempt=self.attempts)
        self.queue = self.queue[count:]
        if self.settings.version == 7 and _imagined:
            self.session.event("v7_tail_discarded", rows=len(self.queue), reason="fresh observation and new plan required")
            self.queue = np.empty((0, 52))
        drained = False
        try:
            self.robot.drain(self.cancelled)
            drained = True
        finally:
            if capture is not None:
                try:
                    self.latest_execution = capture.finish(self.session, drained=drained)
                    self.session.event('v10_execution_feedback', **self.latest_execution)
                except Exception as error:
                    self.latest_execution = {**execution_record, 'capture_status': 'unavailable',
                                             'capture_error': str(error), 'samples': []}
                    self.session.event('v10_execution_capture_error', chunk_id=execution_record['chunk_id'], reason=str(error))
        if self.cancelled():
            raise InterruptedError("operator stopped after issued batch; no further publication")
        if self.settings.budget is not None and self.steps >= self.settings.budget:
            self.finish("budget", "configured published-waypoint budget exhausted", "budget_exhausted")
        elif review:
            self.review("VERIFY_RECOVERY" if self.mode == "ASTRA" else "VERIFY")
        else:
            self.state = "AUTO"
            self.status()

    def operator(self, d):
        if d.get("session_id") != self.session.id or d.get("pause_id") != self.pause_id:
            raise ValueError("stale operator command")
        text(d.get("reason"), "reason")
        op = d.get("operation")
        if op not in ("resume", "complete"):
            raise ValueError("operator operation must be resume or complete")
        if op == "complete":
            self.finish("human", d["reason"])
            return
        if self.state != "PAUSED_FOR_OPERATOR":
            raise ValueError("runner is not paused")
        try:
            self.robot.rearm()  # Does NOT reset ROS or clear hardware ESTOP.
            self.discard()
            self.pause_id = None
            self.session.event("operator_resume", reason=d["reason"])
            if self.settings.version in (2, 3, 4, 5, 6, 7, 8, 10) and self.mode == "WAM":
                self.state = "AUTO"
                self.status()
            else:
                self.review("PREPARE_SUBGOAL" if self.l2 is None else "REPLAN_REQUIRED")
        except Exception as error:
            self.pause(error)
