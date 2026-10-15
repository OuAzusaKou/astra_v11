"""Start an explicit mode-2 session, or submit a request-bound decision."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import threading
import time

from .control import Controller, Settings
from .session import Session, atomic_json, submit
from .remote_supervisor import RemoteSupervisor, RemoteDisconnected


ROOT = Path(__file__).resolve().parents[1]


def serve(controller, stop, keyboard=None, supervisor=None):
    controller.cancelled = stop.is_set
    if keyboard is not None:
        retry_pending = getattr(keyboard, "retry_pending", threading.Event())
        controller.switch_pending = lambda: keyboard.pending.is_set() or keyboard.reset_pending.is_set() or keyboard.subtask_pending.is_set()
        controller.state = "WAITING_FOR_START"
        controller.status()
        controller.session.event("waiting_for_start", key="Enter")
        keyboard.arm_start()
    elif not stop.is_set():
        controller.start()
    while not stop.is_set() and controller.state != "FINISHED":
        try:
            if keyboard is not None:
                if keyboard.closed.is_set():
                    stop.set()
                    raise RuntimeError("keyboard terminal disconnected")
                if keyboard.reset_pending.is_set():
                    retry_pending.clear()
                    keyboard.reset_pending.clear()
                    keyboard.pending.clear()
                    if supervisor:
                        supervisor.invalidate()
                    controller.operator_reset()
                    if controller.state == "WAITING_FOR_START":
                        keyboard.arm_start()
                    continue
                if keyboard.subtask_pending.is_set():
                    keyboard.subtask_pending.clear()
                    if controller.next_subtask():
                        if supervisor:
                            supervisor.invalidate()
                        keyboard.arm_start()
                        continue
                if controller.state == "WAITING_FOR_START":
                    if keyboard.start_requested.is_set():
                        controller.session.event("operator_start", key="Enter")
                        controller.start()
                    else:
                        stop.wait(.05)
                        continue
                if keyboard.pending.is_set():
                    retry_pending.clear()
                    keyboard.pending.clear()
                    if supervisor:
                        supervisor.invalidate()
                    try:
                        controller.keyboard_toggle()
                    except Exception as error:
                        controller.pause(error)
                if retry_pending.is_set():
                    retry_pending.clear()
                    if controller.state == "PAUSED_FOR_OPERATOR" and controller.mode == "ASTRA" and controller.settings.version in (2, 3, 4, 5, 6, 7, 8, 10):
                        if supervisor:
                            supervisor.invalidate()
                    controller.retry_astra()
            controller.tick()
            command = controller.session.consume("operator.json")
            if command is not None:
                controller.operator(command)
            if supervisor:
                if isinstance(supervisor, RemoteSupervisor):
                    supervisor.recover(controller)
                request = controller.session.pending if controller.state == "WAITING" else None
                try:
                    decision = supervisor.poll(request)
                except RemoteDisconnected:
                    raise  # Keep bounded transport recovery before terminal fallback.
                except Exception as error:
                    supervisor.invalidate()
                    controller.supervisor_stopped(error)
                    decision = None
                if decision is not None and not controller.switch_pending():
                    try:
                        controller.decide(decision)
                    except Exception as error:
                        supervisor.invalidate()
                        controller.supervisor_stopped("invalid Astra response: " + str(error))
            elif controller.state == "WAITING":
                decision = controller.session.consume("response.json")
                if decision is not None:
                    controller.decide(decision)
            changed = controller.try_vlm_subtask()
            if not changed:
                changed = controller.try_vlm_takeover()
            if changed and supervisor:
                supervisor.invalidate()
            if controller.state == "AUTO":
                try:
                    controller.auto_step()
                except Exception as error:
                    controller.pause(error)
        except RemoteDisconnected as error:
            controller.pause(error)
            supervisor.wait_for_reconnect(controller)
        except (ValueError, KeyError, TypeError) as error:
            controller.session.event("rejected_input", error=str(error))
            if supervisor:
                controller.pause(error)
        except Exception as error:
            controller.pause(error)
        stop.wait(.1)
    if supervisor:
        supervisor.invalidate()
    if stop.is_set() and controller.state != "FINISHED":
        controller.discard()
        controller.session.pending = None
        controller.state = "EXITED"
        controller.session.event("runner_exited", reason="operator signal", steps=controller.steps,
                                 physical_execution_confirmed=False)
        controller.status()


def positive(value):
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_mode2_prompt(value):
    match = re.fullmatch(r"Task:\s*(.+?)\s+Subtask:\s*(.+)", value.strip())
    if match is None:
        raise ValueError("--prompt must be 'Task: <L3> Subtask: <initial L2>' supplied by the operator")
    l3, l2 = (part.strip() for part in match.groups())
    if not l3 or not l2 or len(l2.split()) > 30 or any(
        marker in part for part in (l3, l2) for marker in ("Task:", "Subtask:", "\n")
    ):
        raise ValueError("provide one non-empty L3 and one short initial L2")
    return l3, l2


def load_control_config(path):
    config = json.loads(Path(path).expanduser().read_text())
    if not isinstance(config, dict) or set(config) != {"version", "host", "port", "expected_checkpoint", "toggle_key"}:
        raise ValueError("control config requires version, host, port, expected_checkpoint and toggle_key")
    if type(config["version"]) is not int or config["version"] not in (1, 2, 3, 4, 5, 6, 7, 8, 10):
        raise ValueError("control version must be 1, 2, 3, 4, 5, 6, 7 or 8")
    if not isinstance(config["host"], str) or not config["host"].strip():
        raise ValueError("control host is empty")
    if type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
        raise ValueError("invalid control port")
    if not isinstance(config["expected_checkpoint"], str) or not config["expected_checkpoint"].strip():
        raise ValueError("expected checkpoint must be explicit")
    if not isinstance(config["toggle_key"], str) or not re.fullmatch("[a-zA-Z]", config["toggle_key"]) or config["toggle_key"].lower() in ("b", "p", "r"):
        raise ValueError("toggle_key must be one ASCII letter")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", type=int, choices=[2], default=2)
    run.add_argument("--config", default=str(ROOT / "wam_direct/client_wam_direct.yaml"))
    run.add_argument("--control-config", default=str(ROOT / "config/astra/version1_review.json"))
    run.add_argument("--host", default=None)
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--prompt", "--instruction", required=True, help="Task: <L3> Subtask: <operator-supplied initial L2>")
    run.add_argument("--next-subtask", action="append", default=[], help="ordered next L2; repeatable, P then Enter selects it (V2/V3/V4)")
    run.add_argument("--vlm-auto-takeover", action="store_true", help="V4: 12s no new high; replay diagnostic only; 10s warmup, 15s hand-back cooldown")
    run.add_argument('--takeover-mode', choices=('manual','vlm'), help='V4: manual T takeover or VLM stagnation; overrides --vlm-auto-takeover')
    run.add_argument('--l2-switch-mode', choices=('manual','vlm'), help='V4: P/Enter or VLM L2 switching; default vlm')
    run.add_argument('--astra-pause-handback', action='store_true', help='explicit operator policy: model planning/collision-risk pauses hand back to WAM; hardware faults remain paused')
    run.add_argument("--vlm-progress-log", type=Path, help="V4 only: this session's observer predictions.jsonl")
    run.add_argument("--vlm-progress-threshold", type=int, choices=(90, 100), default=100)
    run.add_argument("--vlm-progress-count", type=int, choices=range(2, 11), default=2)
    run.add_argument("--session", required=True, help="new session directory (never overwrite)")
    run.add_argument("--owner", required=True, help="identity of the supervising Astra session")
    run.add_argument("--original-step-limit", type=int, required=True)
    run.add_argument("--no-step-limit", action="store_true", help="V4: disable total waypoint budget; retain per-batch limits")
    run.add_argument("--chunk_size", type=int, default=32)
    run.add_argument("--fps", type=positive, default=20.)
    run.add_argument("--review-timeout", type=positive, default=900.)
    run.add_argument("--request-timeout", type=positive, default=60.)
    run.add_argument("--supervisor-backend", choices=("local", "mac"), default="local")
    from .supervisor_models import CHOICES
    run.add_argument('--astra-model',choices=CHOICES,default='astra',help='fixed startup model; QuickRouter supports V4/V8')
    run.add_argument("--supervisor-proxy", default=os.environ.get("ASTRA_HTTPS_PROXY", "http://127.0.0.1:7897"))
    run.add_argument("--supervisor-timeout", type=positive, default=600.)
    run.add_argument("--pointcloud-mode", choices=("preview", "review"), default="review")
    run.add_argument("--pointcloud-overlay", default="/home/liberai/camera_pose/rviz_overlay")
    run.add_argument("--candidate-distance-m", type=positive, default=.10)
    run.add_argument("--dynamics-host", default="10.128.39.115")
    run.add_argument("--dynamics-port", type=int, default=29642)
    run.add_argument("--dynamics-mode", choices=("preview", "review"), default="preview")
    run.add_argument("--dynamics-timeout", type=positive, default=30.)
    run.add_argument("--reference-video", help="demonstration video on the runner machine; visual context for Astra only")
    run.add_argument("--reference-video-frames", type=int, default=12, help="uniformly sampled reference frames (1-32)")
    run.add_argument("--depth-camera", default=None, metavar="SDK_SERIAL_OR_AUTO",
                     help="optional D435i depth judgment for Astra only; auto requires one camera, no XYZ/calibration")
    run.add_argument("--sensor-timeout", type=positive, default=30.)
    run.add_argument("--max-step-m", type=positive, default=.10, help="per-waypoint translation jump guard, not collision checking")
    run.add_argument("--max-step-rad", type=positive, default=.75, help="per-waypoint rotation jump guard")
    run.add_argument("--client-hand-limits", action=argparse.BooleanOptionalAction, default=False,
                     help="enable client URDF hand-target rejection; default off, ROS driver limits unchanged")
    run.add_argument("--review-state-guard", action=argparse.BooleanOptionalAction, default=False,
                     help="reject state drift during review; default log only, fresh-state trajectory guards remain")
    response = sub.add_parser("respond")
    response.add_argument("--session", required=True)
    response.add_argument("--file", required=True)
    status = sub.add_parser("status")
    status.add_argument("--session", required=True)
    for name in ("resume", "complete"):
        operator = sub.add_parser(name)
        operator.add_argument("--session", required=True)
        operator.add_argument("--session-id", required=True)
        operator.add_argument("--pause-id")
        operator.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        print((Path(args.session).expanduser() / "status.json").read_text())
        return 0
    if args.command == "respond":
        submit(args.session, json.loads(Path(args.file).read_text()))
        return 0
    if args.command in ("resume", "complete"):
        submit(args.session, {"operation": args.command, "session_id": args.session_id,
                             "pause_id": args.pause_id, "reason": args.reason}, operator=True)
        return 0
    try:
        control_config = load_control_config(args.control_config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    from .switch_modes import resolve_modes
    try:
        args.takeover_mode, args.l2_switch_mode = resolve_modes(control_config['version'],
            args.takeover_mode, args.l2_switch_mode, args.vlm_auto_takeover)
    except ValueError as error:
        parser.error(str(error))
    args.vlm_auto_takeover = args.takeover_mode == 'vlm'
    args.host = args.host if args.host is not None else control_config["host"]
    args.port = args.port if args.port is not None else control_config["port"]
    if not args.host.strip():
        parser.error("host must not be empty")
    if not args.owner.strip() or not args.prompt.strip() or args.chunk_size < 1 or not 1 <= args.port <= 65535:
        parser.error("invalid owner, prompt, chunk_size or port")
    try:
        l3, initial_l2 = parse_mode2_prompt(args.prompt)
        for subtask in args.next_subtask:
            parse_mode2_prompt(f"Task: {l3} Subtask: {subtask}")
        if args.next_subtask and control_config["version"] not in (2, 3, 4, 5, 6, 7, 8, 10):
            raise ValueError("--next-subtask currently requires V2, V3 or V4")
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.reference_video_frames <= 32:
        parser.error("--reference-video-frames must be between 1 and 32")
    if args.vlm_auto_takeover and control_config["version"] not in (4, 5, 6, 7, 8, 10):
        parser.error("--vlm-auto-takeover requires V4")
    if (control_config["version"] in (4, 5, 6, 7, 8, 10)) != (args.vlm_progress_log is not None):
        parser.error("V4/V5/V6/V7/V8 require --vlm-progress-log; other versions must not use it")
    if args.vlm_progress_log:
        args.vlm_progress_log = str(args.vlm_progress_log.expanduser().resolve())
    if control_config["version"] == 8 and not 0 < args.candidate_distance_m <= .10:
        parser.error("V8 --candidate-distance-m must be >0 and <=0.10 m")
    if args.astra_model!='astra' and control_config['version'] not in (4,8,10):
        parser.error('QuickRouter takeover models currently require V4 or V8')
    settings = Settings(args.original_step_limit, review_timeout=args.review_timeout, version=control_config["version"],
                        astra_pause_handback=args.astra_pause_handback, no_step_limit=args.no_step_limit)
    # Shared by both launch scripts in this copy. Never kills/replaces an owner.
    lock_dir = Path(f"/tmp/leo-astra-{os.getuid()}")
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    info = lock_dir.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("robot lock directory must be private")
    with (lock_dir / "robot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from utils.client_config import load_client_config
        config = load_client_config(args.config)
        session = Session(args.session, args.owner, {
            "instruction": l3, "initial_l2_instruction": initial_l2, "control_config": control_config, "settings": vars(args), "client_config": config,
            "source_snapshot_sha256": hashlib.sha256((ROOT / "SOURCE_SNAPSHOT.json").read_bytes()).hexdigest(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in ("astra", "utils", "envs") for p in (ROOT / folder).rglob("*.py")},
            "auto_reset": False, "server_history": "request_based_no_reset_rpc",
        })
        session.status(state="STARTING", pause_id=None)
        stop = threading.Event()
        handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        cleanup = None
        connection = None
        keyboard = None
        supervisor = None
        depth_camera = None
        try:
            if args.reference_video:
                from .video_context import prepare_reference_video
                session.reference_video = prepare_reference_video(
                    args.reference_video, session.path / "reference_video", args.reference_video_frames)
                atomic_json(session.path / "reference_video.json", session.reference_video)
                session.event("reference_video_prepared", **session.reference_video)
                print(f"[REFERENCE VIDEO] Prepared {len(session.reference_video['images'])} demonstration frames", flush=True)
            from .keyboard import ToggleKeyboard
            keyboard = ToggleKeyboard(control_config["toggle_key"], enable_toggle=settings.version in (2, 3, 4, 5, 6, 7, 8, 10))
            keyboard.start()
            if args.supervisor_backend == "mac":
                from .remote_supervisor import RemoteSupervisor
                supervisor = RemoteSupervisor(session, timeout=args.supervisor_timeout)
            else:
                from .supervisor_models import build_supervisor
                supervisor = build_supervisor(session, model=args.astra_model, version=settings.version,
                    timeout=args.supervisor_timeout, proxy=args.supervisor_proxy)
            supervisor.bootstrap(stop)
            from .robot import create_robot
            from .policy import Connection, Mode2Policy
            from utils.liber_teleop_v22_client import prepare_hand_action_decoder
            if settings.version in (5, 8):
                import uuid
                from .dynamics import rpc, PROTOCOL, ADAPTER_SHA, NORM_SHA
                health = rpc(args.dynamics_host,args.dynamics_port,
                             {"cmd":"health","request_id":uuid.uuid4().hex},args.dynamics_timeout)
                if (health.get("protocol"),health.get("adapter_sha256"),health.get("norm_sha256")) != (PROTOCOL,ADAPTER_SHA,NORM_SHA):
                    raise ValueError("Dynamics adapter/model health fingerprint mismatch")
            prepare_hand_action_decoder("mano")
            robot, cleanup = create_robot(args, config, stop)
            if args.depth_camera:
                from .depth_camera import DepthCamera
                depth_camera = DepthCamera(args.depth_camera)
                session.depth_source = depth_camera
                session.event("depth_camera_ready", serial=depth_camera.serial, purpose="depth_judgment_only")
                print(f"[DEPTH READY] D435i {depth_camera.serial} | relative optical depth only; no XYZ", flush=True)
            connection = Connection(args.host, args.port, stop, args.request_timeout)
            policy = Mode2Policy(connection, session.id, chunk_size=args.chunk_size,
                                 dataset_name=config["policy"].get("dataset_name"),
                                 expected_checkpoint=control_config["expected_checkpoint"])
            print(f"Astra session: {session.path}\nOwner: {session.owner}\nB: reset to task initial pose; Enter required after reset.", flush=True)
            controller = Controller(robot, policy, session, settings, l3, initial_l2=initial_l2, next_subtasks=args.next_subtask)
            if settings.version in (5, 8):
                robot.enforce_review_state = True
                robot.review_hand_tolerance_rad = .15
                print(f"[V{settings.version}] Review hand drift tolerance: 0.15 rad; arm checks unchanged", flush=True)
                from .dynamics import Imagination
                if settings.version == 8:
                    from .dynamics_candidates import CandidateImagination
                    controller.imagination = CandidateImagination(args.dynamics_host, args.dynamics_port,
                        args.dynamics_mode, args.dynamics_timeout, args.candidate_distance_m)
                else:
                    controller.imagination = Imagination(args.dynamics_host, args.dynamics_port,
                                                         args.dynamics_mode, args.dynamics_timeout)
                session.event("imagination_enabled", mode=args.dynamics_mode,
                              host=args.dynamics_host, port=args.dynamics_port)
            if settings.version in (6, 7):
                robot.enforce_review_state = True
                robot.review_hand_tolerance_rad = .15
                from .pointcloud import PointcloudImagination
                controller.imagination = PointcloudImagination(args.pointcloud_mode, overlay=args.pointcloud_overlay, long_plan=settings.version == 7)
                from .pointcloud_capture import PointcloudCapture
                controller.imagination.capture = PointcloudCapture(robot)
                session.event("imagination_enabled", mode=args.pointcloud_mode, backend="urdf_pointcloud")
            if settings.version in (4, 5, 6, 7, 8, 10):
                from .switch_modes import configure_vlm_switching
                configure_vlm_switching(controller, args)
            print(f"Control version: {settings.version} | Server: {args.host}:{args.port}", flush=True)
            serve(controller, stop, keyboard, supervisor)
            return 0 if controller.result else 130
        except Exception as error:
            session.event("startup_or_runner_error", error=str(error))
            session.status(state="EXITED", error=str(error), pause_id=None)
            raise
        finally:
            if depth_camera:
                try:
                    depth_camera.close()
                except Exception as error:
                    session.event("depth_camera_close_error", error=str(error))
            if supervisor:
                supervisor.close()
            if keyboard:
                keyboard.close()
            if connection:
                connection.close()
            if cleanup:
                cleanup()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
