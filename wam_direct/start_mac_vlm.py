#!/usr/bin/env python3
"""Generate a shared run ID, print the Leo3 command, then wait in the Mac bridge."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/liberai/Liberai_Deploy_astra"


def leo_command(run_id, prompt, subtasks, version=4, threshold=100, count=2, auto_takeover=False, dynamics_mode="preview", pointcloud_mode="review", pause_handback=False, candidate_distance_m=.10, astra_model='astra', takeover_mode=None, l2_switch_mode=None, no_step_limit=False, reference_video=None, reference_video_frames=12):
    from astra.switch_modes import resolve_modes
    takeover_mode, l2_switch_mode = resolve_modes(version, takeover_mode, l2_switch_mode, auto_takeover)
    if no_step_limit and version not in (4, 10):
        raise ValueError("--no-step-limit requires V4 or V10")
    if type(reference_video_frames) is not int or not 1 <= reference_video_frames <= 32:
        raise ValueError("--reference-video-frames must be between 1 and 32")
    if reference_video is not None and (not reference_video.strip() or not reference_video.startswith('/')):
        raise ValueError("--reference-video must be an absolute path on Leo3 (not a Windows path)")
    auto_takeover = takeover_mode == 'vlm'
    if astra_model!='astra' and version not in (4,8,10):
        raise ValueError('QuickRouter takeover requires V4 or V8')
    args = ["python3", "wam_direct/launch_tmux.py", "--name", run_id, "--vlm-progress",
            "--client", "astra", "--", "run", "--supervisor-backend", "mac",
            "--control-config", ({3: "config/astra/version3_auto_handback.json", 4: "config/astra/version4_vlm_subtasks.json", 5: "config/astra/version5_dynamics.json", 6: "config/astra/version6_pointcloud.json", 7: "config/astra/version7_prefix.json", 8: "config/astra/version8_dynamics_candidates.json", 10: "config/astra/version10_action_chunk.json"}[version]),
            "--host", "10.128.40.203", "--port", "29572", "--fps", "20", "--prompt", prompt]
    if no_step_limit:
        args += ["--no-step-limit"]
    if reference_video is not None:
        args += ["--reference-video", reference_video, "--reference-video-frames", str(reference_video_frames)]
    if astra_model!='astra':args += ['--astra-model',astra_model]
    if version in (4, 10):
        args += ['--takeover-mode',takeover_mode,'--l2-switch-mode',l2_switch_mode]
    if version in (5, 8):
        args.insert(args.index("--client"), "--dynamics-view")
    if version in (4, 5, 6, 7, 8, 10):
        args += ["--vlm-progress-log", f"{REMOTE}/outputs/progress_vlm/{run_id}_follow/predictions.jsonl",
                 "--vlm-progress-threshold", str(threshold), "--vlm-progress-count", str(count)]
    if version in (5, 8):
        args += ["--dynamics-mode", dynamics_mode]
    if version == 8:
        args += ["--candidate-distance-m", str(candidate_distance_m)]
    if version in (6, 7):
        args.insert(args.index("--client"), "--rviz-record")
        args += ["--pointcloud-mode", pointcloud_mode]
    if auto_takeover:
        args += ["--vlm-auto-takeover"]
    if pause_handback:
        args += ['--astra-pause-handback']
    for task in subtasks:
        args += ["--next-subtask", task]
    args += ["--session", f"{REMOTE}/outputs/astra/{run_id}", "--owner", run_id,
             "--original-step-limit", "400"]
    return ("export ROS_DOMAIN_ID=103\nexport ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET\n"
            + 'export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}\n'
            + 'export LD_LIBRARY_PATH=/opt/openrobots/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n'
            + "RUN_ID=" + shlex.quote(run_id) + "\ncd " + shlex.quote(REMOTE) + "\n" + shlex.join(args) + "\n")


def main():
    sys.path.insert(0,str(ROOT))
    from astra.supervisor_models import CHOICES
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--no-step-limit", action="store_true", help="V4: disable total waypoint budget")
    parser.add_argument("--reference-video", help="demonstration video absolute path on Leo3; passed to every Astra decision as sampled frames")
    parser.add_argument("--reference-video-frames", type=int, choices=range(1, 33), default=12, metavar="1-32")
    parser.add_argument("--host", default="leo3", help="SSH host for the supervisor bridge")
    parser.add_argument("--proxy", default="http://127.0.0.1:7897", help="supervisor proxy; empty disables the override")
    parser.add_argument("--codex-bin", help="optional Codex executable path")
    parser.add_argument('--astra-model',choices=CHOICES,default='astra',help='fixed startup model, never switches automatically')
    parser.add_argument("--pointcloud-mode", choices=("preview", "review"), default="review")
    parser.add_argument("--dynamics-mode", choices=("preview", "review"), default="preview")
    parser.add_argument("--candidate-distance-m", type=float, default=.10)
    parser.add_argument("--vlm-auto-takeover", action="store_true")
    parser.add_argument('--takeover-mode',choices=('manual','vlm'),help='V4: T only or VLM takeover; overrides legacy flag')
    parser.add_argument('--l2-switch-mode',choices=('manual','vlm'),help='V4: P/Enter or VLM switching')
    parser.add_argument('--astra-pause-handback',action='store_true')
    parser.add_argument('--ssh-timeout',type=float,default=60.)
    parser.add_argument('--ssh-connect-timeout',type=int,default=15)
    parser.add_argument('--ssh-retries',type=int,default=3)
    parser.add_argument("--control-version", type=int, choices=(3, 4, 5, 6, 7, 8, 10), default=4)
    parser.add_argument("--vlm-progress-threshold", type=int, choices=(90, 100), default=100)
    parser.add_argument("--vlm-progress-count", type=int, choices=range(2, 11), default=2)
    parser.add_argument("--prompt", default="Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper")
    subtasks = parser.add_mutually_exclusive_group()
    subtasks.add_argument("--next-subtask", action="append", default=None)
    subtasks.add_argument("--single-subtask", action="store_true", help="do not append a default next subtask")
    parser.add_argument("--dry-run", action="store_true", help="print commands only; start nothing")
    args = parser.parse_args()
    if args.reference_video is not None and (not args.reference_video.strip() or not args.reference_video.startswith('/')):
        parser.error("--reference-video must be an absolute path on Leo3 (not a Windows path)")
    from astra.switch_modes import resolve_modes
    try:
        selected_modes = resolve_modes(args.control_version, args.takeover_mode, args.l2_switch_mode, args.vlm_auto_takeover)
    except ValueError as error:
        parser.error(str(error))
    if args.astra_model!='astra' and args.control_version not in (4,8,10):
        parser.error('QuickRouter takeover models require V4 or V8')
    if not 0 < args.candidate_distance_m <= .10:
        parser.error("candidate distance must be >0 and <=0.10 m")
    if args.vlm_auto_takeover and args.control_version not in (4, 5, 6, 7, 8, 10):
        parser.error("auto takeover requires V4")
    run_id = args.run_id or (f"pepper_v{args.control_version}_vlm_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        parser.error("run-id must use letters, digits, underscores or hyphens")
    command = leo_command(run_id, args.prompt, args.next_subtask if args.next_subtask is not None
                          else ([] if args.single_subtask else ["place the red pepper in the bowl"]),
                          args.control_version, args.vlm_progress_threshold, args.vlm_progress_count, args.vlm_auto_takeover, args.dynamics_mode, args.pointcloud_mode, args.astra_pause_handback, args.candidate_distance_m, args.astra_model, args.takeover_mode, args.l2_switch_mode, args.no_step_limit, args.reference_video, args.reference_video_frames)
    session = f"{REMOTE}/outputs/astra/{run_id}"
    launch = ROOT / "outputs/mac_supervisor" / ("launch_" + run_id)
    if not args.dry_run:
        try:
            launch.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error("This run-id was already prepared on Mac; use a new run-id")
        (launch / "leo3_command.sh").write_text(command, encoding="utf-8", newline="\n")
    print(f"V{args.control_version} + VLM recording | RUN_ID={run_id}\nLeo3 tmux: {run_id}\nRunner session: {session}", flush=True)
    if args.control_version in (4, 10):
        print(f'Takeover mode: {selected_modes[0]} (T works both ways) | L2 switch mode: {selected_modes[1]} (P then Enter) | Terminal fallback: {args.astra_pause_handback}',flush=True)
    print(f'Reference video: {args.reference_video} | requested frames: {args.reference_video_frames}'
          if args.reference_video else 'Reference video: none (use --reference-video with a Leo3 absolute path)', flush=True)
    print("\n===== Copy the following block into the Leo3 desktop terminal =====\n" + command
          + "===== End of Leo3 command =====\n", flush=True)
    if args.dry_run:
        return
    print(f"Command saved: {launch / 'leo3_command.sh'}\nKeep this supervisor terminal open; waiting for Leo3.", flush=True)
    bridge_args = ["--session", session, "--host", args.host, "--proxy", args.proxy]
    bridge_args += ['--astra-model',args.astra_model]
    bridge_args += ['--ssh-timeout',str(args.ssh_timeout),'--ssh-connect-timeout',str(args.ssh_connect_timeout),'--ssh-retries',str(args.ssh_retries)]
    if args.codex_bin:
        bridge_args += ["--codex-bin", args.codex_bin]
    if os.name == "nt":
        # Run in this Python process so Ctrl+C reaches the bridge's existing
        # stop handler; Windows needs neither Bash nor a second Python install.
        sys.path.insert(0, str(ROOT))
        from astra.mac_bridge import main as bridge_main
        bridge_main(bridge_args)
        return
    script = ROOT / "wam_direct/run_mac_supervisor.sh"
    os.execvp("bash", ["bash", str(script), *bridge_args])


if __name__ == "__main__":
    main()
