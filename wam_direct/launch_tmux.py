#!/usr/bin/env python3
"""Launch a local client in tmux and open a Linux desktop terminal on leo1."""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def progress_commands(repo, name, client_args, host, port, gui_env):
    session = next((v.split("=", 1)[1] for v in client_args if v.startswith("--session=")), None)
    if "--session" in client_args:
        index = client_args.index("--session")
        session = client_args[index + 1] if index + 1 < len(client_args) else None
    if not session:
        raise ValueError("--vlm-progress requires the runner's --session")
    path = Path(session).expanduser()
    if not path.is_absolute():
        path = Path(repo) / path
    path = path.resolve()
    output = Path(repo).resolve() / "outputs" / "progress_vlm" / (path.name + "_follow")
    if output.exists():
        raise ValueError(f"VLM output already exists; use a new run/session name: {output}")
    keys = ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "XAUTHORITY",
            "ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION",
            "CYCLONEDDS_URI", "FASTRTPS_DEFAULT_PROFILES_FILE")
    env = [f"{k}={gui_env[k]}" for k in keys if gui_env.get(k)]
    invoke = ["env", *env, "bash", str(Path(repo).resolve() / "wam_direct/run_progress_vlm.sh"),
              "--follow-session", str(path), "--host", host, "--port", str(port), "--output", str(output)]
    return ["tmux", "new-window", "-d", "-t", "=" + name, "-n", "VLM-progress", shlex.join(invoke)], output


def dynamics_commands(repo, name, client_args, gui_env):
    session = next((v.split("=",1)[1] for v in client_args if v.startswith("--session=")), None)
    if "--session" in client_args:
        i = client_args.index("--session")
        session = client_args[i+1] if i+1 < len(client_args) else None
    if not session:
        raise ValueError("--dynamics-view requires --session")
    path = Path(session).expanduser()
    if not path.is_absolute():
        path = Path(repo)/path
    env = [f"{k}={gui_env[k]}" for k in ("DISPLAY","WAYLAND_DISPLAY","XDG_RUNTIME_DIR","DBUS_SESSION_BUS_ADDRESS","XAUTHORITY") if gui_env.get(k)]
    command = shlex.join(["env",*env,"python3","-m","astra.dynamics_viewer","--session",str(path.resolve())])
    return ["tmux","new-window","-d","-t","="+name,"-n","Dynamics-preview","-c",str(Path(repo).resolve()),command]


def rviz_record_commands(repo, name, client_args, gui_env):
    command = dynamics_commands(repo, name, client_args, gui_env)
    # Reuse session/GUI argument handling, but invoke a dedicated window recorder.
    invoke = shlex.split(command[-1])
    invoke[invoke.index("astra.dynamics_viewer")] = "astra.rviz_record"
    session = Path(invoke[invoke.index("--session") + 1])
    output = Path(repo).resolve() / "outputs" / "rviz" / session.name
    invoke += ["--output", str(output)]
    command[command.index("Dynamics-preview")] = "RViz-recording"
    command[-1] = shlex.join(invoke)
    return command


def commands(repo, name, client, client_args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("tmux name must use letters, digits, underscores or hyphens")
    if not client_args:
        raise ValueError("supply client arguments after --")
    if client == "astra" and client_args[0] != "run":
        raise ValueError("Astra client arguments must begin with run")
    script = Path(repo).resolve() / "wam_direct" / f"run_{client}.sh"
    keys = ("PYTHONPATH", "LD_LIBRARY_PATH", "ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE",
            "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI", "FASTRTPS_DEFAULT_PROFILES_FILE")
    inherited = [f"{k}={os.environ[k]}" for k in keys if os.environ.get(k)]
    invoke = shlex.join(["env", *inherited, "bash", str(script), *client_args])
    inner = (f"{invoke}; code=$?; "
             'printf "\\nClient exited (status %s). Press Enter to close this pane.\\n" "$code"; '
             'read -r _; exit "$code"')
    create = ["tmux", "new-session", "-d", "-s", name, "-c", str(Path(repo).resolve()),
              shlex.join(["bash", "-c", inner])]
    attach = ["tmux", "attach-session", "-t", "=" + name]
    return create, attach


def desktop_environment(proc_root=Path("/proc")):
    """Recover only GUI connection variables from this user's existing terminal."""
    env = os.environ.copy()
    if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
        return env
    keys = ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "XAUTHORITY")
    candidates = []
    for process in proc_root.glob("[0-9]*"):
        try:
            if process.stat().st_uid != os.getuid():
                continue
            if process.joinpath("comm").read_text().strip() not in ("gnome-terminal-", "konsole", "xfce4-terminal", "xterm"):
                continue
            raw = dict(item.split(b"=", 1) for item in process.joinpath("environ").read_bytes().split(b"\0") if b"=" in item)
            values = {key: raw[key.encode()].decode() for key in keys if key.encode() in raw}
            if values.get("DISPLAY") or values.get("WAYLAND_DISPLAY"):
                candidates.append(values)
        except (OSError, UnicodeError):
            continue
    if candidates:
        displays = {(v.get("DISPLAY"), v.get("WAYLAND_DISPLAY")) for v in candidates}
        if len(displays) != 1:
            raise ValueError("multiple desktop displays found; launch from the intended desktop terminal")
        env.update(candidates[0])
    return env


def terminal_command(attach, env=None):
    env = os.environ if env is None else env
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        raise ValueError("no desktop display; run from the robot graphical login session")
    for name, separator in (("gnome-terminal", "--window --"), ("konsole", "--separate -e"),
                            ("xfce4-terminal", "--disable-server -x"), ("xterm", "-e")):
        path = shutil.which(name)
        if path:
            return [path, *separator.split(), *attach]
    raise ValueError("no supported desktop terminal found (gnome-terminal, konsole, xfce4-terminal or xterm)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rviz-record", action="store_true", help="record the V6 RViz window alongside this runner")
    parser.add_argument("--name", required=True, help="unique local tmux session name")
    parser.add_argument("--client", choices=("astra", "wam"), default="astra")
    parser.add_argument("--dynamics-view", action="store_true", help="read-only V5 imagination window")
    parser.add_argument("--vlm-progress", action="store_true", help="start a read-only VLM window following this Astra session")
    parser.add_argument("--vlm-host", default="10.128.40.203")
    parser.add_argument("--vlm-port", type=int, default=29573)
    parser.add_argument("--dry-run", action="store_true", help="print commands only; start nothing")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    opts = parser.parse_args(argv)
    args = opts.args[1:] if opts.args[:1] == ["--"] else opts.args
    try:
        create, attach = commands(ROOT, opts.name, opts.client, args)
        if opts.rviz_record:
            if opts.client != "astra":
                raise ValueError("--rviz-record requires --client astra")
            rviz_record_commands(ROOT, opts.name, args, {})
        if opts.dynamics_view:
            dynamics_commands(ROOT, opts.name, args, {})
        if opts.vlm_progress:
            if opts.client != "astra" or not 0 < opts.vlm_port <= 65535:
                raise ValueError("--vlm-progress requires --client astra and a valid VLM port")
            progress_commands(ROOT, opts.name, args, opts.vlm_host, opts.vlm_port, {})
    except ValueError as error:
        parser.error(str(error))
    attach_text = shlex.join(attach)
    if opts.dry_run:
        print("Create:", shlex.join(create))
        if opts.rviz_record:
            print("RViz recorder:", shlex.join(rviz_record_commands(ROOT, opts.name, args, {})))
        if opts.dynamics_view:
            print("Dynamics viewer:", shlex.join(dynamics_commands(ROOT, opts.name, args, {})))
        print("Open local desktop terminal and attach:", attach_text)
        if opts.vlm_progress:
            progress, output = progress_commands(ROOT, opts.name, args, opts.vlm_host, opts.vlm_port, {})
            print("VLM follower:", shlex.join(progress))
            print("VLM output:", output)
        return 0
    # Check the GUI before starting any client. Never silently start headless.
    if not shutil.which("tmux"):
        local_tmux = Path.home() / ".local/bin/tmux"
        if local_tmux.is_file() and os.access(local_tmux, os.X_OK):
            os.environ["PATH"] = str(local_tmux.parent) + os.pathsep + os.environ.get("PATH", "")
        else:
            parser.error("tmux is not installed")
    try:
        gui_env = desktop_environment()
        window = terminal_command([shutil.which("tmux"), *attach[1:]], gui_env)
    except ValueError as error:
        parser.error(str(error))
    subprocess.run(create, check=True, timeout=15)
    if opts.vlm_progress:
        progress, output = progress_commands(ROOT, opts.name, args, opts.vlm_host, opts.vlm_port, gui_env)
        try:
            subprocess.run(progress, check=True, timeout=15)
            print(f"VLM observer follows this runner | tmux window VLM-progress | {output}", flush=True)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"VLM observer launch failed: {error}. Astra runner remains in its own startup flow.", flush=True)
    if opts.client == "astra":
        session_arg = next((a.split("=", 1)[1] for a in args if a.startswith("--session=")), None)
        if "--session" in args:
            index = args.index("--session")
            session_arg = args[index + 1] if index + 1 < len(args) else None
        if session_arg:
            session_path = Path(session_arg).expanduser()
            if not session_path.is_absolute():
                session_path = ROOT / session_path
            log = session_path.resolve() / "supervisor.log"
            subprocess.run(["tmux", "new-window", "-d", "-t", "=" + opts.name,
                            "-n", "Astra-supervisor", shlex.join(["tail", "-n", "+1", "-F", str(log)])],
                           check=True, timeout=15)
            print(f"Astra status: tmux window Astra-supervisor | {log}", flush=True)
    if opts.dynamics_view:
        subprocess.run(dynamics_commands(ROOT, opts.name, args, gui_env), check=True, timeout=15)
    if opts.rviz_record:
        try:
            subprocess.run(rviz_record_commands(ROOT, opts.name, args, gui_env), check=True, timeout=15)
            print("RViz recorder: tmux window RViz-recording; outputs/rviz/<RUN_ID>", flush=True)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"RViz recorder launch failed: {error}", flush=True)
    print(f"Created local tmux session: {opts.name}\nReconnect: {attach_text}", flush=True)
    try:
        process = subprocess.Popen(window, start_new_session=True, env=gui_env)
        try:
            code = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            code = None  # Terminal window remains open; do not wait for rollout.
        if code not in (None, 0):
            raise RuntimeError(f"terminal exited with status {code}")
    except (OSError, RuntimeError) as error:
        print(f"Terminal could not open: {error}. The tmux session remains running.")
        print(f"Open a terminal and run: {attach_text}")
        return 1
    for _ in range(30):
        attached = subprocess.run(["tmux", "list-sessions", "-F",
                                   "#{session_name}:#{session_attached}"], capture_output=True, text=True, timeout=3)
        count = next((line.split(":", 1)[1] for line in attached.stdout.splitlines()
                      if line.startswith(opts.name + ":")), "0")
        if attached.returncode == 0 and count.isdigit() and int(count) > 0:
            print("Terminal attached to tmux. Follow the client keyboard instructions shown in that window.")
            return 0
        time.sleep(.1)
    print(f"No terminal attachment confirmed. Open a terminal and run: {attach_text}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
