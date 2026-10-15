"""No ROS/network/model calls. Real child processes exercise cancellation."""
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from astra.supervisor import CodexSupervisor
from astra.session import atomic_json
from astra.mac_bridge import Transport
from wam_direct import start_mac_vlm


STUB = r'''
import json, pathlib, subprocess, sys, time
args = sys.argv[1:]
prompt = sys.stdin.read()
pathlib.Path('prompt.txt').write_text(prompt, encoding='utf-8')
if prompt == 'wait':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    pathlib.Path('child.tmp').write_text(str(child.pid))
    pathlib.Path('child.tmp').replace('child.pid')
    time.sleep(60)
else:
    print(json.dumps({'type':'thread.started','thread_id':'12345678-1234-1234-1234-123456789abc'}), flush=True)
    print(json.dumps({'type':'error','message':'中文诊断'}, ensure_ascii=False), flush=True)
    pathlib.Path(args[args.index('-o')+1]).write_text(json.dumps({'ready':True,'text':prompt}, ensure_ascii=False), encoding='utf-8')
'''


def alive(pid):
    if os.name == 'nt':
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(api.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            api.CloseHandle(handle)
    stat = Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().split()[2] != 'Z'


class SupervisorPlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='astra space 中文 ')
        def cleanup():
            # Windows job termination is asynchronous; inherited file handles
            # can outlive the process exit notification briefly.
            for attempt in range(30):
                try:
                    self.temp.cleanup()
                    return
                except PermissionError:
                    if attempt == 29:
                        raise
                    time.sleep(.1)
        self.addCleanup(cleanup)
        self.root = Path(self.temp.name)
        self.bridge = CodexSupervisor(SimpleNamespace(path=self.root, owner='fixture'),
                                      binary=sys.executable, timeout=5)
        self.commands = []
        self.real_popen = subprocess.Popen

    def launch_stub(self, command, **kwargs):
        self.commands.append(command)
        args = command[command.index(sys.executable)+1:]
        kwargs['env'] = {**kwargs['env'], 'PYTHONIOENCODING':'utf-8'}
        return self.real_popen([sys.executable, '-u', '-c', STUB, *args], **kwargs)

    def test_utf8_roundtrip_resume_and_platform_command(self):
        with patch('astra.supervisor.subprocess.Popen', side_effect=self.launch_stub):
            result = self.bridge._call('中文与 spaces & $ 保留', [], threading.Event(), 'first')
            self.bridge._call('second', [], threading.Event(), 'second')
        self.assertEqual(result['text'], '中文与 spaces & $ 保留')
        self.assertIn('CALL_ERROR: 中文诊断', (self.root/'supervisor.log').read_text(encoding='utf-8'))
        self.assertIn('resume', self.commands[1])
        self.assertIn(self.bridge.thread_id, self.commands[1])
        self.assertEqual(self.commands[0][0], sys.executable if platform.system() in ('Windows','Darwin') else 'bash')
        self.assertIn('permissions.astra_observe.network.enabled=true', self.commands[0])
        self.assertIn('gpt-6-astra', self.commands[0])
        for command in self.commands:
            self.assertEqual('model_providers.astra_https.supports_websockets=false' in command,
                             platform.system() in ('Darwin', 'Windows'))

    def check_tree_stopped(self, cancel):
        stop = threading.Event()
        errors = []
        def run():
            try:
                self.bridge._call('wait', [], stop, 'cancel')
            except Exception as error:
                errors.append(error)
        if not cancel:
            self.bridge.timeout = 1.5
        with patch('astra.supervisor.subprocess.Popen', side_effect=self.launch_stub):
            worker = threading.Thread(target=run)
            worker.start()
            deadline = time.monotonic()+4
            pid_file = self.root/'child.pid'
            while not pid_file.exists() and worker.is_alive() and time.monotonic()<deadline:
                time.sleep(.02)
            try:
                self.assertTrue(pid_file.exists(), errors)
                pid = int(pid_file.read_text())
                self.assertTrue(alive(pid))
            finally:
                if cancel:
                    stop.set()
                worker.join(8)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError if cancel else TimeoutError)
        deadline = time.monotonic()+3
        while alive(pid) and time.monotonic()<deadline:
            time.sleep(.02)
        self.assertFalse(alive(pid), 'cancelled supervisor left a descendant alive')

    def test_cancel_terminates_descendants(self):
        self.check_tree_stopped(True)

    def test_timeout_terminates_descendants(self):
        self.check_tree_stopped(False)

    def test_atomic_json_is_utf8(self):
        path = self.root/'中文.json'
        atomic_json(path, {'text':'中文 🦾'})
        self.assertEqual(json.loads(path.read_bytes().decode('utf-8')), {'text':'中文 🦾'})


class LauncherTests(unittest.TestCase):
    def test_dry_run_never_starts_bridge_or_creates_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(start_mac_vlm,'ROOT',Path(directory)), \
             patch.object(sys,'argv',['helper','--dry-run','--control-version','4','--run-id','fixture']), \
             patch.object(start_mac_vlm.os,'execvp') as execute, \
             patch('astra.mac_bridge.main') as bridge:
            start_mac_vlm.main()
            self.assertEqual(list(Path(directory).iterdir()), [])
            execute.assert_not_called()
            bridge.assert_not_called()

    def test_launch_preserves_posix_and_uses_current_python_on_windows(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(start_mac_vlm,'ROOT',Path(directory)), \
             patch.object(sys,'argv',['helper','--run-id','fixture','--proxy','', '--host','leo3']), \
             patch.object(start_mac_vlm.os,'execvp') as execute, \
             patch('astra.mac_bridge.main') as bridge:
            start_mac_vlm.main()
            saved=Path(directory)/'outputs/mac_supervisor/launch_fixture/leo3_command.sh'
            self.assertNotIn(b'\r\n', saved.read_bytes())
            if os.name == 'nt':
                execute.assert_not_called()
                forwarded = bridge.call_args.args[0]
                self.assertEqual(forwarded[forwarded.index('--proxy')+1], '')
                self.assertEqual(forwarded[forwarded.index('--ssh-timeout')+1], '60.0')
            else:
                bridge.assert_not_called()
                self.assertEqual(execute.call_args.args[0], 'bash')

    def test_ssh_uses_utf8_and_bound_session(self):
        with patch('astra.mac_bridge.subprocess.run',return_value=SimpleNamespace(
                stdout='{"waiting_for_runner": true}',stderr='',returncode=0)) as run:
            Transport('leo3','/repo space','/repo space/outputs/astra/new').call('status')
        self.assertEqual(run.call_args.kwargs['encoding'],'utf-8')
        self.assertEqual(run.call_args.kwargs['timeout'],60)
        self.assertIn("'/repo space/outputs/astra/new'", run.call_args.args[0][-1])

    @unittest.skipUnless(os.name == 'nt', 'Windows npm layout')
    def test_npm_shim_resolution_and_failure(self):
        from astra.windows_process import native_codex
        with tempfile.TemporaryDirectory() as directory:
            shim=Path(directory)/'codex.cmd'
            shim.touch()
            with patch('astra.windows_process.platform.machine',return_value='AMD64'):
                with self.assertRaises(FileNotFoundError):
                    native_codex(shim)
                binary=Path(directory)/'node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe'
                binary.parent.mkdir(parents=True)
                binary.touch()
                self.assertEqual(native_codex(shim), str(binary))


if __name__ == '__main__':
    unittest.main()
