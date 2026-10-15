"""Windows-only CLI resolution and ownership of one supervisor process tree.

Imported lazily: Linux and Darwin never load Win32 APIs. No cmd.exe/Bash quoting.
"""
import ctypes
from ctypes import wintypes
from pathlib import Path
import platform


def native_codex(binary):
    """Resolve npm's script shim to its matching bundled native executable."""
    path = Path(binary)
    if path.suffix.lower() == '.exe':
        return str(path)
    arch = {'amd64': 'x86_64', 'x86_64': 'x86_64', 'arm64': 'aarch64',
            'aarch64': 'aarch64'}.get(platform.machine().lower())
    if arch:
        package_arch = 'x64' if arch == 'x86_64' else 'arm64'
        package = path.parent / 'node_modules/@openai/codex'
        vendor = f'vendor/{arch}-pc-windows-msvc'
        candidates = [
            package / f'node_modules/@openai/codex-win32-{package_arch}' / vendor / 'bin/codex.exe',
            package / vendor / 'codex/codex.exe',
            path.with_name('codex.exe'),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError('Cannot resolve native codex.exe; pass --codex-bin PATH_TO_CODEX_EXE')


class _BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
        'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IoCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class ProcessJob:
    """Closing the job ends this call and its descendants, including tool shells."""
    def __init__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                        wintypes.DWORD], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None
