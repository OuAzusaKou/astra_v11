"""Notebook/pen V10 preset; model and mode constants may be edited before launch."""
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_direct.start_mac_vlm import main
from wam_direct.start_notebook_test import TASK, SUBTASKS

MODEL = 'astra'
TAKEOVER_MODE = 'manual'
L2_SWITCH_MODE = 'manual'
NO_STEP_LIMIT = True
PROXY = 'http://127.0.0.1:7897' if MODEL == 'astra' else ''

if __name__ == '__main__':
    run = 'notebook_pen_v10_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    args = ['--control-version', '10', '--astra-model', MODEL, '--proxy', PROXY,
            '--run-id', run, '--takeover-mode', TAKEOVER_MODE, '--l2-switch-mode', L2_SWITCH_MODE,
            '--astra-pause-handback', '--prompt', 'Task: '+TASK+' Subtask: '+SUBTASKS[0]]
    for task in SUBTASKS[1:]:
        args += ['--next-subtask', task]
    if NO_STEP_LIMIT:
        args += ['--no-step-limit']
    sys.argv = [sys.argv[0], *args, *sys.argv[1:]]
    main()
