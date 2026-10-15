"""V4 notebook/pen task; edit MODEL and CONTROL_VERSION before a new run."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wam_direct.start_mac_vlm import main

MODEL = 'opus5.5'
CONTROL_VERSION = 4
NO_STEP_LIMIT = True  # V4: operator stops/completes the run; no total waypoint cap.
TAKEOVER_MODE = 'manual'  # V4: 'manual' (T) or 'vlm'; terminal-error fallback remains enabled.
L2_SWITCH_MODE = 'manual'  # V4: 'manual' (P then Enter) or 'vlm'.
SUPERVISOR_PROXY = '' if MODEL in ('glm5.3-flash','glm-5.3-flash','opus5.5','claude-opus-5-5') else 'http://127.0.0.1:7897'
TASK = 'Open the notebook, pick up the pen inside it, place it in the pen holder, and then close the notebook.'
SUBTASKS = ('open the notebook', 'pick up the pen inside the notebook',
            'place the pen in the pen holder', 'close the notebook')

if __name__=='__main__':
    from datetime import datetime
    import uuid
    run='notebook_pen_v'+str(CONTROL_VERSION)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    options=['--control-version',str(CONTROL_VERSION),'--astra-model',MODEL,
             '--run-id',run,'--proxy',SUPERVISOR_PROXY,'--vlm-auto-takeover','--astra-pause-handback',
             '--prompt','Task: '+TASK+' Subtask: '+SUBTASKS[0]]
    for task in SUBTASKS[1:]: options+=['--next-subtask',task]
    if CONTROL_VERSION in (4, 10):
        options+=['--takeover-mode',TAKEOVER_MODE,'--l2-switch-mode',L2_SWITCH_MODE]
        if NO_STEP_LIMIT:
            options+=['--no-step-limit']
    sys.argv=[sys.argv[0],*options,*sys.argv[1:]]
    main()
