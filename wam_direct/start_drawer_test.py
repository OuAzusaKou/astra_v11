"""Edit MODEL / CONTROL_VERSION here, then launch a new drawer test."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wam_direct.start_mac_vlm import main

# Manual startup configuration. Never changes an already running session.
MODEL = 'gemini'  # 'astra', 'opus5.5', 'gemini', 'glm5.3-flash', 'qwen3.8-max'
CONTROL_VERSION = 4  # 4 or 8
TAKEOVER_MODE = 'vlm'  # V4: 'manual' (T) or 'vlm'; terminal-error fallback remains enabled.
L2_SWITCH_MODE = 'vlm'  # V4: 'manual' (P then Enter) or 'vlm'.
DIRECTION_DISTANCE_M = 0.10
SUPERVISOR_PROXY = '' if MODEL in ('glm5.3-flash', 'glm-5.3-flash') else 'http://127.0.0.1:7897'

if __name__=='__main__':
    from datetime import datetime
    import uuid
    run='drawer_open_v'+str(CONTROL_VERSION)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    # Additional flags (e.g. --dry-run) can be supplied on the command line.
    modes=['--takeover-mode',TAKEOVER_MODE,'--l2-switch-mode',L2_SWITCH_MODE] if CONTROL_VERSION == 4 else []
    sys.argv=[sys.argv[0],'--control-version',str(CONTROL_VERSION),'--astra-model',MODEL,
        '--run-id',run,'--proxy',SUPERVISOR_PROXY,'--single-subtask','--dynamics-mode','review','--candidate-distance-m',str(DIRECTION_DISTANCE_M),
        '--vlm-auto-takeover','--astra-pause-handback','--prompt',
        'Task: Pull the storage rack drawer outward by a moderate distance. Subtask: pull out the drawer',*modes,*sys.argv[1:]]
    main()
