"""Independent V4 takeover and subtask triggers; manual keys remain available."""

def resolve_modes(version, takeover_mode=None, l2_switch_mode=None, legacy_auto_takeover=False):
    if version not in (4, 10) and (takeover_mode is not None or l2_switch_mode is not None):
        raise ValueError('--takeover-mode and --l2-switch-mode require V4 or V10')
    for value in (takeover_mode, l2_switch_mode):
        if value is not None and value not in ('manual', 'vlm'):
            raise ValueError('switch mode must be manual or vlm')
    return (takeover_mode or ('vlm' if legacy_auto_takeover else 'manual'), l2_switch_mode or 'vlm')


def configure_vlm_switching(controller, args):
    from .vlm_gate import ProgressGate, TakeoverGate
    controller.vlm_gate = None
    controller.vlm_takeover = None
    if args.l2_switch_mode == 'vlm':
        controller.vlm_gate = ProgressGate(args.vlm_progress_log, args.vlm_progress_threshold,
                                           args.vlm_progress_count)
        controller.session.event('vlm_auto_l2_enabled', log=args.vlm_progress_log,
                                 threshold=args.vlm_progress_threshold, count=args.vlm_progress_count)
    if args.takeover_mode == 'vlm':
        controller.vlm_takeover = TakeoverGate(args.vlm_progress_log)
        controller.session.event('vlm_auto_takeover_enabled', min_l2_seconds=10, no_new_high_seconds=12,
                                 replay_count=0, replay_required=False, window_count=8, min_window_seconds=6,
                                 progress_below=90, handback_cooldown_seconds=15)
    controller.switch_modes = dict(takeover=args.takeover_mode, l2=args.l2_switch_mode,
                                  terminal_fallback=controller.settings.astra_pause_handback)
    controller.session.event('switch_modes', **controller.switch_modes)
    print(f'[V{controller.settings.version}] Takeover: {args.takeover_mode}; T switches WAM/ASTRA both ways. '
          f'L2: {args.l2_switch_mode}; P selects next L2, Enter confirms (return to WAM with T first). '
          f'Code fallback on terminal errors/stops: {controller.settings.astra_pause_handback}. '
          'Initial start remains manual Enter.', flush=True)
