"""Offline trigger selection and manual-key behavior; no ROS or real inference."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from astra.control import Controller, Settings
from astra.session import Session
from astra.switch_modes import resolve_modes, configure_vlm_switching
from test_astra_mode2 import Robot, Policy
from wam_direct.start_mac_vlm import leo_command


class V4SwitchModes(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.session=Session(Path(self.temp.name)/'run','fixture',{})
        self.robot=Robot(); self.policy=Policy()
        self.c=Controller(self.robot,self.policy,self.session,Settings(100,version=4,astra_pause_handback=True),
                          'notebook task','open notebook',['pick up pen','place pen','close notebook'])
        self.args=SimpleNamespace(vlm_progress_log=str(Path(self.temp.name)/'predictions.jsonl'),
                                  vlm_progress_threshold=100,vlm_progress_count=2)

    def configure(self,takeover,l2):
        self.args.takeover_mode=takeover;self.args.l2_switch_mode=l2
        configure_vlm_switching(self.c,self.args)

    def test_independent_combinations_and_backward_defaults(self):
        for takeover in ('manual','vlm'):
            for l2 in ('manual','vlm'):
                self.configure(takeover,l2)
                self.assertEqual(self.c.vlm_takeover is not None,takeover=='vlm')
                self.assertEqual(self.c.vlm_gate is not None,l2=='vlm')
                self.assertTrue(self.c.settings.astra_pause_handback)
                self.c.status()
                status=json.loads((self.session.path/'status.json').read_text())
                self.assertEqual(status['switch_modes']['takeover'],takeover)
                self.assertEqual(status['switch_modes']['l2'],l2)
        self.assertEqual(resolve_modes(4),('manual','vlm'))
        self.assertEqual(resolve_modes(4,legacy_auto_takeover=True),('vlm','vlm'))
        self.assertEqual(resolve_modes(4,'manual','manual',True),('manual','manual'))
        with self.assertRaises(ValueError):resolve_modes(8,'manual')

    def test_manual_keys_both_directions_and_l2_enter_gate(self):
        self.configure('manual','manual');self.c.start();self.c.steps=11
        self.assertFalse(self.c.try_vlm_takeover());self.assertFalse(self.c.try_vlm_subtask())
        self.c.keyboard_toggle();self.assertEqual(self.c.mode,'ASTRA')
        self.assertFalse(self.c.next_subtask())  # T back before P.
        self.c.keyboard_toggle();self.assertEqual(self.c.mode,'WAM')
        self.assertTrue(self.c.next_subtask())
        self.assertEqual(self.c.l2,'pick up pen');self.assertEqual(self.c.state,'WAITING_FOR_START')
        self.assertEqual(self.c.steps,11);self.assertEqual(self.robot.published,[])
        self.c.auto_step();self.assertEqual(self.policy.calls,[])
        self.c.start();self.assertEqual(self.c.state,'AUTO')

    def test_manual_mode_keeps_terminal_fallback_and_hardware_guard(self):
        self.configure('manual','manual');self.c.start();self.c.keyboard_toggle()
        self.assertTrue(self.c.supervisor_stopped('final API failure'))
        self.assertEqual(self.c.mode,'WAM')
        self.c.keyboard_toggle();self.robot.fault=True
        self.assertFalse(self.c.supervisor_stopped('final API failure'))
        self.assertEqual(self.c.mode,'ASTRA');self.assertEqual(self.c.state,'PAUSED_FOR_OPERATOR')
        self.assertEqual(self.robot.published,[])

    def test_helper_forwards_modes_and_keeps_recording_and_fallback(self):
        for takeover in ('manual','vlm'):
            for l2 in ('manual','vlm'):
                command=leo_command('fixture','Task: notebook Subtask: open',[],version=4,
                                    auto_takeover=True,pause_handback=True,takeover_mode=takeover,l2_switch_mode=l2)
                self.assertIn('--takeover-mode '+takeover,command)
                self.assertIn('--l2-switch-mode '+l2,command)
                self.assertEqual('--vlm-auto-takeover' in command,takeover=='vlm')
                self.assertIn('--vlm-progress',command);self.assertIn('--astra-pause-handback',command)
        with self.assertRaises(ValueError):leo_command('fixture','x',[],version=8,l2_switch_mode='manual')


if __name__=='__main__':unittest.main()
