import unittest
from unittest.mock import patch
from astra.vlm_gate import TakeoverGate
from test_vlm_auto_l2 import V4Tests

class TakeoverTests(V4Tests):
    def setup_takeover(self):
        self.t=self.c.vlm_takeover=TakeoverGate(self.path)
        self.t.poll(self.c,now=100)
    def feed(self, t, replay=False, progress=30):
        self.append(self.row(t,progress,anomalous_replay=replay))
        return self.t.poll(self.c,now=t+.5)
    def test_stall_three_replays_in_eight_triggers_once_on_final_l2(self):
        self.c.subtask_index=2;self.setup_takeover()
        for t in range(101,114):
            decision=self.feed(t,replay=t>=111)
        self.assertIsNotNone(decision)
        with patch('astra.vlm_gate.time.time',return_value=113.5):
            self.assertTrue(self.c.try_vlm_takeover())
            self.assertFalse(self.c.try_vlm_takeover())
        self.assertEqual(self.c.mode,'ASTRA')
        self.assertEqual(self.session.pending['stage'],'ASTRA_TAKEOVER')
        self.assertEqual(self.robot.published,[])
    def test_stall_without_any_replay_triggers(self):
        self.setup_takeover()
        for t in range(101,113):
            self.assertIsNone(self.feed(t,replay=False))
        result=self.feed(113,replay=False)
        self.assertIsNotNone(result)
        self.assertEqual(result['replay_count'],0)
        self.assertFalse(self.t.status['replay_required'])
        self.assertEqual(self.t.status['required_replays'],0)

    def test_new_high_resets_stagnation_without_replay(self):
        self.setup_takeover()
        for t in range(101,113):self.assertIsNone(self.feed(t,replay=False))
        self.assertIsNone(self.feed(113,False,40))
        for t in range(114,125):self.assertIsNone(self.feed(t,False,40))
        self.assertIsNotNone(self.feed(125,False,40))

    def test_invalid_gap_duplicate_and_near_complete_do_not_trigger(self):
        self.setup_takeover()
        for t in range(101,113):self.feed(t,True)
        self.append({'event':'observation_error'})
        self.assertIsNone(self.feed(113,True))
        self.assertIsNone(self.feed(117,True))
        self.assertIsNone(self.feed(117,True))
        for t in range(118,133):self.assertIsNone(self.feed(t,True,90))
    def test_handback_cooldown(self):
        self.setup_takeover()
        self.c.keyboard_toggle();self.t.poll(self.c,now=101)
        from dataclasses import replace
        self.c.settings=replace(self.c.settings,astra_pause_handback=True)
        self.assertTrue(self.c.supervisor_stopped('final model error'))
        with patch('astra.vlm_gate.time.time',return_value=102):self.t.poll(self.c,now=102)
        for t in range(103,117):self.assertIsNone(self.feed(t,True))
        self.assertIsNotNone(self.feed(117,True))

    def test_delayed_consumption_preserves_valid_history_and_uses_fresh_latest(self):
        self.setup_takeover()
        for t in range(101,114):
            self.append(self.row(t,30 if t==101 else 20,anomalous_replay=t>=111))
        # WAM blocked the reader: early records are old now, but were fresh on arrival.
        evidence=self.t.poll(self.c,now=113.5)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence['peak'],30)
        self.assertEqual(evidence['no_new_high_seconds'],12)

    def test_expiration_and_invalid_records_preserve_peak_but_cannot_authorize(self):
        self.setup_takeover()
        for t in range(101,114):self.append(self.row(t,30 if t==101 else 10,anomalous_replay=True))
        self.assertIsNone(self.t.poll(self.c,now=120))
        self.assertEqual(self.t.peak,30)
        self.assertEqual(self.t.samples,[])
        self.append({'event':'observation_error'})
        self.assertIsNone(self.feed(121,True,10))
        self.assertEqual(self.t.peak,30)
        self.assertEqual(len(self.t.samples),1)

    def test_real_condition_change_resets_peak(self):
        self.setup_takeover();self.feed(101,False,70)
        self.c.next_subtask();self.c.start()
        self.assertIsNone(self.t.poll(self.c,now=102))
        self.assertIsNone(self.t.peak)
        self.feed(103,False,10)
        self.assertEqual(self.t.peak,10)

    def test_arrival_invalid_old_record_never_sets_peak(self):
        self.setup_takeover()
        self.append(self.row(101,80,age_on_arrival=4,time=105))
        self.t.poll(self.c,now=106)
        self.assertIsNone(self.t.peak)

if __name__=='__main__':unittest.main()
