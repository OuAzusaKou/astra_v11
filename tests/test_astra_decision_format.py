import copy
import tempfile
import unittest
from pathlib import Path
from astra.decision import normalize_decision
from astra.control import Controller, Settings
from astra.session import Session
from test_astra_mode2 import Robot, Policy

class DecisionFormatTests(unittest.TestCase):
    def test_list_conversion_preserves_decision_and_input(self):
        d={'operation':'pause','evidence':['first observation','second observation'],
           'reason':'unclear clearance','request_id':'original','actions':[[1,2]]}
        original=copy.deepcopy(d)
        result=normalize_decision(d)
        self.assertEqual(result['evidence'],'first observation\nsecond observation')
        self.assertEqual(d,original)
        self.assertEqual({k:v for k,v in result.items() if k!='evidence'},
                         {k:v for k,v in d.items() if k!='evidence'})

    def test_invalid_evidence_not_stringified(self):
        for evidence in ([],[''],['a',None],['a',1],[{'text':'a'}],{'text':'a'},None,False,42):
            with self.subTest(evidence=evidence),self.assertRaises(ValueError):
                normalize_decision({'evidence':evidence})

    def test_string_unchanged(self):
        self.assertEqual(normalize_decision({'evidence':' existing text '}),{'evidence':' existing text '})

    def test_runner_accepts_list_pause_without_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            robot=Robot();session=Session(Path(temp)/'run','owner',{})
            c=Controller(robot,Policy(),session,Settings(100,version=2),'task','subtask')
            c.start();c.keyboard_toggle()
            d={k:session.pending[k] for k in ('session_id','owner','request_id')}
            d.update(operation='pause',reason='model uncertainty',evidence=['image observation','trajectory observation'])
            c.decide(d)
            self.assertEqual(c.state,'PAUSED_FOR_OPERATOR')
            self.assertEqual(robot.published,[])
            events=(session.path/'events.jsonl').read_text()
            self.assertIn('decision_format_normalized',events)
            self.assertNotIn('invalid Astra response',events)
