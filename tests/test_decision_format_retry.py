import json
import unittest
from unittest.mock import patch
import test_api_supervisor as helpers


class DecisionFormatRetryTests(unittest.TestCase):
    setUp=helpers.APITests.setUp
    call=helpers.APITests.call

    def test_explanation_with_one_json_fence_needs_no_second_model_call(self):
        response=helpers.completion(None)
        response['choices'][0]['message']['content']='Looking at the observations.\n```json\n'+json.dumps(self.decision)+'\n```\nEnd of explanation.'
        with patch.object(self.bridge,'_http',return_value=response) as send:
            self.assertEqual(self.call(),self.decision)
        self.assertEqual(send.call_count,1)

    def test_ambiguous_incomplete_or_non_json_blocks_rejected(self):
        for content in ('```json\n{}\n```\n```json\n{}\n```','```json\n{}',
                        '```python\n{}\n```','prose {"operation":"observe"}'):
            with self.assertRaises(ValueError):helpers.parse_decision(content)

    def test_prose_or_empty_reply_corrected_without_executing_old_reply(self):
        for text in ('I will inspect the notebook first.', ''):
            with self.subTest(text=text):
                self.bridge.history=[]
                bad=helpers.completion(None);bad['choices'][0]['message']['content']=text
                with patch.object(self.bridge,'_http',side_effect=[bad,helpers.completion(self.decision)]) as send, patch.object(self.bridge,'_tool') as tool:
                    self.assertEqual(self.call(),self.decision)
                self.assertEqual(send.call_count,2);tool.assert_not_called()
                self.assertIn('current identity fields',self.bridge.history[-2]['content'])
                self.assertIn('"request_id": "r"',self.bridge.history[-2]['content'])
        saved=list(self.bridge.path.glob('turn.response_*.json'))
        self.assertEqual(len(saved),4)
        self.assertTrue(any(json.loads(p.read_text())['choices'][0]['message']['content']=='' for p in saved))

    def test_format_retries_are_bounded_and_bad_responses_saved(self):
        bad=helpers.completion(None);bad['choices'][0]['message']['content']='Not JSON'
        with patch.object(self.bridge,'_http',return_value=bad) as send:
            with self.assertRaisesRegex(ValueError,'after 2 correction attempts'):self.call()
        self.assertEqual(send.call_count,3)
        self.assertFalse(self.bridge.history)
        self.assertFalse((self.bridge.path/'turn.json').exists())
        self.assertEqual(len(list(self.bridge.path.glob('turn.response_*.json'))),3)

    def test_reformatted_but_wrong_identity_is_still_rejected(self):
        bad=helpers.completion(None);bad['choices'][0]['message']['content']='Not JSON'
        stale=helpers.completion(dict(self.decision,request_id='old'))
        with patch.object(self.bridge,'_http',side_effect=[bad,stale]),self.assertRaisesRegex(ValueError,'stale/mismatched'):
            self.call()
        self.assertFalse(self.bridge.history)


if __name__=='__main__':unittest.main()
