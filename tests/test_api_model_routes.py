"""Leo3 model identity is pinned at session startup, including reconnects."""
import json,tempfile,unittest,uuid
from pathlib import Path
from astra.session import Session
from astra.remote_io import handle
from astra.supervisor_models import model_info
from wam_direct.start_mac_vlm import leo_command


class ModelRoutes(unittest.TestCase):
    def test_v4_v8_launch_pin_model_and_keep_defaults(self):
        for version in (4,8):
            for model in ('opus5.5','gemini','glm5.3-flash','qwen3.8-max'):
                command=leo_command('fixture','Task: open Subtask: pull',[],version=version,astra_model=model)
                self.assertIn('--astra-model '+model,command)
        self.assertNotIn('--astra-model',leo_command('fixture','Task: open Subtask: pull',[]))
        with self.assertRaises(ValueError):leo_command('fixture','Task: open Subtask: pull',[],version=5,astra_model='gemini')

    def test_wrong_model_ready_cannot_acquire_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Session(Path(tmp)/'run','owner',{'settings':{'supervisor_backend':'mac','astra_model':'gemini'}})
            s.status(state='STARTING');client=str(uuid.uuid4())
            packet={'client_id':client,'operation':'ready','thread_id':str(uuid.uuid4()),**model_info('opus5.5')}
            with self.assertRaises(ValueError):handle(s.path,packet)
            self.assertFalse((s.path/'mac_peer.json').exists())
            packet.update(model_info('gemini'));reply=handle(s.path,packet)
            self.assertEqual(reply['peer']['model'],'gemini-3.8-flash')
            self.assertEqual(reply['manifest']['astra_model'],'gemini')
            packet.update(model_info('glm5.3-flash'))
            with self.assertRaises(ValueError):handle(s.path,packet)

    def test_qwen_ready_requires_tool_free_mode_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Session(Path(tmp)/'run','owner',{'settings':{'supervisor_backend':'mac','astra_model':'qwen3.8-max'}})
            s.status(state='STARTING')
            packet={'client_id':str(uuid.uuid4()),'operation':'ready','thread_id':str(uuid.uuid4()),**model_info('qwen3.8-max')}
            del packet['inspection_mode']
            with self.assertRaises(ValueError):handle(s.path,packet)
            packet['inspection_mode']='preloaded_no_tools'
            reply=handle(s.path,packet)
            self.assertEqual(reply['peer']['inspection_mode'],'preloaded_no_tools')


if __name__=='__main__':unittest.main()
