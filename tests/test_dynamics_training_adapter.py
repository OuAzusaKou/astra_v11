"""Run on PRO6000 with V5_TRAINING_REPO and V5_TRAINING_RUN; no GPU requests."""
import os
import unittest
import numpy as np
from astra.dynamics import PROTOCOL,ADAPTER_SHA,NORM_SHA


@unittest.skipUnless(os.environ.get('V5_TRAINING_REPO'),'training runtime not selected')
class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from astra.dynamics_gateway import TrainingAdapter
        cls.a=TrainingAdapter(os.environ['V5_TRAINING_REPO'],os.environ['V5_TRAINING_RUN'])

    def request(self,h=30):
        identity=np.array([1,0,0,0,1,0],np.float32)
        fields={}
        for side in ('left','right'):
            for typ in ('observation','action'):
                fields[f'{typ}.mano.{side}_ee_'+('state' if typ=='observation' else 'action')]=np.tile(np.r_[np.zeros(3,np.float32),identity],(30,1))
                fields[f'{typ}.mano.{side}_hand_theta']=np.tile(identity,(30,15))
            fields[f'observation.mano.{side}_hand_beta']=np.zeros((30,10),np.float32)
        return dict(protocol=PROTOCOL,request_id='test',dataset_name='teleop_v22',
                    adapter_sha256=ADAPTER_SHA,norm_sha256=NORM_SHA,fields=fields,
                    images=[np.full((64,96,3),v,np.uint8) for v in (32,96,192)],
                    action_length=h,duration_seconds=h/30)

    def test_exact_media_and_normalization_with_masks(self):
        r=self.a.assemble(self.request())
        self.assertEqual(r['input_image'].shape,(1,3,368,320))
        np.testing.assert_allclose(r['input_image'][0,:,100,100],32/127.5-1,atol=1e-6)
        np.testing.assert_allclose(r['input_image'][0,:,300,50],96/127.5-1,atol=1e-6)
        np.testing.assert_allclose(r['input_image'][0,:,300,250],192/127.5-1,atol=1e-6)
        self.assertTrue(np.isfinite(r['action']).all())
        self.assertTrue((r['action'][~r['action_valid_mask']]==0).all())
        self.assertFalse(r['action_valid_mask'][0,:,100:125].any())
        np.testing.assert_array_equal(r['future_proprio_valid_mask'],r['proprio_valid_mask'][0])
        np.testing.assert_allclose(r['action'][0,0,3:9],[1,0,0,0,1,0])

    def test_padding_zero_after_normalization(self):
        req=self.request(15);a=self.a.assemble(req)
        for k,x in req['fields'].items():
            if k.startswith('action.') and 'ee_action' in k:x[15:,:3]=100
        b=self.a.assemble(req)
        np.testing.assert_array_equal(a['action'],b['action'])
        self.assertFalse(b['action_valid_mask'][:,15:].any())
        self.assertFalse(b['action'][:,15:].any())

    def test_bad_norm_nan_and_missing_fields_rejected(self):
        r=self.request();r['norm_sha256']='bad'
        with self.assertRaises(ValueError):self.a.assemble(r)
        r=self.request();r['fields']['action.mano.left_ee_action'][3,0]=np.nan
        with self.assertRaises(ValueError):self.a.assemble(r)
        r=self.request();del r['fields']['observation.mano.left_hand_beta']
        with self.assertRaises(KeyError):self.a.assemble(r)
