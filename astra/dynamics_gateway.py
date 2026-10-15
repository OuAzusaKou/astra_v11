"""CPU native-to-canonical adapter beside the existing DynamicsWAM server.

Run with the training environment and --training-repo. No model weights are loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socketserver
import sys
from types import SimpleNamespace
import numpy as np
from policy_server.msgpack_numpy import packb, unpackb
from .dynamics import PROTOCOL, ADAPTER_SHA, NORM_SHA, rpc


class TrainingAdapter:
    def __init__(self, repo, run):
        sys.path.insert(0,str(Path(repo).resolve()))
        import torch
        from wam.dataset.lerobot.config import load_adapter_config
        from wam.dataset.lerobot.assembly import AdapterTensorAssembler, all_valid_mapping_validity
        from wam.dataset.lerobot.temporal import interpolate_source,apply_transform,safe_value
        from wam.dataset.lerobot.norm import project_field_norm
        from wam.dataset.lerobot.dataset import LerobotSampleCore
        from wam.dataset.lerobot.media import build_mosaic
        self.torch,self.core,self.mosaic = torch,LerobotSampleCore,build_mosaic
        torch.set_num_threads(2)
        run = Path(run)
        ap=run/'adapter_config.yaml';npth=run.parents[1]/'assets/norm/norm_v22_H30.json'
        self.contract={'adapter_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),
                       'norm_sha256':hashlib.sha256(npth.read_bytes()).hexdigest()}
        if self.contract != {'adapter_sha256':ADAPTER_SHA,'norm_sha256':NORM_SHA}:
            raise ValueError('training assets do not match pinned 40k deployment')
        cfg=load_adapter_config(ap,action_horizon=30);self.adapter=cfg['adapters']['teleop_mano'];can=cfg['canonical']
        self.assembler=AdapterTensorAssembler(dataset_name='teleop_v22',adapter=self.adapter,
            state_offsets={f['name']:tuple(f['dims']) for f in can['state']},
            action_offsets={f['name']:tuple(f['dims']) for f in can['action']},
            state_dim=346,action_dim=326,action_horizon=30,
            interpolate_source=interpolate_source,apply_transform=apply_transform,safe_value=safe_value)
        self.validity=all_valid_mapping_validity(self.adapter,30)
        proj=project_field_norm(json.loads(npth.read_text()),state_mappings=self.adapter['state'],
            action_mappings=self.adapter['action'],state_fields=can['state'],action_fields=can['action'],
            state_dim=346,action_dim=326,horizon=30)
        binding={w:{k:proj[w+'_'+k] for k in ('scale','offset','clip_min','clip_max','valid')} for w in ('state','action')}
        self.norm_context=SimpleNamespace(normalizers={'teleop_v22':binding})
        self.source=SimpleNamespace(name='teleop_v22',group='Liber_Teleop')

    def assemble(self,req):
        if req.get('protocol')!=PROTOCOL or req.get('dataset_name')!='teleop_v22':
            raise ValueError('expected native teleop request')
        if any(req.get(k)!=v for k,v in self.contract.items()):
            raise ValueError('native request training fingerprint mismatch')
        h=req['action_length'];duration=req['duration_seconds']
        if type(h) is not int or not 15<=h<=30 or not np.isfinite(duration) or abs(h/30-duration)>1e-6:
            raise ValueError('native action timing mismatch')
        arrays={}
        for mapping in self.adapter['state']+self.adapter['action']:
            key=mapping['source'];raw=np.asarray(req['fields'][key])
            width=10 if key.endswith('_hand_beta') else 90 if key.endswith('_hand_theta') else 9
            if raw.shape!=(30,width) or raw.dtype!=np.float32 or not np.isfinite(raw).all():
                raise ValueError('invalid or missing native field '+key)
            arrays[key]=raw
        state,sm=self.assembler.assemble_state(arrays=arrays,anchor=0,validity=self.validity)
        action,am=self.assembler.assemble_action(arrays=arrays,timestamps=np.arange(30)/30,
            query_times=np.arange(30)/30,anchor=0,mapping_validity=self.validity,validity_resolver=None)
        am[h:]=False
        state=self.core._normalize(self.norm_context,self.source,'state',state,sm)
        action=self.core._normalize(self.norm_context,self.source,'action',action,am)
        frames=[]
        if len(req['images'])!=3:raise ValueError('three RGB views required')
        for im in req['images']:
            im=np.asarray(im)
            if im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3 or min(im.shape[:2])<16 or max(im.shape[:2])>4096:
                raise ValueError('invalid RGB image')
            frames.append(self.torch.from_numpy(np.array(im,copy=True)).permute(2,0,1)[None])
        image=self.mosaic(frames,self.adapter['media']['mosaic'])[:,0].numpy()[None]
        return {'protocol':'dynamics-canonical-v1','request_id':req['request_id'],
                'dataset_name':'teleop_v22',**self.contract,'input_image':image,
                'proprio':state[None],'proprio_valid_mask':sm[None],
                'action':action[None],'action_valid_mask':am[None],
                'action_lengths':np.array([h],np.int64),
                'target_horizon_frames':np.array([duration*30],np.float32),
                'future_proprio_valid_mask':sm.copy(),'num_inference_steps':10,'seed':42}


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(30)
        def read(n):
            out=bytearray()
            while len(out)<n:
                x=self.request.recv(n-len(out))
                if not x:raise ConnectionError('closed native socket')
                out.extend(x)
            return bytes(out)
        ident=None
        try:
            size=int.from_bytes(read(4),'big')
            if not 0<size<=16*1024*1024:raise ValueError('native payload exceeds limit')
            req=unpackb(read(size));ident=req.get('request_id')
            if req.get('cmd')=='health':
                upstream=rpc(self.server.host,self.server.port,{'cmd':'health','request_id':ident})
                if any(upstream['contract'].get(k)!=v for k,v in self.server.adapter.contract.items()):
                    raise ValueError('upstream assets changed')
                result={'success':True,'request_id':ident,'protocol':PROTOCOL,**self.server.adapter.contract}
            else:
                canonical=self.server.adapter.assemble(req)
                result=rpc(self.server.host,self.server.port,canonical)
                result.update(self.server.adapter.contract)
                result["input_image_preview"] = np.clip((canonical["input_image"][0].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)
        except Exception as exc:
            result={'success':False,'request_id':ident,'error':str(exc)}
        payload=packb(result);self.request.sendall(len(payload).to_bytes(4,'big')+payload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-repo',required=True);p.add_argument('--run',required=True)
    p.add_argument('--bind',default='127.0.0.1');p.add_argument('--port',type=int,default=29642)
    p.add_argument('--model-host',default='127.0.0.1');p.add_argument('--model-port',type=int,default=29640)
    args=p.parse_args();adapter=TrainingAdapter(args.training_repo,args.run)
    class Server(socketserver.TCPServer):allow_reuse_address=True
    with Server((args.bind,args.port),Handler) as server:
        server.adapter=adapter;server.host=args.model_host;server.port=args.model_port
        print(f'READY native adapter {args.bind}:{args.port}',flush=True);server.serve_forever()

if __name__=='__main__':main()
