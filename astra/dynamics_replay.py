"""Read a recorded Astra session and imagine one recovery prefix; never imports ROS."""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image
from .dynamics import native_request,rpc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',default='10.128.39.115');p.add_argument('--port',type=int,default=29642)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in (args.session/'events.jsonl').read_text().splitlines()]
    anchor=None;decision=None
    for row in rows:
        if row['kind']=='request':anchor=row['observation']
        if row['kind']=='decision' and row.get('operation')=='takeover':decision=row;break
    if anchor is None or decision is None:raise ValueError('No recorded Astra takeover')
    obs={k:np.asarray(v) for k,v in anchor['state'].items()}
    obs['timestamp']=float(obs['timestamp'])
    for k in ('image','image_wrist_left','image_wrist_right'):
        # Session PNGs are RGB; restore the live observation BGR contract.
        obs[k]=np.asarray(Image.open(anchor['images'][k]['path']).convert('RGB'))[..., ::-1].copy()
    # This replay targets the recorded deployment's 20fps / 0.1s lead schedule.
    if len(decision['actions'])<18:raise ValueError('Recorded prefix needs 18 waypoints')
    started=time.monotonic();request=native_request(obs,np.asarray(decision['actions'])[:18],20,.1,uuid.uuid4().hex)
    native_seconds=time.monotonic()-started
    result=rpc(args.host,args.port,request,30)
    Image.fromarray(result['future_image']).save(args.output/'future.png')
    Image.fromarray(result['input_image_preview']).save(args.output/'anchor.png')
    np.savez_compressed(args.output/'request_fields.npz',**request['fields'])
    np.savez_compressed(args.output/'prediction.npz',future_proprio=result['future_proprio'],
                        future_proprio_valid_mask=result['future_proprio_valid_mask'])
    summary={'recorded_session':str(args.session),'request_id':request['request_id'],
             'adapter_sha256':result['adapter_sha256'],'norm_sha256':result['norm_sha256'],
             'native_seconds':native_seconds,'total_seconds':time.monotonic()-started,
             'model_seconds':result.get('seconds'),'action_length':request['action_length'],
             'duration_seconds':request['duration_seconds'],'robot_actions_published':False,
             'quality_validated':False}
    (args.output/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
