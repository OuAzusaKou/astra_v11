"""V6 candidate review via URDF point clouds; no Dynamics service connection."""
import copy,hashlib,json,subprocess,sys,time,uuid,os
from pathlib import Path
import numpy as np
from .dynamics import Imagination,same_anchor,current52
from .session import atomic_json


class PointcloudImagination(Imagination):
    def __init__(self, mode='review', timeout=60., overlay='/home/liberai/camera_pose/rviz_overlay', long_plan=False):
        self.mode=mode;self.timeout=timeout;self.overlay=overlay;self.long_plan=long_plan
        self.pending=None;self.process=None;self.log=None;self.directory=None;self.capture=None

    def cancel(self):
        self.pending=None
        if self.directory is not None:(self.directory/'cancel').touch()
        if self.process is not None:
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log is not None:self.log.close()
        self.process=self.log=None

    def start(self,c,batch):
        self.cancel();obs=c.robot.observe()
        if c.observation is not None and not same_anchor(c.observation,obs):
            raise RuntimeError('Scene/state changed since planning; observe and replan')
        if self.capture is None:raise RuntimeError('V6 scene capture was not initialized')
        frozen=self.capture.freeze(float(obs['timestamp']))
        fps=getattr(c.robot,'fps',20.);lead=getattr(c.robot,'lead_time',.1)
        count=len(batch) if self.long_plan else min(len(batch),int(np.floor((1.-lead)*fps+1e-6)))
        if count<1:raise ValueError('empty V6 candidate')
        actions=np.asarray(batch[:count],dtype=np.float64).copy();ident=uuid.uuid4().hex
        directory=c.session.path/'imagination'/ident;directory.mkdir(parents=True)
        np.savez_compressed(directory/'frozen_scene.npz',**frozen)
        refs=c.session.snapshot(obs,'imagination_'+ident)
        self.pending={'id':ident,'actions':actions,'anchor':copy.deepcopy(obs),'snapshot':refs,
                      'phase':c.phase,'condition':c.condition_version,'created':time.monotonic(),
                      'fps':fps,'lead':lead,'sha256':hashlib.sha256(actions.tobytes()).hexdigest()}
        self.directory=directory;p=self.pending
        if self.long_plan:
            from .prefix_plan import prefix_limit
            p['prefix'] = prefix_limit(actions, current52(obs), c.plan_stages, fps, lead, c.settings.budget-c.steps)
            p['plan_stages'] = copy.deepcopy(c.plan_stages)
        atomic_json(directory/'request.json',{'candidate_id':ident,'candidate_sha256':p['sha256'],
                    'actions':actions.tolist(),'anchor':current52(obs).tolist(),'anchor_timestamp':float(obs['timestamp']),
                    'prefix_steps':p.get('prefix',{}).get('count'),'plan_stages':p.get('plan_stages'),
                    'fps':fps,'lead_time':lead,'l3':c.l3,'l2':c.l2,'parent_pid':os.getpid()})
        self.log=(directory/'worker.log').open('w')
        self.process=subprocess.Popen([sys.executable,'-m','astra.pointcloud_worker','--request',str(directory/'request.json'),
                                      '--overlay',self.overlay],cwd=Path(__file__).resolve().parents[1],stdout=self.log,stderr=subprocess.STDOUT)
        c.session.pending=None;c.state='IMAGINING'
        c.session.event('imagination_started',candidate_id=ident,candidate_sha256=p['sha256'],
                        actions=actions.tolist(),anchor=refs,mode=self.mode,backend='urdf_pointcloud')
        c.status()

    def poll(self,c):
        p=self.pending
        if p is None or c.state!='IMAGINING':return
        c.robot.check()
        if c.phase!=p['phase'] or c.condition_version!=p['condition'] or c.mode!='ASTRA':raise RuntimeError('candidate ownership changed')
        if time.monotonic()-p['created']>self.timeout:raise TimeoutError('V6 pointcloud imagination timed out; inspect worker.log')
        path=self.directory/'result.json'
        if not path.exists():
            if self.process.poll() is not None:
                detail=(self.directory/'worker.log').read_text(errors='replace')[-1600:]
                raise RuntimeError('V6 pointcloud worker failed: '+detail)
            return
        result=json.loads(path.read_text())
        if result['candidate_id']!=p['id'] or result['candidate_sha256']!=p['sha256']:raise ValueError('V6 candidate mismatch')
        if not result.get('images'):raise ValueError('V6 missing review images')
        for asset in result['images']:
            file=Path(asset['path'])
            if file.resolve().parent!=self.directory.resolve() or hashlib.sha256(file.read_bytes()).hexdigest()!=asset['sha256']:
                raise ValueError('V6 review image integrity failure')
        info={**result,'synthetic':True,'not_execution_evidence':True,'actions':p['actions'].tolist(),
              'anchor_timestamp':float(p['anchor']['timestamp']),'target_horizon_seconds':p['lead']+len(p['actions'])/p['fps'],
              'fps':p['fps'],'lead_time':p['lead']}
        if self.long_plan:
            info.update(prefix=p['prefix'],plan_stages=p['plan_stages'],max_execute_steps=p['prefix']['count'])
        metadata={**info,'l3':c.l3,'l2':c.l2,'phase':p['phase'],'saved_at':time.time(),'status_at_save':'prediction_only'}
        atomic_json(self.directory/'metadata.json',metadata)
        with (self.directory.parent/'index.jsonl').open('a') as f:f.write(json.dumps({'candidate_id':p['id'],'metadata':str(self.directory/'metadata.json')})+'\n')
        atomic_json(self.directory.parent/'latest.json',{'candidate_id':p['id'],'panel':result['images'][0]['path'],'backend':'urdf_pointcloud'})
        p['info']=info;c.session.event('imagination_completed',**info)
        if self.mode=='preview':c.pause('V6 geometry preview saved; no candidate executed');return
        c.queue=p['actions'].copy();c.review('REVIEW_IMAGINATION')

    def approve(self,c,d):
        if self.long_plan:
            p=self.pending
            if p is None or 'info' not in p:raise ValueError('no reviewed V7 plan')
            count=d.get('execute_steps')
            if type(count) is not int or not 1 <= count <= p['prefix']['count']:
                raise ValueError('V7 approval exceeds bounded prefix')
            from .prefix_plan import prefix_limit
            fresh_limit=prefix_limit(p['actions'],current52(c.robot.observe()),p['plan_stages'],p['fps'],p['lead'],c.settings.budget-c.steps)
            if count > fresh_limit['count']:
                raise ValueError('V7 prefix no longer valid from fresh measured state; observe/replan')
            # Base approval validates identity, ownership, clock, anchor and the full immutable plan.
            full=dict(d,execute_steps=len(p['actions']))
            super().approve(c,full)
            c.approved_imagination['plan_target_horizon_seconds']=c.approved_imagination['target_horizon_seconds']
            c.approved_imagination['target_horizon_seconds']=p['lead']+count/p['fps']
            c.approved_imagination['executed_prefix_steps']=count
            c.session.event('v7_prefix_approved',candidate_id=p['id'],planned_steps=len(p['actions']),execute_steps=count,limits=p['prefix'])
        else:
            count=super().approve(c,d)
        self.cancel()
        return count
