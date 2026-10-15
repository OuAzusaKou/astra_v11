"""Record only the V6 RViz X11 window. No robot or supervisor commands."""
import argparse,json,os,re,signal,struct,subprocess,time
from pathlib import Path
import numpy as np
from .session import atomic_json
from progress_vlm.recording import WindowRecorder


def decode_xwd(data):
    if len(data)<100:raise ValueError('truncated XWD header')
    h=struct.unpack('>25I',data[:100])
    size,version,fmt,depth,width,height,xoff,order,unit,bitorder,pad,bpp,stride,visual,red,green,blue,bits,entries,ncolors,*_=h
    if version!=7 or fmt!=2 or bpp not in (24,32) or order not in (0,1) or xoff!=0:
        raise ValueError('unsupported XWD format')
    if not 1<=width<=8192 or not 1<=height<=8192 or stride<width*bpp//8 or size<100:
        raise ValueError('invalid XWD dimensions')
    offset=size+ncolors*12
    if len(data)<offset+height*stride:raise ValueError('truncated XWD pixels')
    raw=np.frombuffer(data,dtype=np.uint8,count=height*stride,offset=offset).reshape(height,stride)
    raw=raw[:,:width*(bpp//8)].reshape(height,width,bpp//8).astype(np.uint32)
    shifts=np.arange(bpp//8)*8
    if order==1:shifts=shifts[::-1]
    values=(raw<<shifts).sum(2,dtype=np.uint32)
    channels=[]
    for mask in (blue,green,red):
        if not mask:raise ValueError('missing XWD RGB masks')
        shift=(mask&-mask).bit_length()-1;maximum=mask>>shift
        channels.append((((values&mask)>>shift)*255//maximum).astype(np.uint8))
    return np.stack(channels,axis=-1)


def find_window(env):
    tree=subprocess.run(['xwininfo','-root','-tree'],env=env,capture_output=True,text=True,timeout=3,check=True).stdout
    matches=[]
    for line in tree.splitlines():
        if 'version6_pointcloud.rviz' not in line or '("rviz2" "rviz2")' not in line:continue
        m=re.search(r'(0x[0-9a-fA-F]+).*?\s(\d+)x(\d+)\+',line)
        if m and int(m[2])>100 and int(m[3])>100:matches.append(m[1])
    if len(matches)>1:raise RuntimeError('multiple V6 RViz windows; close the duplicate')
    return matches[0] if matches else None


def runner_alive(session):
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=p.joinpath('cmdline').read_bytes().split(b'\0')
            if b'astra.run' not in argv:continue
            if b'--session' in argv:
                i=argv.index(b'--session')
                if i+1<len(argv) and Path(os.fsdecode(argv[i+1])).resolve()==session:return True
            if ('--session='+str(session)).encode() in argv:return True
        except (OSError,ValueError):pass
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,help='bounded recording check')
    args=p.parse_args();session=args.session.expanduser().resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    from wam_direct.launch_tmux import desktop_environment
    env=desktop_environment()
    if not env.get('DISPLAY'):raise RuntimeError('RViz recording requires an X11/XWayland DISPLAY')
    import cv2
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    recorder=WindowRecorder(out/'window_video',fps=10,segment_seconds=60)
    index=(out/'frames.jsonl').open('x',buffering=1)
    started=time.monotonic();seen=False;last_alive=started;last_probe=0;last_window_check=0;window=None;runner={};last_message=None
    status='STARTING';error=None;captures=0
    try:
        while not stopped:
            now=time.monotonic()
            if args.seconds is not None and now-started>=args.seconds:break
            if now-last_probe>=1:
                alive=runner_alive(session);last_probe=now
                if alive:seen=True;last_alive=now
                if not args.seconds and ((seen and not alive and now-last_alive>3) or (not seen and now-started>180)):
                    status='RUNNER_ENDED' if seen else 'RUNNER_NOT_STARTED';break
                try:runner=json.loads((session/'status.json').read_text())
                except (OSError,ValueError):runner={}
            if now-last_window_check>=1:
                window=find_window(env);last_window_check=now
            if window:
                try:
                    raw=subprocess.run(['xwd','-silent','-id',window],env=env,capture_output=True,timeout=3,check=True).stdout
                    frame=decode_xwd(raw);status='RECORDING';error=None;captures+=1
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    frame=None;window=None;status='CAPTURE_UNAVAILABLE';error=str(exc)
            else:frame=None;status='WAITING_FOR_V6_RVIZ'
            canvas=np.zeros((1008,1280,3),np.uint8)
            if frame is not None:
                h,w=frame.shape[:2];scale=min(1280/w,960/h);w,h=round(w*scale),round(h*scale)
                canvas[(960-h)//2:(960-h)//2+h,(1280-w)//2:(1280-w)//2+w]=cv2.resize(frame,(w,h))
            else:cv2.putText(canvas,status,(40,460),cv2.FONT_HERSHEY_SIMPLEX,1,(0,180,255),2)
            label=f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {runner.get('state','starting')} | {runner.get('control_mode','')} | phase {runner.get('phase','?')}"
            cv2.putText(canvas,label,(10,980),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(canvas,str(runner.get('l2_instruction',''))[:130],(10,1000),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            recorder.write(canvas)
            index.write(json.dumps({'time':time.time(),'elapsed':time.monotonic()-started,'window_id':window,'capture_status':status,
                                    'state':runner.get('state'),'phase':runner.get('phase'),'l2':runner.get('l2_instruction')})+'\n')
            if status!=last_message:
                print('[RVIZ RECORD] '+status+' | '+str(out),flush=True);last_message=status
            atomic_json(out/'status.json',{'status':status,'updated_at':time.time(),'session':str(session),
                                          'window_id':window,'captured_frames':captures,'error':error})
            time.sleep(max(0,.1-(time.monotonic()-now)))
    except Exception as exc:
        status="FAILED";error=str(exc)
        raise
    finally:
        recorder.close();index.close()
        atomic_json(out/'status.json',{'status':'CLOSED','reason':status,'updated_at':time.time(),'captured_frames':captures,'session':str(session),'error':error})
        print(f'[RVIZ RECORD] Closed; {captures} captures saved to {out}',flush=True)

if __name__=='__main__':main()
