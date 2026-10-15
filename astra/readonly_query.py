"""Bounded remote inspection. No arbitrary commands, writes or control APIs."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOTS = (Path('/home/liberai/Liberai_Deploy_astra'),
         Path('/home/liberai/Desktop/colcon_ws'), Path('/home/liberai/camera_pose'),
         Path('/opt/ros/jazzy/share'))
DENIED = {'.ssh', '.aws', '.codex', '.git', '.env', 'node_modules', '__pycache__'}
SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.jsonl', '.urdf', '.xacro', '.xml',
            '.md', '.txt', '.sh', '.cpp', '.hpp', '.h', '.launch', '.rviz', '.log'}


def checked_path(value):
    if not isinstance(value, str) or not value or len(value)>2048:
        raise ValueError('invalid path')
    raw = Path(value)
    path = (raw if raw.is_absolute() else ROOTS[0]/raw).resolve(strict=True)
    roots = tuple(p.resolve() for p in ROOTS)
    if not any(path == root or path.is_relative_to(root) for root in roots):
        raise ValueError('path outside approved inspection roots')
    if any(part in DENIED or part.startswith('.env') for part in path.parts):
        raise ValueError('private path denied')
    return path


def integer(value, default, low, high):
    if value is None: return default
    if type(value) is not int or not low<=value<=high: raise ValueError('invalid range')
    return value


def text_file(path):
    if not path.is_file() or path.suffix.lower() not in SUFFIXES:
        raise ValueError('only supported text files may be read')
    if path.stat().st_size>8_000_000:
        raise ValueError('file too large; select a smaller source file')
    return path.read_text(encoding='utf-8',errors='replace')


def tf_query(target, source):
    # Do not spawn tf2_echo: it runs forever and its ROS launcher can leave a
    # child behind on timeout. This subscriber returns one fresh transform.
    import rclpy
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node=Node('astra_readonly_tf')
    try:
        buffer=Buffer(); listener=TransformListener(buffer,node)
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if buffer.can_transform(target,source,rclpy.time.Time()):
                tf=buffer.lookup_transform(target,source,rclpy.time.Time())
                xyz=tf.transform.translation; q=tf.transform.rotation
                return {'target':target,'source':source,'stamp':tf.header.stamp.sec+tf.header.stamp.nanosec/1e9,
                        'translation_m':[xyz.x,xyz.y,xyz.z],'quaternion_xyzw':[q.x,q.y,q.z,q.w]}
        raise TimeoutError('TF unavailable within 4 seconds')
    finally:
        node.destroy_node();rclpy.shutdown()


def ros_query(args):
    kind=args.get('kind')
    if kind == 'topics':
        command=['ros2','topic','list','--no-daemon','--spin-time','2']
    elif kind in ('topic_info','topic_sample'):
        topic=args.get('topic','')
        if not re.fullmatch(r'/[A-Za-z0-9_/]+',topic): raise ValueError('invalid topic')
        command=['ros2','topic','info',topic,'--verbose'] if kind=='topic_info' else ['ros2','topic','echo',topic,'--once']
    elif kind == 'tf':
        frames=[args.get('target',''),args.get('source','')]
        if not all(re.fullmatch(r'[A-Za-z][A-Za-z0-9_/]*',f) for f in frames): raise ValueError('invalid frame')
        return tf_query(*frames)
    else: raise ValueError('unsupported read-only ROS query')
    # This module is invoked with the existing ROS environment already sourced.
    try:
        r=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=4)
        return {'command':command,'returncode':r.returncode,'output':(r.stdout+r.stderr)[:24000]}
    except subprocess.TimeoutExpired as error:
        def decode(value): return value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value or ''
        return {'command':command,'timed_out':True,'output':(decode(error.stdout)+decode(error.stderr))[:24000]}


def handle(tool, args):
    if not isinstance(args,dict): raise ValueError('arguments must be an object')
    if tool == 'ros_query': return ros_query(args)
    if tool not in ('read_file','list_directory','search_text'): raise ValueError('unknown read-only tool')
    path=checked_path(args.get('path',''))
    if tool=='read_file':
        start=integer(args.get('start_line'),1,1,1_000_000)
        count=integer(args.get('line_count'),100,1,200)
        rows=text_file(path).splitlines()
        selected=[f'{i+1}: {rows[i]}' for i in range(start-1,min(len(rows),start-1+count))]
        text='\n'.join(selected)
        return {'path':str(path),'start_line':start,'total_lines':len(rows),'text':text[:30000], 'truncated':len(text)>30000}
    if not path.is_dir(): raise ValueError('directory required')
    if tool=='list_directory':
        entries=[]
        for child in sorted(path.iterdir()):
            try: resolved=checked_path(str(child))
            except (ValueError,OSError): continue
            entries.append({'name':child.name,'directory':resolved.is_dir()})
            if len(entries)>=200: break
        return {'path':str(path),'entries':entries,'limit':200}
    query=args.get('query','')
    if not isinstance(query,str) or not 1<=len(query)<=200: raise ValueError('invalid literal search')
    matches=[]; scanned=0; limited=False
    for directory, dirs, files in os.walk(path,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in DENIED and not d.startswith('.'))
        for name in sorted(files):
            scanned+=1
            if scanned>5000: limited=True;break
            try:
                file=checked_path(str(Path(directory)/name))
                if file.suffix.lower() not in SUFFIXES or file.stat().st_size>1_000_000:continue
                for number,line in enumerate(text_file(file).splitlines(),1):
                    if query.casefold() in line.casefold():
                        matches.append({'path':str(file),'line':number,'text':line[:500]})
                        if len(matches)>=40:limited=True;break
            except (OSError,ValueError):continue
            if limited:break
        if limited:break
    return {'matches':matches,'truncated':limited,'scanned_files':scanned}


def main():
    try:
        request=json.loads(sys.stdin.read(16001))
        result=handle(request['tool'],request.get('arguments',{}))
        print(json.dumps(result,ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'error':str(error)},ensure_ascii=False))
        return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
