"""V6 display-only ROS worker. Publishes ONLY a dedicated PointCloud2 topic."""
import argparse,importlib.util,json,time,hashlib,os
from pathlib import Path
import numpy as np
from .pointcloud_geometry import HandKinematics,render_reviews
from .session import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--overlay',type=Path,default=Path('/home/liberai/camera_pose/rviz_overlay'))
    args=p.parse_args();req=json.loads(args.request.read_text());directory=args.request.parent
    import rclpy
    from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2,PointField
    from .pointcloud_capture import validate_frozen
    rclpy.init();node=rclpy.create_node('astra_v6_imagined_hands')
    pub=node.create_publisher(PointCloud2,'/astra/imagination/hand_points',QoSProfile(depth=1))
    try:
        with np.load(directory/'frozen_scene.npz',allow_pickle=False) as data:
            validate_frozen(data,req['anchor_timestamp'])
            xyz=data['scene'];rgb=data['scene_rgb'];stamp=float(data['scene_timestamp'])
            captured_at=float(data['captured_at']);xml=str(data['xml'])
            tf_matrices=dict(zip(data['tf_names'].tolist(),data['tf_matrices']))
        source=args.overlay/'publish_hand_model_cloud.py'
        spec=importlib.util.spec_from_file_location('existing_hand_sampler',source);sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampler)
        groups=sampler.sample_hand_visuals(xml,density=120000)
        fk=HandKinematics(xml,groups);anchor=np.asarray(req['anchor'],float)
        transforms=fk.transforms(anchor)
        # Compare against TF frozen at the same anchor, never against later live TF.
        tf_errors=[]
        probes=np.array([[0,0,0],[.03,0,0],[0,.03,0],[0,0,.03]])
        for name,pts in groups:
            expected=tf_matrices[name]
            measured=probes@expected[:3,:3].T+expected[:3,3]
            calculated=probes@transforms[name][:3,:3].T+transforms[name][:3,3]
            tf_errors.append(float(np.linalg.norm(calculated-measured,axis=1).max()))
        if max(tf_errors)>.015:raise ValueError('URDF FK disagrees with anchor TF by > 15 mm')
        real=fk.points(anchor);actions=np.asarray(req['actions'],float)
        frames=np.stack([fk.points(a) for a in actions])
        np.savez_compressed(directory/'geometry.npz',scene=xyz,scene_rgb=rgb,anchor_points=real,candidate_points=frames,actions=actions)
        (directory/'robot.urdf').write_text(xml)
        images=render_reviews(xyz,rgb,real,frames,directory)
        if req.get('prefix_steps'):
            prefix_dir=directory/'prefix';prefix_dir.mkdir()
            prefix_images=render_reviews(xyz,rgb,real,frames[:req['prefix_steps']],prefix_dir)
            for f in prefix_images:
                target=directory/('prefix_'+f.name);f.replace(target);images.append(target)
        dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')]
        def publish(points):
            packed=np.empty(len(points),dtype=dtype)
            for i,k in enumerate(('x','y','z')):packed[k]=points[:,i]
            packed['rgb']=0xFFA51E
            out=PointCloud2();out.header.frame_id='chest_link';out.header.stamp=node.get_clock().now().to_msg()
            out.height=1;out.width=len(points);out.fields=[PointField(name=k,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,k in enumerate(packed.dtype.names)]
            out.point_step=16;out.row_step=16*len(points);out.is_dense=True;out.data=packed.tobytes();pub.publish(out)
        # Animate once using the actual candidate schedule. No joint_states, TF or action topics are published.
        publish(real);time.sleep(req['lead_time'])
        for frame in frames:
            if (directory/'cancel').exists():return
            time.sleep(1/req['fps']);publish(frame);rclpy.spin_once(node,timeout_sec=0)
        info={'candidate_id':req['candidate_id'],'candidate_sha256':req['candidate_sha256'],
              'backend':'urdf_pointcloud','frame':'chest_link','scene_timestamp':stamp,'scene_captured_at':captured_at,'tf_timestamp':req['anchor_timestamp'],
              'scene_stamp_semantics':'camera host receipt time, not hardware synchronized',
              'max_fk_tf_error_m':max(tf_errors),'sampler_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'images':[{'path':str(f.resolve()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in images],
              'static_scene':True,'object_dynamics_predicted':False,'collision_free_certified':False}
        atomic_json(directory/'result.json',info)
        until=time.monotonic()+650
        while time.monotonic()<until and not (directory/'cancel').exists() and os.getppid()==req['parent_pid']:
            publish(frames[-1]);rclpy.spin_once(node,timeout_sec=.1)
    finally:
        # Remove the ghost when the candidate is invalidated/consumed.
        try:
            empty=PointCloud2();empty.header.frame_id='chest_link';empty.height=1;pub.publish(empty)
        finally:node.destroy_node();rclpy.shutdown()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        print('V6 pointcloud failed: '+str(exc),file=sys.stderr,flush=True)
        raise
