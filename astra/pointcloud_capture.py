"""Freeze scene and historical TF alongside the robot anchor, before disk/render work."""
from collections import deque
import threading
import xml.etree.ElementTree as ET
import numpy as np


def choose_cloud(clouds, anchor, now):
    valid=[(stamp,msg) for stamp,msg in clouds if -.1<=now-stamp<=2 and abs(stamp-anchor)<=2]
    if not valid:
        latest=max((stamp for stamp,_ in clouds),default=None)
        raise RuntimeError(f'No fresh aligned scene cloud: now={now:.6f}, anchor={anchor:.6f}, latest={latest}; require age<=2s and anchor gap<=2s')
    return min(valid,key=lambda item:abs(item[0]-anchor))


class PointcloudCapture:
    def __init__(self, robot):
        from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
        from std_msgs.msg import String
        from sensor_msgs.msg import PointCloud2
        self.robot=robot;self.node=robot.env._node;self.lock=threading.Lock();self.clouds=deque(maxlen=16);self.xml=None
        self.subs=[self.node.create_subscription(String,'/robot_description',self.description,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)),
                   self.node.create_subscription(PointCloud2,'/leo/d435i/chest_points',self.cloud,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))]

    def description(self,msg):
        with self.lock:self.xml=msg.data

    def cloud(self,msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        with self.lock:self.clouds.append((stamp,msg))

    def freeze(self,anchor):
        from rclpy.time import Time
        from rclpy.duration import Duration
        from scipy.spatial.transform import Rotation
        from sensor_msgs_py.point_cloud2 import read_points_numpy
        now=self.node.get_clock().now().nanoseconds*1e-9
        with self.lock:
            xml=self.xml;stamp,msg=choose_cloud(self.clouds,anchor,now)
        if not xml:raise RuntimeError('No robot_description for V6')
        if msg.header.frame_id!='chest_link':raise ValueError('scene cloud must use chest_link')
        names=sorted(l.get('name') for l in ET.fromstring(xml).findall('link') if l.get('name').startswith(('left_hand_','right_hand_')))
        transforms=[]
        for name in names:
            tf=self.robot.env._tf_buffer.lookup_transform('chest_link',name,Time(seconds=anchor),timeout=Duration(seconds=.3))
            r=tf.transform.rotation;p=tf.transform.translation;t=np.eye(4)
            t[:3,:3]=Rotation.from_quat([r.x,r.y,r.z,r.w]).as_matrix();t[:3,3]=[p.x,p.y,p.z];transforms.append(t)
        xyz=read_points_numpy(msg,field_names=['x','y','z'],skip_nans=False).reshape(-1,3)
        finite=np.isfinite(xyz).all(1);xyz=xyz[finite]
        if len(xyz)<100:raise ValueError('scene cloud empty')
        values=read_points_numpy(msg,field_names=['rgb'],skip_nans=False).reshape(-1)
        bits=values.view(np.uint32) if values.dtype.kind=='f' else values.astype(np.uint32)
        bits=bits[finite];rgb=np.stack([(bits>>16)&255,(bits>>8)&255,bits&255],axis=1).astype(np.uint8)
        return {'scene':xyz.copy(),'scene_rgb':rgb,'scene_timestamp':stamp,'captured_at':now,
                'anchor_timestamp':anchor,'tf_names':np.asarray(names),'tf_matrices':np.asarray(transforms),'xml':xml}


def validate_frozen(data,anchor):
    stamp=float(data['scene_timestamp']);captured=float(data['captured_at'])
    if abs(float(data['anchor_timestamp'])-anchor)>1e-6:raise ValueError('frozen scene anchor mismatch')
    if not -.1<=captured-stamp<=2 or abs(stamp-anchor)>2:raise ValueError('frozen scene was stale at capture')
    if not np.isfinite(data['scene']).all() or not np.isfinite(data['tf_matrices']).all():raise ValueError('nonfinite frozen geometry')
