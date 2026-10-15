"""Display-only live chest-frame projections for RViz Image docks."""
import os,sys,time
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from astra.pointcloud_geometry import PROJECTION_VIEWS,projection_panel

EMPTY=np.empty((0,3),float)


def decode_cloud(msg):
    """Honor PointCloud2 field offsets, endian, point and row strides."""
    if msg.width*msg.height == 0:return EMPTY.copy(),np.empty((0,3),np.uint8)
    endian='>' if msg.is_bigendian else '<'
    fields={f.name:f for f in msg.fields}
    def column(name):
        f=fields[name]
        types={6:'u4',7:'f4',8:'f8'}
        if f.datatype not in types:raise ValueError('unsupported field '+name)
        dtype=endian+types[f.datatype]
        return np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,
                          offset=f.offset,strides=(msg.row_step,msg.point_step)).reshape(-1)
    xyz=np.stack([column(k) for k in ('x','y','z')],axis=1)
    key='rgb' if 'rgb' in fields else 'rgba' if 'rgba' in fields else None
    rgb=np.full(xyz.shape,150,np.uint8)
    if key:
        bits=column(key)
        if bits.dtype.kind=='f':bits=bits.view(endian+'u4')
        rgb=np.stack([(bits>>16)&255,(bits>>8)&255,bits&255],axis=1).astype(np.uint8)
    valid=np.isfinite(xyz).all(1);xyz=xyz[valid];rgb=rgb[valid]
    stride=max(1,int(np.ceil(len(xyz)/80000)))
    return xyz[::stride],rgb[::stride]


def crop_projection(panel, factor):
    """Center-crop the raster, keeping the RViz panel size and pixel colors."""
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('RVIZ_PROJECTION_CROP_FACTOR must be finite and >= 1')
    image=Image.fromarray(panel)
    width,height=image.size
    crop_width=max(1,int(width/factor));crop_height=max(1,int(height/factor))
    left=(width-crop_width)//2;top=(height-crop_height)//2
    return image.crop((left,top,left+crop_width,top+crop_height)).resize(
        (width,height),Image.Resampling.NEAREST)


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2,Image as ROSImage
    crop_factor=float(os.environ.get('RVIZ_PROJECTION_CROP_FACTOR','1'))
    if not np.isfinite(crop_factor) or crop_factor < 1:
        raise ValueError('RVIZ_PROJECTION_CROP_FACTOR must be finite and >= 1')
    rclpy.init();node=rclpy.create_node('astra_live_projections')
    node.get_logger().info(f'Projection center crop: {crop_factor:g}x')
    cache={};bounds=None
    pubs={name:node.create_publisher(ROSImage,'/astra/projections/'+name,qos_profile_sensor_data)
          for name,_ in PROJECTION_VIEWS}
    def receive(key,msg):
        if msg.header.frame_id.lstrip('/')!='chest_link':
            cache.pop(key,None);return
        try:cache[key]=(*decode_cloud(msg),time.monotonic())
        except (ValueError,KeyError,TypeError) as error:
            cache.pop(key,None);node.get_logger().warning(str(error))
    subscriptions=[]
    for key,topic in [('scene','/leo/d435i/chest_points'),('real','/leo/wuji/model_points'),
                      ('ghost','/astra/imagination/hand_points')]:
        subscriptions.append(node.create_subscription(PointCloud2,topic,lambda msg,k=key:receive(k,msg),qos_profile_sensor_data))
    def draw():
        nonlocal bounds
        now=time.monotonic()
        def fresh(key):
            item=cache.get(key)
            return item[:2] if item is not None and now-item[2]<2 else (EMPTY,np.empty((0,3),np.uint8))
        scene,rgb=fresh('scene');real,_=fresh('real');ghost,_=fresh('ghost')
        hands=np.concatenate([real,ghost])
        if len(hands):
            low=hands.min(0)-.18;high=hands.max(0)+.18
            # Retain framing while the ghost animates; reset after the candidate clears.
            bounds=(np.minimum(bounds[0],low),np.maximum(bounds[1],high)) if bounds is not None and len(ghost) else (low,high)
        low,high=bounds if bounds is not None else (np.array([-.4,-.7,-.6]),np.array([.8,.7,.6]))
        inside=((scene>=low)&(scene<=high)).all(1)
        for name,axes in PROJECTION_VIEWS:
            image=Image.new('RGB',(480,540),(20,23,28));d=ImageDraw.Draw(image)
            panel=projection_panel(scene[inside],rgb[inside],real,ghost,low,high,axes)
            image.paste(crop_projection(panel,crop_factor),(0,60))
            x,y=('XYZ'[a] for a in axes)
            d.text((10,8),f'{name} | horizontal +{x}, vertical +{y}',fill='white')
            d.text((10,25),'CYAN: live hands | ORANGE: imagination',fill='white')
            missing=[k for k,p in [('scene',scene),('hands',real)] if not len(p)]
            status='STALE / WAITING: '+', '.join(missing) if missing else 'LIVE chest_link | orthographic, no perspective'
            d.text((10,42),status,fill='#ffba55' if missing else 'white')
            d.text((10,523),'DISPLAY ONLY | '+('preview active' if len(ghost) else 'no preview'),fill='white')
            msg=ROSImage();msg.header.stamp=node.get_clock().now().to_msg();msg.header.frame_id='chest_link'
            msg.height=540;msg.width=480;msg.encoding='rgb8';msg.step=480*3;msg.data=np.asarray(image).tobytes()
            pubs[name].publish(msg)
    node.create_timer(.2,draw)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
