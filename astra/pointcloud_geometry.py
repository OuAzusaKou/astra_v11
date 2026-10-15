"""Pure URDF hand FK and review rendering. No ROS or controller publication."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw


def origin(element):
    t=np.eye(4)
    if element is not None:
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
        t[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
    return t


class HandKinematics:
    def __init__(self, xml, groups):
        self.groups=groups
        self.parents={j.find('child').get('link'):j for j in ET.fromstring(xml).findall('joint')}

    def transforms(self, action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(52,) or not np.isfinite(action).all():raise ValueError('expected finite 52D action')
        transforms={};q={}
        for side,off,hand in (('left',0,12),('right',6,32)):
            t=np.eye(4);t[:3,3]=action[off:off+3];t[:3,:3]=Rotation.from_rotvec(action[off+3:off+6]).as_matrix()
            transforms[side+'_link_ee']=t
            for i in range(20):q[f'{side}_hand_finger{i//4+1}_joint{i%4+1}']=action[hand+i]
        def resolve(link,seen=()):
            if link in transforms:return transforms[link]
            if link in seen or link not in self.parents:raise ValueError('hand link not rooted at action EE: '+link)
            j=self.parents[link];t=origin(j.find('origin'));kind=j.get('type')
            if kind!='fixed':
                if kind not in ('revolute','continuous','prismatic') or j.find('mimic') is not None:
                    raise ValueError('unsupported hand joint: '+j.get('name'))
                if j.get('name') not in q:raise ValueError('missing hand joint: '+j.get('name'))
                axis=j.find('axis');axis=np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ')
                norm=np.linalg.norm(axis)
                if norm<1e-8:raise ValueError('zero joint axis')
                axis=axis/norm;motion=np.eye(4)
                if kind=='prismatic':motion[:3,3]=axis*q[j.get('name')]
                else:motion[:3,:3]=Rotation.from_rotvec(axis*q[j.get('name')]).as_matrix()
                t=t@motion
            transforms[link]=resolve(j.find('parent').get('link'),seen+(link,))@t
            return transforms[link]
        return {name:resolve(name) for name,_ in self.groups}

    def points(self, action):
        transforms=self.transforms(action)
        return np.concatenate([pts@transforms[name][:3,:3].T+transforms[name][:3,3] for name,pts in self.groups])


PROJECTION_VIEWS = (('top_XY',(0,1)),('front_YZ',(1,2)),('side_XZ',(0,2)))


def projection_panel(scene, rgb, anchor, imagined, low, high, axes):
    """Shared orthographic raster for live RViz panels and Astra review sheets."""
    panel=np.full((460,480,3),[20,23,28],np.uint8)
    span=np.maximum(np.asarray(high)[list(axes)]-np.asarray(low)[list(axes)],1e-6)
    scale=min(430/span)
    def paint(points,colors):
        points=np.asarray(points)
        uv=np.rint((points[:,axes]-np.asarray(low)[list(axes)])*scale).astype(int)
        x=uv[:,0]+20;y=439-uv[:,1];ok=(x>=0)&(x<480)&(y>=0)&(y<460)
        colors=np.asarray(colors,np.uint8)
        panel[y[ok],x[ok]]=colors[ok] if colors.ndim==2 else colors
    paint(scene,(rgb.astype(float)*.55).astype(np.uint8))
    paint(anchor,[0,220,255]);paint(imagined,[255,165,30])
    return panel


def render_reviews(scene, rgb, anchor, frames, directory):
    """Same geometry as RViz, deterministic orthographic projections, not a screenshot."""
    sweep=np.concatenate([anchor,*frames]);low=sweep.min(0)-.18;high=sweep.max(0)+.18
    inside=((scene>=low)&(scene<=high)).all(1);scene=scene[inside];rgb=rgb[inside]
    if len(scene)>80000:
        step=int(np.ceil(len(scene)/80000));scene=scene[::step];rgb=rgb[::step]
    keys=sorted(set((0,len(frames)//2,len(frames)-1)));assets=[]
    for name,axes in PROJECTION_VIEWS:
        canvas=Image.new('RGB',(480*len(keys),540),(20,23,28));draw=ImageDraw.Draw(canvas)
        for col,k in enumerate(keys):
            panel=projection_panel(scene,rgb,anchor,frames[k],low,high,axes)
            canvas.paste(Image.fromarray(panel),(col*480,60))
            draw.text((col*480+12,8),f'{name} candidate step {k+1}/{len(frames)}',fill='white')
            draw.text((col*480+12,26),'CYAN: anchor | ORANGE: imagined hands',fill='white')
            draw.text((col*480+12,43),'STATIC scene; not object-motion prediction',fill='white')
        draw.text((10,522),f'chest_link metres | axes {name} | bounds {np.round(low,3)} to {np.round(high,3)}',fill='white')
        path=directory/(name+'.png');canvas.save(path);assets.append(path)
    return assets
