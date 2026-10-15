import unittest
from types import SimpleNamespace as NS
import numpy as np
from astra.live_projections import decode_cloud
from astra.pointcloud_geometry import projection_panel,PROJECTION_VIEWS

class ProjectionTests(unittest.TestCase):
    def test_padded_rows_and_color(self):
        for big in (False,True):
            endian='>' if big else '<';data=bytearray(48)
            for offset in (0,24):
                data[offset:offset+12]=np.array([.1,.2,.3],dtype=endian+'f4').tobytes()
                data[offset+12:offset+16]=np.array([0x12ABEF],dtype=endian+'u4').tobytes()
            msg=NS(width=1,height=2,is_bigendian=big,point_step=16,row_step=24,data=data,
                   fields=[NS(name=n,offset=i*4,datatype=7) for i,n in enumerate(['x','y','z','rgb'])])
            xyz,rgb=decode_cloud(msg)
            np.testing.assert_allclose(xyz,[[.1,.2,.3]]*2)
            np.testing.assert_array_equal(rgb,[[18,171,239]]*2)
    def test_positive_axes_and_ghost_overlays_real(self):
        empty=np.empty((0,3));point=np.array([[.2,.3,.4]])
        for _,axes in PROJECTION_VIEWS:
            panel=projection_panel(empty,empty,point,point,np.zeros(3),np.ones(3),axes)
            x,y=np.rint(point[0,list(axes)]*430).astype(int)
            np.testing.assert_array_equal(panel[439-y,20+x],[255,165,30])
    def test_empty_clears_ghost(self):
        xyz,rgb=decode_cloud(NS(width=0,height=1))
        self.assertEqual(xyz.shape,(0,3));self.assertEqual(rgb.shape,(0,3))

    def test_optional_display_crop(self):
        from astra.live_projections import crop_projection
        panel=np.arange(60,dtype=np.uint8).reshape(4,5,3)
        np.testing.assert_array_equal(np.asarray(crop_projection(panel,1)),panel)
        self.assertEqual(crop_projection(panel,2).size,(5,4))
        for factor in (0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):crop_projection(panel,factor)
