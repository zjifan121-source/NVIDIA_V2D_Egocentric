"""CPU tests of contact propagation; these do not execute FoundationPose."""
import unittest
import numpy as np
from contact import project_contact, apply_to_tracker

class Pose:
    def __init__(self,T):self.T=np.array(T,copy=True)
    def to_matrix(self):return self.T.copy()

class Tracker:
    def __init__(self):self.resets=[]
    def reset_to_pose(self,p):self.resets.append(p.to_matrix())

class ContactTests(unittest.TestCase):
    def test_world_normal_correction_preserves_rotation_and_geometry(self):
        c=np.eye(4);c[:3,:3]=[[0,0,1],[1,0,0],[0,1,0]];c[:3,3]=[.1,-.2,.03]
        t=np.eye(4);point=np.array([.01,.01,.02]);plane=np.array([0,0,1,0])
        out,info=project_contact(t,c,point,plane,0,[0],.06)
        np.testing.assert_array_equal(out[:3,:3],t[:3,:3])
        w=c@out;self.assertAlmostEqual(float((w[:3,:3]@point+w[:3,3])@plane[:3]),0)
        np.testing.assert_array_equal(t,np.eye(4));self.assertTrue(info['contact_active'])

    def test_active_correction_resets_tracker_but_noncontact_does_not(self):
        t=np.eye(4);t[2,3]=.02;p=Pose(t);tr=Tracker()
        cfg=dict(camera_to_world_file=np.tile(np.eye(4),(2,1,1)),point_file=np.zeros(3),plane_file=np.array([0,0,1,0]),rest_frames=[0],maximum_shift_m=.06)
        out,raw,info=apply_to_tracker(tr,p,cfg,0,Pose)
        self.assertEqual(len(tr.resets),1);self.assertAlmostEqual(tr.resets[0][2,3],0)
        np.testing.assert_array_equal(raw,t)
        out,raw,info=apply_to_tracker(tr,p,cfg,1,Pose)
        self.assertEqual(len(tr.resets),1);np.testing.assert_array_equal(out.to_matrix(),t)

    def test_contact_bound_fails_instead_of_clipping(self):
        t=np.eye(4);t[2,3]=.2
        with self.assertRaisesRegex(ValueError,'CONTACT_BOUND'):
            project_contact(t,np.eye(4),np.zeros(3),[0,0,1,0],1,[1],.06)
        out,_=project_contact(t,np.eye(4),np.zeros(3),[0,0,1,0],1,[0],.06)
        np.testing.assert_array_equal(out,t)

if __name__=='__main__':unittest.main()
