"""CPU contracts and a tiny complete path with only the GPU backend replaced."""
import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tools.track3.r78 import fitting, generation, tracking, selection, shared_scale
from tools.track3.r78.io import Inputs, artifact, decode_depth, read, sha, write, freeze


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        p = self.root
        Image.fromarray(np.zeros((64,64,3), np.uint8)).save(p/'rgb.png')
        mask = np.zeros((64,64), np.uint8); mask[16:49,16:49] = 255
        Image.fromarray(mask).save(p/'mask.png')
        Image.fromarray(np.full((64,64),255,np.uint8)).save(p/'valid.png')
        np.save(p/'depth.npy', np.ones((64,64),np.float32))
        write(p/'K.json',dict(fx=100.,fy=100.,cx=32.,cy=32.))
        (p/'mesh.obj').write_text('v -0.16 -0.16 0\nv 0.16 -0.16 0\nv 0.16 0.16 0\nv -0.16 0.16 0\nf 1 2 3\nf 1 3 4\n')
        self.d = dict(schema_version=1,source_revision='fixture',source_code_commit='fixture',
            frame_grid='OpenCV 64x64',depth_encoding='float_m_npy',depth_source='frozen_stereo',
            episode=0,object='test_object',mesh='mesh.obj',mesh_scale_baked=True,
            pose_convention='original_mesh_to_camera',
            structure_review=dict(state='PASS',GT_used=False,mesh_sha256=sha(p/'mesh.obj'),
                critical_features={'surface':'present','neighbor':'absent'},critical_missing_or_extraneous_count=0),
            frames=[dict(id=i,role='support' if i in [0,2,4] else 'heldout',rgb='rgb.png',
                depth='depth.npy',mask='mask.png',validity='valid.png',K='K.json') for i in range(5)],
            sha256={f:sha(p/f) for f in ['mesh.obj','rgb.png','depth.npy','mask.png','valid.png','K.json']})
        self.manifest = p/'inputs.json';write(self.manifest,self.d)

    def seed(self):
        calls=[]
        class Backend:
            def __init__(self, mesh, weights):
                self.mesh = mesh
            def step(self,f,register):
                calls.append((f['id'],register));T=np.eye(4);T[2,3]=1.;return T
        tracking.run(self.manifest,'unused',self.root/'seed',backend_factory=Backend)
        self.assertEqual(calls,[(i,True) for i in range(5)])
        return Backend

    def test_complete_cpu_fit_select_and_normal_fp_calls(self):
        backend=self.seed()
        d=fitting.run(self.root/'seed/fit_manifest.json',self.root/'fit')
        self.assertEqual(d['state'],'FIT_BOUNDED')
        self.assertTrue(d['world_scale_unchanged'])
        self.assertEqual(read(self.root/'fit/heldout_pose_trace.json')['fixed_scale'],d['uniform_scale_multiplier'])
        write(self.root/'pool.json',dict(GT_used=False,frozen_before_evaluation=True,candidates=[dict(id='C1',fit='fit')]))
        selection.run(self.root/'pool.json',self.root/'selected')
        r=tracking.run(self.manifest,'unused',self.root/'track',selection=self.root/'selected/artifact.json',backend_factory=backend)
        self.assertEqual([f['operation'] for f in r['frames']],['register']+['track']*4)
        self.assertTrue(r['FP_entry']['skip_object_scale_estimation'])
        self.assertEqual(r['mesh_sha256'],d['mesh_sha256'])
        self.assertEqual(len(list((self.root/'track/poses').glob('*.json'))),5)
        with self.assertRaises(FileExistsError):
            tracking.run(self.manifest,'unused',self.root/'track',selection=self.root/'selected/artifact.json',backend_factory=backend)
        (self.root/'fit/mesh.obj').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            selection.selected_asset(self.root/'selected/artifact.json')

    def test_missing_structure_and_changed_inputs_fail_before_gpu(self):
        d=copy.deepcopy(self.d);d.pop('structure_review');write(self.manifest,d)
        with self.assertRaisesRegex(ValueError,'structural PASS'):
            tracking.run(self.manifest,'unused',self.root/'seed')
        write(self.manifest,self.d);(self.root/'depth.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            tracking.run(self.manifest,'unused',self.root/'seed')

    def test_new_mesh_with_old_poses_rejected(self):
        self.seed();p=self.root/'seed/fit_manifest.json';d=read(p)
        changed=self.root/'other.obj';changed.write_text((self.root/'mesh.obj').read_text().replace('0.16','0.17'))
        d['mesh']=str(changed);d['sha256'][str(changed)]=sha(changed)
        d['structure_review']['mesh_sha256']=sha(changed);write(p,d)
        with self.assertRaisesRegex(ValueError,'Seed poses and mesh'):
            fitting.run(p,self.root/'bad_fit')
        self.assertEqual(read(self.root/'bad_fit/failure.json')['state'],'FAILED')
        self.assertFalse((self.root/'bad_fit/artifact.json').exists())

    def test_heldout_split_and_pixel_grid_mismatch(self):
        self.seed();p=self.root/'seed/fit_manifest.json';d=read(p);d['frames'][0]['role']='heldout';write(p,d)
        with self.assertRaisesRegex(ValueError,'three support'):
            fitting.run(p,self.root/'bad_split')
        Image.fromarray(np.zeros((32,32),np.uint8)).save(self.root/'mask.png')
        self.d['sha256']['mask.png']=sha(self.root/'mask.png');write(self.manifest,self.d)
        with self.assertRaisesRegex(ValueError,'pixel grids'):
            Inputs(self.manifest).frame(self.d['frames'][0])

    def test_prepared_pointmap_masks_invalid_support(self):
        d=copy.deepcopy(self.d);d['frames']=d['frames'][:1]
        d['candidate_views']=[dict(frame_id=0,clear_visible_structure=True,no_neighbor_contamination=True,mask_review_pass=True)]
        write(self.manifest,d)
        generation.prepare(self.manifest,1,self.root/'prepared')
        points=np.load(self.root/'prepared/pointmap.npy')
        self.assertTrue(np.isnan(points[0,0]).all())
        np.testing.assert_allclose(points[32,32],[0,0,1])
        self.assertEqual(artifact(self.root/'prepared/artifact.json',{'GENERATION_INPUT_READY'})['seed'],0)

    def test_generation_delegates_once_and_requires_separate_structure_review(self):
        d=copy.deepcopy(self.d);d['frames']=d['frames'][:1]
        d['candidate_views']=[dict(frame_id=0,clear_visible_structure=True,no_neighbor_contamination=True,mask_review_pass=True)]
        write(self.manifest,d);generation.prepare(self.manifest,1,self.root/'prepared')
        calls=[]
        def infer(rgb,mask,mesh,transform,K,weights,**kw):
            calls.append(kw)
            Path(mesh).write_text((self.root/'mesh.obj').read_text())
            write(transform,dict(rotation=[1.,0.,0.,0.],translation=[0.,0.,1.],scale=[1.,1.,1.]))
            write(K,dict(fx=100.,fy=100.,cx=32.,cy=32.,width=64,height=64))
        module=types.ModuleType('v2d.sam3d.lib.image_to_mesh');module.image_to_mesh=infer
        with patch.dict('sys.modules',{'v2d.sam3d.lib.image_to_mesh':module}):
            generation.run(self.root/'prepared','unused',self.root/'candidate')
        self.assertEqual(len(calls),1);self.assertEqual(calls[0]['seed'],0)
        self.assertTrue(Path(calls[0]['pointmap_path']).is_file())
        review=read(self.root/'candidate/structure_review.template.json')
        self.assertEqual(review['state'],'PENDING')
        self.assertEqual(review['mesh_sha256'],sha(self.root/'candidate/mesh.obj'))

    def test_bounds_fail_without_exporting_accepted_mesh(self):
        self.seed()
        result=dict(state='BOUND_REACHED',scale=1.,poses=[],trace=[],fixed_scale=None)
        with patch.object(fitting,'fit',return_value=result):
            with self.assertRaisesRegex(ValueError,'SCALE_INCOMPLETE'):
                fitting.run(self.root/'seed/fit_manifest.json',self.root/'bounded')
        self.assertFalse((self.root/'bounded/artifact.json').exists())
        self.assertFalse((self.root/'bounded/mesh.obj').exists())

    def test_freeze_relocates_paths_and_refuses_changed_snapshot(self):
        target=self.root/'nested/frozen.json'
        freeze(self.manifest,target)
        self.assertEqual(Inputs(target).frame(read(target)['frames'][0])['id'],0)
        with self.assertRaises(FileExistsError):freeze(self.manifest,target)
        (self.root/'depth.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Refusing to refresh'):
            freeze(self.manifest,self.root/'new.json')

    def test_missing_candidate_is_not_silently_treated_as_failed(self):
        write(self.root/'pool.json',dict(GT_used=False,frozen_before_evaluation=True,candidates=[dict(id='C1',fit='missing')]))
        with self.assertRaisesRegex(ValueError,'missing completion or failure'):
            selection.run(self.root/'pool.json',self.root/'selected')


class GeometryTests(unittest.TestCase):
    def test_shared_scale_independent_se3_and_fixed_holdout_scale(self):
        rng=np.random.default_rng(0);X=rng.normal(size=(300,3))*.07
        rows=[];truth=[]
        for i in range(3):
            R=Rotation.from_rotvec([.1*i,.2*i,-.08*i]).as_matrix();t=np.array([.2*i,-.1*i,1.])
            rows.append((X,1.23*X@R.T+t,np.ones(len(X))));truth.append((R,t))
        s,poses=shared_scale.solve_corresponded_shared_scale(rows)
        self.assertAlmostEqual(s,1.23,places=12)
        for T,(R,t) in zip(poses,truth):
            np.testing.assert_allclose(T[:3,:3],R,atol=1e-12);np.testing.assert_allclose(T[:3,3],t,atol=1e-12)
        s,_=shared_scale.solve_corresponded_shared_scale(rows,fixed_scale=.9);self.assertEqual(s,.9)

    def test_exported_mesh_pose_pair_nonzero_center(self):
        center=np.array([.2,-.3,.1]);v=np.array([[.1,.2,.3],[-.2,.1,.4]])
        T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([.3,-.7,.2]).as_matrix();T[:3,3]=[.1,.2,1.]
        s=1.3;public=shared_scale.public_pose(T,center);metric=center+s*(v-center)
        np.testing.assert_allclose(metric@public[:3,:3].T+public[:3,3],s*(v-center)@T[:3,:3].T+T[:3,3])
        np.testing.assert_allclose(shared_scale.centered_pose(public,center),T)

    def test_invalid_inverse_depth_codes_do_not_become_far_geometry(self):
        z=decode_depth(np.array([[0,65535,32768]],np.uint16),'v2d_inverse_uint16')
        self.assertTrue(np.isnan(z[0,:2]).all());self.assertAlmostEqual(z[0,2],65535/32768-1)
        with self.assertRaises(ValueError):decode_depth(np.ones((2,2),np.uint8),'float_m_npy')

    def test_heldout_conflict_uses_frozen_order_not_single_residual(self):
        def row(name,iou,depth,coverage):
            return dict(id=name,heldout=[dict(candidate=dict(mask_iou=iou,median_surface_distance_m=depth,valid_observed_mask_coverage=coverage))]*2)
        a=row('C1',.9,.006,.85);b=row('C2',.85,.004,.9)
        self.assertEqual(selection.choose([a,b])[0]['id'],'C1')
        b=row('C2',.95,.004,.95);self.assertEqual(selection.choose([a,b])[0]['id'],'C2')

if __name__=='__main__':unittest.main()
