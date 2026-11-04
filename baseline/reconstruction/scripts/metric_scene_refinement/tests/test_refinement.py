"""CPU contracts for paired geometry, observations, and fixed refinement inputs."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bundle_files, fields, freeze, load_frozen, sha, validate_spec
from geometry import projection, skew
from renderer import raster_surface
from part_shape import build_basis


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        b = self.root / 'bundle'; b.mkdir()
        self.b = b
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=.05)
        mesh.export(b / 'mesh.obj')
        n = 4
        pose = np.tile(np.eye(4), (n,1,1)); pose[:,2,3] = .5
        camera = np.tile(np.eye(4), (n,1,1))
        np.savez(b/'result.npz', object_to_camera_transform=pose,
                 object_to_world_transform=pose, camera_to_world_transform=camera,
                 object_is_valid=np.ones(n,bool), camera_is_valid=np.ones(n,bool),
                 object_scale=1.)
        obs = self.root/'observations';obs.mkdir();self.obs=obs
        (obs/'K_fp.json').write_text(json.dumps(dict(fx=50,fy=50,cx=8,cy=8,width=16,height=16)))
        for sub in ['rgb','depth','masks']:(obs/sub).mkdir()
        for t in range(n):
            cv2.imwrite(str(obs/'rgb'/f'{t:06d}.png'),np.zeros((16,16,3),np.uint8))
            cv2.imwrite(str(obs/'depth'/f'{t:06d}.png'),np.full((16,16),43690,np.uint16))
            cv2.imwrite(str(obs/'masks'/f'{t:06d}.png'),np.ones((16,16),np.uint8))
        marks=self.root/'marks.json';marks.write_text(json.dumps(dict(parts=[0,1],center=2)))
        self.spec=dict(kind='part_shape',object_slot=0,
            coordinate_convention='T_camera_from_original_mesh_metres',
            structure_review=dict(accepted=True,evidence='Input structure review'),
            mesh_pose_pair=dict(mesh_sha256=sha(b/'mesh.obj'),result_sha256=sha(b/'result.npz')),
            inputs=dict(bundle=str(b),observations=str(obs),semantic_vertices=str(marks)),
            parameters=dict(support=[0,2],heldout=[1,3],maximum_iterations=15,maximum_seconds=60))

    def test_freeze_detects_input_change_and_refuses_overwrite(self):
        f=self.root/'spec.json';f.write_text(json.dumps(self.spec));out=self.root/'frozen.json'
        freeze(f,out);self.assertEqual(load_frozen(out)['frame_count'],4)
        with self.assertRaises(ValueError):freeze(f,out)
        (self.obs/'masks/000001.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Input changed'):load_frozen(out)

    def test_structure_review_and_mesh_pose_pair_required(self):
        for field,value in [('structure_review',dict(accepted=False,evidence='')),('mesh_pose_pair',dict(mesh_sha256='old',result_sha256='old'))]:
            s=copy.deepcopy(self.spec);s[field]=value
            with self.assertRaises(ValueError):validate_spec(s)

    def test_grid_frames_and_support_split_rejected(self):
        s=copy.deepcopy(self.spec);s['parameters']['heldout']=[0,3]
        with self.assertRaisesRegex(ValueError,'Disjoint'):validate_spec(s)
        cv2.imwrite(str(self.obs/'depth/000000.png'),np.zeros((8,8),np.uint16))
        with self.assertRaisesRegex(ValueError,'grid'):validate_spec(self.spec)

    def test_incomplete_camera_and_nonrigid_pose_rejected(self):
        z=dict(np.load(self.b/'result.npz'));z['camera_is_valid'][1]=False
        np.savez(self.b/'result.npz',**z)
        with self.assertRaisesRegex(ValueError,'incomplete'):validate_spec(self.spec)
        z['camera_is_valid'][1]=True;z['object_to_camera_transform'][0,0,0]=2
        np.savez(self.b/'result.npz',**z)
        with self.assertRaisesRegex(ValueError,'SE\(3\)'):validate_spec(self.spec)

    def test_no_unpinned_material_dependencies(self):
        with (self.b/'mesh.obj').open('a') as f:f.write('\nmtllib missing.mtl\n')
        with self.assertRaisesRegex(ValueError,'dependency'):bundle_files(self.b)

    def test_unknown_parameters_and_fractional_slots_rejected(self):
        for edit in [lambda s:s['parameters'].update(smooth_weight=99),lambda s:s.update(object_slot=0.3)]:
            s=copy.deepcopy(self.spec);edit(s)
            with self.assertRaises(ValueError):validate_spec(s)

    def temporal(self):
        s=copy.deepcopy(self.spec);s['kind']='temporal_pose';s['inputs'].pop('semantic_vertices')
        for key,filename in [('table_plane','plane.npy'),('landmarks','points.npz'),('measurement_receipt','measurement.json'),('calibration','calibration.json')]:
            s['inputs'][key]=str(self.root/filename)
        np.save(self.root/'plane.npy',[0.,0.,1.,0.])
        np.savez(self.root/'points.npz',frame_ids=np.arange(4),points_camera=np.ones((4,2,3)),qualified=np.ones((4,2),bool),sigma_z=np.ones((4,2))*.01)
        (self.root/'measurement.json').write_text(json.dumps(dict(state='OBSERVABILITY_PASS')))
        (self.root/'calibration.json').write_text(json.dumps(dict(contact_point_local_m=[0,0,0])))
        s['parameters']=dict(calibration_frames=[0,2],fit_frames=[0,2],validation_frames=[1,3],contact_frames=[0],landmark_groups=[0,1],maximum_seconds=60)
        return s

    def test_temporal_calibration_and_fit_do_not_consume_validation(self):
        s=self.temporal();validate_spec(s)
        s['parameters']['calibration_frames']=[0,1]
        with self.assertRaisesRegex(ValueError,'leakage'):validate_spec(s)

    def test_unqualified_or_insufficient_landmarks_rejected(self):
        s=self.temporal()
        np.savez(self.root/'points.npz',frame_ids=np.arange(4),points_camera=np.ones((4,2,3)),qualified=np.zeros((4,2),bool),sigma_z=np.ones((4,2))*.01)
        with self.assertRaisesRegex(ValueError,'two qualified'):validate_spec(s)


class GeometryTests(unittest.TestCase):
    def test_perspective_surface_barycentrics_are_metric(self):
        v=np.array([[-.2,-.2,1.],[.3,-.2,2.],[-.2,.3,1.5]])
        k=np.array([[40.,0,16],[0,40,16],[0,0,1.]])
        z,fi,bc=raster_surface(v,np.array([[0,1,2]]),k,(32,32))
        y,x=np.where(fi>=0);self.assertGreater(len(x),10)
        q=bc[y,x]@v;uv,_=projection(q,k)
        np.testing.assert_allclose(uv,np.c_[x,y],atol=1e-5)
        np.testing.assert_allclose(q[:,2],z[y,x],atol=1e-6)

    def test_pose_projection_derivatives(self):
        q=np.array([[.1,-.1,.6],[.02,.12,1.]])
        k=np.array([[500.,0,320],[0,490,240],[0,0,1.]])
        uv,j=projection(q,k);eps=1e-7
        for axis in range(3):
            e=np.eye(3)[axis]*eps
            np.testing.assert_allclose((projection(q+e,k)[0]-projection(q-e,k)[0])/(2*eps),j[:,:,axis],atol=1e-6)
            qp=q@Rotation.from_rotvec(e).as_matrix().T
            qm=q@Rotation.from_rotvec(-e).as_matrix().T
            np.testing.assert_allclose((qp-qm)/(2*eps),-skew(q)[:,:,axis],atol=1e-8)

    def test_part_basis_keeps_center_and_disconnected_anchors_fail(self):
        m=trimesh.creation.icosphere(subdivisions=2)
        z=build_basis(m,dict(parts=[0,1],center=30))
        np.testing.assert_array_equal(z['basis'][30],0)
        bad=trimesh.util.concatenate([m,m.copy().apply_translation([10,0,0])])
        with self.assertRaisesRegex(ValueError,'connect'):build_basis(bad,dict(parts=[0,len(m.vertices)],center=30))

if __name__=='__main__':unittest.main()
