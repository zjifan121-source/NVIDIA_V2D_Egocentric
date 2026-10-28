"""Photometric/depth check of spoon thickness; no scans or ground truth used."""
from pathlib import Path
import sys,json,numpy as np,shutil
P=Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28');REPO=P.parents[1];sys.path.insert(0,str(REPO/'tools/track3'));from fit_shape import Fit
S=Path('/data/user_data/edmondwa/v2d/ken_night/variants/ep33_B');raw=Path('/data/user_data/edmondwa/v2d/recon_data/outputs/fs_sam3d/episode_000033/wooden_spoon');proxy=P/'spoon_shape_proxy/ep33/wooden_spoon';proxy.mkdir(parents=True,exist_ok=True)
for name in ['frames','masks','depth']:
 dest=proxy/name
 if dest.is_symlink():dest.unlink()
 if not dest.exists():dest.symlink_to(raw/name,target_is_directory=True)
b=proxy/'result_slam_gravity_aligned';b.mkdir(exist_ok=True);d=np.load(S/'bundle_merged/result.npz');C=d['camera_to_world_transform'].astype(float);O=d['object_1_to_world_transform'].astype(float);oc=np.linalg.inv(C)@O;np.savez(b/'result.npz',object_to_camera_transform=oc,camera_intrinsics=d['camera_intrinsics'],object_scale=d['object_1_scale']);shutil.copyfile(S/'bundle_merged/mesh_1.obj',b/'mesh.obj');(proxy/'object_track.json').write_text(json.dumps(dict(object_id=1)))
import trimesh
# This retarget environment has trimesh3; preserve the deterministic sampling
# behavior of the newer fit tool without changing the shared library on disk.
old_sample=trimesh.Trimesh.sample
rng_state=np.random.get_state();np.random.seed(0)
trimesh.Trimesh.sample=lambda mesh,count,seed=None: old_sample(mesh,count)
try:F=Fit(proxy)
finally:trimesh.Trimesh.sample=old_sample;np.random.set_state(rng_state)
rows=[]
for scale in [.45,.55,.7,.85,1.,1.15]:
 x=np.zeros(9);x[2]=np.log(scale);costs=F.costs(x);keep=max(1,int(.7*len(costs)));row=dict(thickness_scale=scale,trimmed_cost=float(np.sort(costs)[:keep].mean()),mean_cost=float(costs.mean()),frame_costs=costs.tolist());rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='frame_costs'}),flush=True)
(P/'spoon_shape_thickness_check.json').write_text(json.dumps(dict(episode=33,frames=F.frames.tolist(),object_camera_pose_source=str(S/'bundle_merged/result.npz'),rgb_depth_masks_source=str(raw),pca_axes=F.P.tolist(),mesh_center=F.c.tolist(),rows=rows),indent=2)+'\n')
