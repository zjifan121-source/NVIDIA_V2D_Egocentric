"""Export the actual imported object collision geometry for episode30 diagnostics."""
from pathlib import Path
import json,sys,numpy as np,trimesh,pyarrow.parquet as pq,warp as wp
wp.config.kernel_cache_dir='/tmp/v2d_tool_collision_cache'
import newton
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[1]/'baseline/robotic_grounding/flash_chord/src'));from flash_chord.assets.mesh import convex_decompose
S=P/'ep30_table_full';t=pq.read_table(next(S.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet')));paths=t['object_urdf_paths'][0].as_py();arr=dict(pos=np.asarray(t['object_body_position'][0].as_py()),quat=np.asarray(t['object_body_wxyz'][0].as_py()));info=[]
for i,path in enumerate(paths):
 builder=newton.ModelBuilder();builder.add_urdf(path,floating=True,collapse_fixed_joints=True);sources=[builder.shape_source[j] for j in range(builder.shape_count) if int(builder.shape_flags[j])&int(newton.ShapeFlags.COLLIDE_SHAPES)];parts=[];meshes=[]
 for src in sources:
  v=np.asarray(src.vertices);f=np.asarray(src.indices).reshape(-1,3);meshes.append(trimesh.Trimesh(v,f,process=True));parts.extend(convex_decompose(v,f))
 merged=trimesh.util.concatenate(meshes);arr[f'obj{i}_v']=merged.vertices;arr[f'obj{i}_f']=merged.faces;arr[f'obj{i}_n']=np.int64(len(parts))
 for j,p in enumerate(parts):arr[f'obj{i}_h{j}_v']=p.vertices;arr[f'obj{i}_h{j}_f']=p.faces
 row=dict(object=i,urdf=path,hulls=len(parts),watertight=bool(merged.is_watertight));info.append(row);print(json.dumps(row),flush=True)
np.savez(P/'ep30_object_collision_geometry.npz',**arr);(P/'ep30_object_collision_geometry.json').write_text(json.dumps(info,indent=2)+'\n')
