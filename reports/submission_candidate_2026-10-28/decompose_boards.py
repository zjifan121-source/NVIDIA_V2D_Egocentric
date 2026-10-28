"""Test tighter convex collision geometry without changing visual/scoring meshes."""
import json,time
from pathlib import Path
import numpy as np,trimesh,coacd
P=Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28');S=P/'ep25_pinch_scene';coacd.set_log_level('error');reports=[]
for i in [1,2]:
 out=P/f'ep25_board{i}_tight_hulls.npz'
 if out.exists():continue
 mesh=trimesh.load(S/f'score_input/meshes/episode_000025/wooden_piece_{i}.obj',force='mesh',process=False);mesh.merge_vertices(merge_tex=True,merge_norm=True);mesh.process(validate=True);assert mesh.is_watertight
 start=time.time();parts=coacd.run_coacd(coacd.Mesh(np.ascontiguousarray(mesh.vertices,dtype=np.float64),np.ascontiguousarray(mesh.faces,dtype=np.int32)),threshold=.005,preprocess_mode='off',resolution=10000,mcts_iterations=150,seed=0)
 arr={'n':np.int64(len(parts))}
 for j,(v,f) in enumerate(parts):arr[f'v{j}']=v;arr[f'f{j}']=f
 tmp=out.with_suffix('.tmp.npz');np.savez(tmp,**arr);tmp.replace(out);r=dict(board=i,hulls=len(parts),wall_seconds=time.time()-start,threshold=.005,preprocess_mode='off',resolution=10000);reports.append(r);print(json.dumps(r),flush=True)
P.joinpath('ep25_tight_decomposition.json').write_text(json.dumps(reports,indent=2)+'\n')
