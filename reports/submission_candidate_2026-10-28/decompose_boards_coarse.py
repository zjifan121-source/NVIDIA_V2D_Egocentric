from pathlib import Path
import json,time,numpy as np,trimesh,coacd
P=Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28');S=P/'ep25_pinch_scene';coacd.set_log_level('error')
for threshold in [.05,.02]:
 for i in [1,2]:
  out=P/f'ep25_board{i}_off{threshold:.3f}_hulls.npz'
  if out.exists():continue
  m=trimesh.load(S/f'score_input/meshes/episode_000025/wooden_piece_{i}.obj',force='mesh',process=False);m.merge_vertices(merge_tex=True,merge_norm=True);m.process(validate=True);assert m.is_watertight
  start=time.time();parts=coacd.run_coacd(coacd.Mesh(np.ascontiguousarray(m.vertices,dtype=np.float64),np.ascontiguousarray(m.faces,dtype=np.int32)),threshold=threshold,preprocess_mode='off',resolution=10000,mcts_iterations=150,seed=0);a={'n':np.int64(len(parts))}
  for j,(v,f) in enumerate(parts):a[f'v{j}']=v;a[f'f{j}']=f
  tmp=out.with_suffix('.tmp.npz');np.savez(tmp,**a);tmp.replace(out);print(json.dumps(dict(board=i,hulls=len(parts),threshold=threshold,wall_seconds=time.time()-start)),flush=True)
