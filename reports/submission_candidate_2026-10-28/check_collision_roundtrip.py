import json,sys,time
from pathlib import Path
import numpy as np,trimesh
P=Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28');S=P/'ep25_assembly_scene';sys.path.insert(0,str(P.parents[1]/'baseline/robotic_grounding/flash_chord/src'))
from flash_chord.assets.mesh import convex_decompose
rows=[];start=time.time()
for p in sorted(S.glob('assets/**/slot_collision/*.stl')):
 m=trimesh.load(p,force='mesh');parts=convex_decompose(m.vertices,m.faces);v=np.concatenate([h.vertices for h in parts]);bound_error=float(np.abs(np.r_[v.min(0)-m.vertices.min(0),v.max(0)-m.vertices.max(0)]).max());assert len(parts)==1 and bound_error<1e-5,(p,len(parts),bound_error);rows.append(dict(file=str(p.relative_to(S)),pieces=len(parts),bound_error_m=bound_error))
print('validated',len(rows),'collision pieces; seconds',time.time()-start,flush=True);(S/'collision_roundtrip.json').write_text(json.dumps(dict(pieces=len(rows),all_single_convex=True,rows=rows),indent=2)+'\n')
