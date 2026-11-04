import glob, hashlib, json, os, sys
import numpy as np, pyarrow.parquet as pq
s=sys.argv[1]; P="assets/human_motion_data/ego_recon/processed"
f=sorted(glob.glob(f"{s}/{P}/sequence_id=*/robot_name=*/*.parquet")); assert len(f)==1,f
t=pq.read_table(f[0]); d={c:t.column(c)[0].as_py() for c in t.column_names}
h=lambda a:hashlib.sha256(np.ascontiguousarray(np.array(a,float)).tobytes()).hexdigest()[:12]
out={"seq":f[0].split('sequence_id=')[1].split('/')[0],"frames":len(d['object_body_position']),"bodies":d['object_body_names'],
     "obj_pos":h(d['object_body_position']),"obj_rot":h(d['object_body_wxyz']),"wrist_L":h(d['robot_left_wrist_position']),"wrist_R":h(d['robot_right_wrist_position']),
     "mano_L":h(d['mano_left_trans']),"mano_R":h(d['mano_right_trans']),"robot_frames_R":h(d['robot_right_frames']),"radius":[round(x,5) for x in d['object_mesh_radius']]}
for m in sorted(glob.glob(f"{s}/{P}/*.obj")): out["mesh:"+os.path.basename(m).split('_',3)[-1]]=hashlib.sha256(open(m,'rb').read()).hexdigest()[:12]
for u in sorted(glob.glob(f"{s}/{P}/*_rigid.urdf")): out["urdf_origin:"+os.path.basename(u).split('_',3)[-1]]=[l.strip() for l in open(u) if '<origin' in l][0][:60]
print(json.dumps(out,indent=0))
