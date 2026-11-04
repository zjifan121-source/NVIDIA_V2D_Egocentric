import glob, hashlib, json, os
import numpy as np, pyarrow.parquet as pq
P="assets/human_motion_data/ego_recon/processed"
new="/root/v2d/lead_v3/resthold/scenes/ep33_rawspinroll"; base="/root/v2d/lead_v3/handspinroll/scenes/ep33_rawspinroll"
def pqs(s): return sorted(glob.glob(f"{s}/{P}/sequence_id=*/robot_name=*/*.parquet"))
def load(f):
    t=pq.read_table(f); return {c:t.column(c)[0].as_py() for c in t.column_names}
fn=pqs(new); fb=pqs(base); print("parquets new",len(fn),"base",len(fb))
A=load(fn[0]); B=load(fb[0]); us=A['object_urdf_paths']+A['object_mesh_paths']
print("seq",fn[0].split('sequence_id=')[1].split('/')[0],"paths own:",all(p.startswith(new) for p in us),"exist:",all(os.path.exists(p) for p in us),"score meshes:",sorted(os.path.basename(x) for x in glob.glob(new+"/score_input/meshes/episode_*/*.obj")),"apple:",len(glob.glob(new+"/**/._*",recursive=True)),"bodies",A['object_body_names'])
sha=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
fa={os.path.relpath(f,base) for f in glob.glob(base+"/**/*",recursive=True) if os.path.isfile(f)}
fbn={os.path.relpath(f,new) for f in glob.glob(new+"/**/*",recursive=True) if os.path.isfile(f)}
print("only in resthold:",sorted(fbn-fa)[:8]); print("only in base:",sorted(fa-fbn)[:5])
d=[f for f in sorted(fa&fbn) if sha(base+"/"+f)!=sha(new+"/"+f)]; print("files differing:",len(d),[x.split('/')[-1][:40] for x in d[:8]])
for u in sorted(glob.glob(f"{new}/{P}/*_rigid.urdf")): print("  ",os.path.basename(u),[l.strip() for l in open(u) if '<origin' in l][:1])
cols=[]
for c in sorted(set(A)&set(B)):
    x,y=A[c],B[c]
    if x==y: continue
    try:
        X=np.array(x,float); Y=np.array(y,float); cols.append(f"{c}: max {np.abs(X-Y).max():.4g}" if X.shape==Y.shape else f"{c}: SHAPE")
    except Exception: cols.append(f"{c}: strings")
print("columns differing:",len(cols)); print("  "+"\n  ".join(cols[:30]))
pa=np.array(A['object_body_position'],float); pb=np.array(B['object_body_position'],float)
dif=np.linalg.norm(pa-pb,axis=2)*100  # frames x bodies, cm
for i,n in enumerate(A['object_body_names']):
    moved=np.where(dif[:,i]>0.05)[0]
    print(f"  {n}: frames changed {len(moved)} ({moved.min() if len(moved) else '-'}..{moved.max() if len(moved) else '-'}), max shift {dif[:,i].max():.1f} cm")
for f in ("RESTHOLD.json","REST_HOLD.json","SCENE_CHANGE_NOTE.json","OUTPUT_CHECK.json","HAND_CONTACT.json"):
    p=f"{new}/{f}"
    if os.path.exists(p):
        dd=json.load(open(p))
        print(f,":",json.dumps(dd.get("penetration_frames") if f=="OUTPUT_CHECK.json" else ({o:{k:(c.get("within_1cm") if isinstance(c,dict) else c) for k,c in v.items()} for o,v in dd["objects"].items()} if f=="HAND_CONTACT.json" else dd))[:400])
    else: print(f,": missing")
