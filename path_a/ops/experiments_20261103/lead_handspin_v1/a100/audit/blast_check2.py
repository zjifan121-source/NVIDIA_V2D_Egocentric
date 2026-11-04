import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq
P="assets/human_motion_data/ego_recon/processed"
def pqs(s): return sorted(glob.glob(f"{s}/{P}/sequence_id=*/robot_name=*/*.parquet"))
def load(f):
    t=pq.read_table(f); return {c:t.column(c)[0].as_py() for c in t.column_names}
for ep in (10,14):
    new=f"/root/v2d/lead_v3/blast/scenes/ep{ep:02d}_hold"; fn=pqs(new)
    print(f"===== ep{ep} parquets={len(fn)} seq={fn[0].split('sequence_id=')[1].split('/')[0] if fn else None}")
    if not fn: continue
    A=load(fn[0]); us=A['object_urdf_paths']+A['object_mesh_paths']
    print(" paths own:",all(p.startswith(new) for p in us),"exist:",all(os.path.exists(p) for p in us),"score meshes:",sorted(os.path.basename(x) for x in glob.glob(new+"/score_input/meshes/episode_*/*.obj")),"apple:",len(glob.glob(new+"/**/._*",recursive=True)),"bodies:",A['object_body_names'],"frames:",len(A['object_body_position']))
    for f in ("OUTPUT_CHECK.json","HAND_CONTACT.json","SCENE_COM_FIX.json","SHIFT_CUPS.json","ATTACH_HOLD.json"):
        p=f"{new}/{f}"
        if not os.path.exists(p): print(f" {f}: missing"); continue
        d=json.load(open(p))
        if f=="OUTPUT_CHECK.json": print(" penetration:",d.get("penetration_frames"),"task_error:",d.get("task_error_cm"),"frame0 seating:",d.get("frame0_seating_cm"))
        elif f=="HAND_CONTACT.json": print(" contact:",{o:{k:(c.get("within_1cm") if isinstance(c,dict) else c) for k,c in v.items()} for o,v in d["objects"].items()})
        elif f=="SCENE_COM_FIX.json": print(" com:",[(o.get("body"),o.get("offset_cm"),o.get("patched")) for o in d.get("objects",[])])
        else: print(f" {f}:",json.dumps(d)[:300])
    for u in sorted(glob.glob(f"{new}/{P}/*_rigid.urdf")): print("  ",os.path.basename(u),[l.strip() for l in open(u) if '<origin' in l][:1])
    cands={"lead_v3":f"/root/v2d/lead_v3/stage/scenes/episode_{ep:06d}/scene","rawexcl":f"/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes/ep{ep:02d}_rawexcl","control":f"/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes/ep{ep:02d}_control","blast_excl":f"/root/v2d/lead_v3/blast/scenes/ep{ep:02d}_excl","blast_kenclean":f"/root/v2d/lead_v3/blast/scenes/ep{ep:02d}_kenclean"}
    for name,base in cands.items():
        fb=pqs(base)
        if not fb: continue
        B=load(fb[0]); pa=np.array(A['object_body_position'],float); pb=np.array(B['object_body_position'],float)
        if pa.shape!=pb.shape: print(f" vs {name}: shape differs {pa.shape} {pb.shape}"); continue
        qa=np.array(A['object_body_wxyz'],float); qb=np.array(B['object_body_wxyz'],float)
        dl=np.linalg.norm(np.array(A['robot_left_wrist_position'],float)-np.array(B['robot_left_wrist_position'],float),axis=1); dr=np.linalg.norm(np.array(A['robot_right_wrist_position'],float)-np.array(B['robot_right_wrist_position'],float),axis=1)
        print(f" vs {name}: obj pos max cm {(np.abs(pa-pb).max(axis=(0,2))*100).round(2).tolist()} median {(np.median(np.abs(pa-pb),axis=(0,2))*100).round(2).tolist()} rot(1-|dot|) max {np.abs(1-np.abs((qa*qb).sum(-1))).max(axis=0).round(4).tolist()} wrist med cm L={np.median(dl)*100:.1f} R={np.median(dr)*100:.1f}")
