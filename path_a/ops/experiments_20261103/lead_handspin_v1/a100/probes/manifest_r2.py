import glob, hashlib, json
import numpy as np, pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
repo="rose-tea-cafe/v2d-t3-recon"; api=HfApi()
p=hf_hub_download(repo,"experiments/wentacc/lead_current/CURRENT_MANIFEST_r2.json",repo_type="dataset",local_dir="/root/probes/hf_current3",force_download=True)
d=json.load(open(p)); print("manifest r2 updated",d.get("updated_utc"))
for e in d["episodes"]:
    if e["episode"] in (25,28,33):
        print(e["episode"],e.get("scene_path"),"|",str(e.get("status"))[:90],"|",json.dumps({k:v for k,v in e.items() if "candidate" in k or "alt" in k},ensure_ascii=False)[:300])
fs=api.list_repo_files(repo,repo_type="dataset")
pre="experiments/wentacc/scored20_lead_v3_hands_r1_20261104/scenes/episode_000028_r2/"
ps=[f for f in fs if f.startswith(pre) and "/processed/" in f and f.endswith(".parquet")]
print("ep28_r2 processed parquets:",len(ps))
cols=['object_body_position','object_body_wxyz','robot_left_wrist_position','robot_right_wrist_position']
def fp(f):
    t=pq.read_table(f,columns=cols); h=hashlib.sha256()
    for c in cols: h.update(np.array(t.column(c)[0].as_py(),float).tobytes())
    return h.hexdigest()[:12]
if ps:
    q=hf_hub_download(repo,ps[0],repo_type="dataset",local_dir="/root/probes/hf_current3")
    loc=glob.glob("/root/v2d/lead_v3/handspin28r2/scenes/ep28_hold/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet")[0]
    print("fingerprint HF",fp(q),"local r2",fp(loc))
