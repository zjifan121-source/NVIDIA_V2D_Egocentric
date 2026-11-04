import json, os, sys
from huggingface_hub import hf_hub_download, HfApi
repo="rose-tea-cafe/v2d-t3-recon"
api=HfApi()
fs=api.list_repo_files(repo,repo_type="dataset")
cur=[f for f in fs if f.startswith("experiments/wentacc/lead_current/")]
print("lead_current files:",cur)
hr=[f for f in fs if f.startswith("experiments/wentacc/scored20_lead_v3_hands_r1_20261104/")]
print("hands_r1 files:",len(hr), sorted({'/'.join(f.split('/')[3:5]) for f in hr})[:30])
p=hf_hub_download(repo,"experiments/wentacc/lead_current/CURRENT_MANIFEST_r1.json",repo_type="dataset",local_dir="/root/probes/hf_current")
d=json.load(open(p))
print("top keys:",list(d.keys())[:20])
eps=d.get("episodes",d)
for k in sorted(eps,key=lambda x:str(x)):
    if any(str(k).endswith(e) or str(k)==e for e in ("25","28","33","5","1","14","000025","000028","000033")):
        print(k, json.dumps(eps[k],ensure_ascii=False)[:520])
print(json.dumps({k:v for k,v in d.items() if k!="episodes"},ensure_ascii=False)[:900])
