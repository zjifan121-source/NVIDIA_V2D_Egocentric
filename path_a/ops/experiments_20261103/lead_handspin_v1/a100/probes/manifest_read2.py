import json
from huggingface_hub import hf_hub_download
repo="rose-tea-cafe/v2d-t3-recon"
d=json.load(open("/root/probes/hf_current/experiments/wentacc/lead_current/CURRENT_MANIFEST_r1.json"))
e=d["episodes"]; print(type(e),len(e))
items=e if isinstance(e,list) else [dict(v,episode=k) if isinstance(v,dict) else {"episode":k,"v":v} for k,v in e.items()]
print("keys:",list(items[0].keys()))
for it in items:
    ep=int(str(it.get("episode")).lstrip("0") or 0)
    if ep in (1,5,14,25,28,30,33):
        print(json.dumps(it,ensure_ascii=False)[:620])
for f in ["experiments/wentacc/lead_current/VERSION_LEDGER_r1.md","experiments/wentacc/scored20_lead_v3_hands_r1_20261104/README.md"]:
    p=hf_hub_download(repo,f,repo_type="dataset",local_dir="/root/probes/hf_current")
    t=open(p).read(); print("=====",f,len(t)); print(t[:2800])
