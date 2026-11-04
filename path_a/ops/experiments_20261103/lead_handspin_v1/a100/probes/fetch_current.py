# Fetch the scenes CURRENT_MANIFEST_r1.json names for the A100 queue. HF is read only here.
import json, sys
from huggingface_hub import snapshot_download
repo="rose-tea-cafe/v2d-t3-recon"
m=json.load(open("/root/probes/hf_current/experiments/wentacc/lead_current/CURRENT_MANIFEST_r1.json"))
want={e["episode"]:e["scene_path"] for e in m["episodes"] if e["episode"] in (1,25,28,33)}
print(want,flush=True)
pats=[p.rstrip("/")+"/*" for p in want.values()]+[p.rstrip("/").rsplit("/",1)[0]+"/PROVENANCE.json" for p in want.values()]
snapshot_download(repo,repo_type="dataset",local_dir="/root/lead_current",allow_patterns=pats,max_workers=6)
print("FETCH_DONE",flush=True)
