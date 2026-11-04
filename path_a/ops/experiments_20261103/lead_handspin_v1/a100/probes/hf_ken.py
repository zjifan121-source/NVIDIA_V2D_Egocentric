import json
from huggingface_hub import HfApi, hf_hub_download
repo="rose-tea-cafe/v2d-t3-recon"; api=HfApi()
fs=api.list_repo_files(repo,repo_type="dataset")
print("files",len(fs))
r=[f for f in fs if "scored20_lead_v3_hands_r1_20261104/scenes/episode_000028" in f and f.endswith(".parquet")]
print("r1 ep28 parquets:",[x.split("scenes/")[1][:100] for x in r])
print("hands_r2-like prefixes:",sorted({f.split("/")[2] for f in fs if f.startswith("experiments/wentacc/") and ("hands" in f.split("/")[2] or "r2" in f.split("/")[2])}))
k=[f for f in fs if f.startswith("experiments/haisenk/")]
print("haisenk top:",sorted({f.split("/")[2] for f in k}))
hf="experiments/haisenk/hand_fixes_wentao_v1_20261103/"
sub=sorted({"/".join(f[len(hf):].split("/")[:3]) for f in k if f.startswith(hf)})
print("hand_fixes entries:",len(sub)); print("\n".join(sub[:60]))
pol=[f for f in k if f.endswith(".safetensors") and ("ep1" in f or "ep5" in f or "episode_000001" in f or "episode_000005" in f)]
print("ep1/ep5 policies:",pol[:12])
res=[f for f in k if f.endswith(("episode_000001.parquet","episode_000005.parquet"))]
print("ep1/ep5 rollouts:",res[:12])
try:
    p=hf_hub_download(repo,"experiments/wentacc/lead_current/CURRENT_MANIFEST_r1.json",repo_type="dataset",local_dir="/root/probes/hf_current2",force_download=True)
    d=json.load(open(p)); print("manifest updated:",d["updated_utc"])
    for e in d["episodes"]:
        if e["episode"] in (25,28,33): print(e["episode"],e["scene_path"],"|",e["status"])
except Exception as ex: print("manifest err",ex)
