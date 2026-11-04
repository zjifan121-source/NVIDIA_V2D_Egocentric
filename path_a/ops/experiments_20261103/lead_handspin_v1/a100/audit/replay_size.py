import glob, json, re
from safetensors import safe_open
for n in ("ep33_spinroll_def_s200","ep25_r1_wide_s201","ep28_r2com_def_s200"):
    fs=sorted(glob.glob(f"/root/runs/{n}/train/policy_*.safetensors"))+sorted(glob.glob(f"/root/runs/{n}/train/state_*.safetensors"))
    print(n,len(fs))
f=sorted(glob.glob("/root/runs/ep28_r2com_def_s200/train/state_*.safetensors"))[-1]
with safe_open(f,"numpy") as s:
    m=s.metadata() or {}
    print({k:(v[:160]) for k,v in m.items() if any(x in k.lower() for x in ("obs","action","dim","world","context"))})
    ks=list(s.keys()); print(len(ks)); print([k for k in ks if "observation" in k.lower() or "context" in k.lower()][:10])
    for k in ks:
        if "observation" in k.lower() or "critic_context" in k.lower():
            print(k,s.get_tensor(k).shape,s.get_tensor(k).dtype)
