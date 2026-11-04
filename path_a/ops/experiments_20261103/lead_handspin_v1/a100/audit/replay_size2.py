import glob, json
from safetensors import safe_open
f=sorted(glob.glob("/root/runs/ep28_r2com_def_s200/train/state_*.safetensors"))[-1]
with safe_open(f,"numpy") as s:
    ks=list(s.keys())
    sh={k:s.get_slice(k).get_shape() for k in ks}
print(len(ks))
big=[(k,v) for k,v in sh.items() if ("kernel" in k.lower() and "actor" in k.lower())][:6]+[(k,v) for k,v in sh.items() if ("kernel" in k.lower() and "critic" in k.lower())][:4]
print(big)
print([ (k,v) for k,v in sh.items() if any(x in k.lower() for x in ("norm","mean","count","running"))][:8])
