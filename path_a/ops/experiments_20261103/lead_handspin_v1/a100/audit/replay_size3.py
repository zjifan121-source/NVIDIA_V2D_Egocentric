import glob
from safetensors import safe_open
f=sorted(glob.glob("/root/runs/ep28_r2com_def_s200/train/state_*.safetensors"))[-1]
with safe_open(f,"numpy") as s:
    sh=[(k,s.get_slice(k).get_shape()) for k in s.keys()]
print(sh[:14])
two=[(k,v) for k,v in sh if len(v)==2]
print("2D first:",two[:10])
print("largest input dims:",sorted({v[0] for k,v in two},reverse=True)[:8])
print("non-2D sample:",[(k,v) for k,v in sh if len(v)==1 and v[0] not in (1,) ][:14])
