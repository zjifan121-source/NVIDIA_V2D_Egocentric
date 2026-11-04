P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python; S=/scratch/haisenk-v2d/hand5w/scenes; O=/scratch/haisenk-v2d/hand5w/r2/ep1
ls $S/w_ctl_ep1/
$P - <<'PY'
import numpy as np, glob
S="/scratch/haisenk-v2d/hand5w/scenes"; W="/scratch/haisenk-v2d/wentao_mix_20261103/experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103/scenes/episode_000001/bundle/result.npz"
w=np.load(W,allow_pickle=True)
for n in ("w_ctl_ep1","w_raw_ep1","w_hold_ep1"):
    for p in glob.glob(f"{S}/{n}/bundle_cleaned/result.npz"):
        h=np.load(p,allow_pickle=True)
        for k in ("object_to_camera_transform","object_1_to_camera_transform","object_to_world_transform","object_1_to_world_transform"):
            print(n,k,h[k].dtype,w[k].dtype,"maxabs",float(np.abs(h[k].astype(float)-w[k].astype(float)).max()))
e={n:np.load(f"/scratch/haisenk-v2d/hand5w/r2/ep1/ev_{n}.npz") for n in ("w_ctl_ep1","w_raw_ep1","w_hold_ep1")}
for side,edges in (("left",[32,127,221,304]),("right",[45,47,59,68,250,255,272,274])):
    for n,d in e.items():
        s=np.linalg.norm(np.diff(d[f"{side}_J"][:,0],axis=0),axis=1)*1000
        print(f"{n} {side} MANO wrist step mm around edges: "+" | ".join(f"f{a}: "+" ".join(f"{s[t]:.0f}" for t in range(a-3,a+3)) for a in edges))
# lid carry: right tips to lid while lid descending/ascending
for n,d in e.items():
    sd=d["right_sd"][:,1,:]*1000
    for a,b in ((45,68),(250,274)):
        print(f"{n} right on lid [{a},{b}): wrist med {np.median(sd[a:b,0]):.0f} tips med "+" ".join(f"{np.median(sd[a:b,j]):.0f}" for j in range(1,6))+f"; frames min tip >20mm {(sd[a:b,1:].min(1)>20).sum()}")
    sdl=d["left_sd"][:,0,:]*1000
    for a,b in ((32,127),(127,221),(221,304)):
        print(f"{n} left on pot [{a},{b}): wrist med {np.median(sdl[a:b,0]):.0f} tips med "+" ".join(f"{np.median(sdl[a:b,j]):.0f}" for j in range(1,6))+f"; frames min tip >10mm {(sdl[a:b,1:].min(1)>10).sum()}; tip sd sd(std) mm {sdl[a:b,1:].std(0).round(0)}")
PY
