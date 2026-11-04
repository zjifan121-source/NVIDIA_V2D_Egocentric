import glob, json, numpy as np
from scipy.spatial.transform import Rotation as R
W='/data/user_data/edmondwa/v2d/wentao_mix/hub/experiments/wentacc/scored20_alignment_mix_v1_20261030/scenes/episode_{:06d}/bundle/result.npz'
P='/data/user_data/edmondwa/v2d/recon_data/outputs/pathb_p4s/episode_{:06d}/*/result_slam_gravity_aligned/result.npz'
def stats(f):
    d=np.load(f); out={}
    for s in ('left','right'):
        T=d[f'hand_{s}_wrist_to_world_transform'].astype(float); v=d[f'hand_{s}_is_valid'].astype(bool)
        ok=v[1:]&v[:-1]
        r=R.from_matrix(T[:,:3,:3]); deg=np.degrees((r[1:]*r[:-1].inv()).magnitude())[ok]
        out[s]=(round(float(v.mean()),2), round(float(deg.max()),0) if len(deg) else None, int((deg>60).sum()))
    return out
for ep in [1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]:
    try: w=stats(W.format(ep))
    except Exception as e: w=repr(e)[:40]
    p=stats(sorted(glob.glob(P.format(ep)))[0])
    print(ep, 'W', w, '| PB', p)
