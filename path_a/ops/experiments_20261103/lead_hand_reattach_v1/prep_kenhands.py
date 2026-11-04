"""Prepare 'kenclean' bundles: our v1 objects + Ken's manually corrected raw hands (hand_* arrays of their
experiments/haisenk/hand_fixes_wentao_v1_20261103/episode_X/bundle/result.npz, which are the team raw hands re-expressed
in the fb11 world = our world, with ep3's left-hand HaMeR rerun and ep14's chirality fix). wrist_to_world is recomputed
from our camera_to_world for safety. Usage: prep_kenhands.py <ep> [...]"""
from pathlib import Path
import sys, json, shutil, hashlib
import numpy as np
SRC=Path(__file__).resolve().parent; import os
O=Path(os.environ.get('LEAD_HAND_OUT',str(SRC)))
S=Path(os.environ.get('LEAD_BUNDLES_ROOT',os.environ.get('V2D_ROOT','/Volumes/E/Code/V2D')+'/work/lead-sprint-release-20261103/stage/scenes'))
K=SRC/'ken_bundles'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for ep in [int(a) for a in sys.argv[1:]]:
    src=S/f'episode_{ep:06d}/bundle'; kb=K/f'episode_{ep:06d}/result.npz'; out=O/'bundles'/f'ep{ep:02d}_kenclean'
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(src,out)
    w=dict(np.load(out/'result.npz',allow_pickle=False)); k=dict(np.load(kb,allow_pickle=False))
    assert np.array_equal(w['camera_to_world_transform'],k['camera_to_world_transform']),(ep,'camera differs from Ken bundle')
    c2w=w['camera_to_world_transform'].astype(np.float64); replaced=[]
    for key in [x for x in k if x.startswith('hand_')]:
        assert key in w and w[key].shape==k[key].shape,(ep,key); w[key]=k[key]; replaced.append(key)
    for side in ('left','right'):
        w2c=k[f'hand_{side}_wrist_to_camera_transform'].astype(np.float64); w[f'hand_{side}_wrist_to_world_transform']=(c2w@w2c).astype(k[f'hand_{side}_wrist_to_world_transform'].dtype)
    np.savez(out/'result.npz',**w)
    chk=np.load(out/'result.npz',allow_pickle=False); orig=np.load(src/'result.npz',allow_pickle=False)
    assert all(np.array_equal(chk[x],orig[x]) for x in orig.files if not x.startswith('hand_')),'non-hand arrays changed'
    (out/'HANDFIX_PREP.json').write_text(json.dumps(dict(episode=ep,mode='kenclean',source_bundle=str(src),ken_bundle_hf_path=f'experiments/haisenk/hand_fixes_wentao_v1_20261103/episode_{ep:06d}/bundle/result.npz',ken_result_sha256=sha(kb),replaced_keys=replaced,non_hand_arrays_identical=True),indent=1)+'\n')
    print('PREP_KEN_OK',ep,out,flush=True)
