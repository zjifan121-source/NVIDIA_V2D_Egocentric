"""Prepare 'rawhands' and 'control' bundles on the lead_sprint_v1 objects (haisenk wprep.py re-implemented, 2026-11-03).
rawhands: copy of our HF-stage bundle with every hand_* array replaced by the team's RAW pre-cleanup hands
(night_2026-10-24/scenes/episode_X/ken_c_scored_attach/bundle_merged/result.npz on HF), re-expressed in our world:
wrist_to_world = our camera_to_world @ raw wrist_to_camera (camera-frame arrays are world-independent). Objects, camera
and everything else stay ours byte for byte. control: exact copy of our bundle (fb11 hands = team cleaned hands attached
to the OLD object tracks). Usage: prep_rawhands.py <ep> [<ep> ...]"""
from pathlib import Path
import sys, json, shutil, hashlib
import numpy as np
SRC=Path(__file__).resolve().parent; import os
O=Path(os.environ.get('LEAD_HAND_OUT',str(SRC)))
S=Path(os.environ.get('LEAD_BUNDLES_ROOT',os.environ.get('V2D_ROOT','/Volumes/E/Code/V2D')+'/work/lead-sprint-release-20261103/stage/scenes'))
RAW=SRC/'raw_hands/night_2026-10-24/scenes'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for ep in [int(a) for a in sys.argv[1:]]:
    src=S/f'episode_{ep:06d}/bundle'; raw_path=RAW/f'episode_{ep:06d}/ken_c_scored_attach/bundle_merged/result.npz'
    for mode in ['control','rawhands']:
        out=O/'bundles'/f'ep{ep:02d}_{mode}'
        if out.exists(): shutil.rmtree(out)
        shutil.copytree(src,out)
        info=dict(episode=ep,mode=mode,source_bundle=str(src),source_result_sha256=sha(src/'result.npz'),hf_stage_prefix='experiments/wentacc/scored20_lead_sprint_v1_20261103',
                  raw_hands_hf_path=f'night_2026-10-24/scenes/episode_{ep:06d}/ken_c_scored_attach/bundle_merged/result.npz' if mode=='rawhands' else None)
        if mode=='rawhands':
            w=dict(np.load(out/'result.npz',allow_pickle=False)); raw=dict(np.load(raw_path,allow_pickle=False)); info['raw_result_sha256']=sha(raw_path)
            assert w['camera_to_world_transform'].shape==raw['camera_to_world_transform'].shape,(ep,'frame count')
            c2w=w['camera_to_world_transform'].astype(np.float64); replaced=[]
            for side in ('left','right'):
                keys=[k for k in raw if k.startswith(f'hand_{side}_')]
                for k in keys:
                    assert k in w and w[k].shape==raw[k].shape and w[k].dtype==raw[k].dtype,(ep,k,w.get(k).shape if k in w else None,raw[k].shape)
                    w[k]=raw[k]; replaced.append(k)
                w2c=raw[f'hand_{side}_wrist_to_camera_transform'].astype(np.float64)
                w[f'hand_{side}_wrist_to_world_transform']=(c2w@w2c).astype(raw[f'hand_{side}_wrist_to_world_transform'].dtype)
                rc=raw['camera_to_world_transform'].astype(np.float64)@w2c
                info[f'raw_{side}_selfconsistency_max']=float(np.abs(rc-raw[f'hand_{side}_wrist_to_world_transform']).max())
            info['replaced_keys']=replaced
            np.savez(out/'result.npz',**w)
            chk=np.load(out/'result.npz',allow_pickle=False); orig=np.load(src/'result.npz',allow_pickle=False)
            untouched=[k for k in orig.files if not k.startswith('hand_')]; assert all(np.array_equal(chk[k],orig[k]) for k in untouched),'non-hand arrays changed'
            info['non_hand_arrays_identical']=True
        (out/'HANDFIX_PREP.json').write_text(json.dumps(info,indent=1)+'\n'); print('PREP_OK',ep,mode,out,flush=True)
