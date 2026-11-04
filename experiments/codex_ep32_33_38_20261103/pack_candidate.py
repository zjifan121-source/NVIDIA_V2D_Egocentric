"""Remote-only: replace selected real rollouts in an immutable full20 control.

Mapping is by validated scene object-body order, not by reconstructed asset names.
No reference poses, interpolated frames, or altered evaluator digests are submitted.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, re, sys, tempfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--code-commit',required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--base',type=Path,default=Path('/data/v2d/inputs/path_b/runs/v4_eval_g4/pack_60m/track3_submission.parquet'))
p.add_argument('--base-sha256',default='298b9c1f73ddd99eedabb5a2b08a1e2ccdde24de6ea980ba989acd3aa35ab3ad')
a=p.parse_args()
root=Path('/data/v2d');kit=root/'repo/assets/data/v2d_submission_kit'
sys.path.insert(0,str(kit))
from v2dlb.submission_parquet import read_trajectory, provenance
from v2dlb.track3_pose import pack_object_meshes
from v2dlb import object_pose as op
spec=importlib.util.spec_from_file_location('official_submit',kit/'submit/v2d_submit.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
config=json.loads((kit/'config/tracks.json').read_text())
roster=json.loads((kit/'data/track_3_evaluation_objects.json').read_text())['episodes']
manifest=json.loads(a.manifest.read_text())
base=a.base
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert re.fullmatch('[0-9a-f]{64}',a.base_sha256) and sha(base)==a.base_sha256
assert re.fullmatch(r'https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}',a.code_commit)
original=official.validate_submission(base,'track_3',config)
frame=original.copy();epcol=frame.row_id.str.extract(r'/e(\d+)/')[0].astype(int)
audit={'base':str(base),'base_sha256':sha(base),'replacements':{},'code_commit_url':a.code_commit}
assert set(map(int,manifest)).issubset({32,33,38}) and manifest
with tempfile.TemporaryDirectory(prefix='pack_',dir=root) as tmp:
 for key,entry in manifest.items():
  ep=int(key);traj=read_trajectory(Path(entry['rollout']))
  assert traj.episode_index==ep and traj.world_valid.all()
  scene=pq.ParquetFile(entry['scene_parquet']).read().to_pylist()[0]
  bodies=scene['object_body_names'];meshes=scene['object_mesh_paths']
  assert len(bodies)==len(meshes)==len(roster[key])
  # The scene lists mesh paths in the same body order used by the simulator.
  for slot,name in enumerate(bodies):
   mask=traj.object_slot==slot
   assert mask.any() and np.all(traj.object_name[mask]==name),(ep,slot,name)
  recorded=provenance(traj)
  for field in ('eval_script_sha256','eval_code_sha256'):
   assert recorded[field]==original[field].iloc[0],(ep,field,'mixed evaluator builds')
  assert re.fullmatch('[0-9a-f]{64}',recorded['checkpoint_sha256'])
  ids=[f'track_3/e{ep:06d}/w{int(w):03d}/f{int(f):06d}/o{int(s)}'
       for w,f,s in zip(traj.world_index,traj.frame_index,traj.object_slot)]
  assert len(ids)==len(set(ids))
  lookup=pd.Series(np.arange(len(ids)),index=ids)
  rows=frame.index[epcol==ep]
  assert set(frame.loc[rows,'row_id']).issubset(lookup.index)
  poses=traj.pose_xyzw[lookup.loc[frame.loc[rows,'row_id']].to_numpy()]
  assert np.isfinite(poses).all()
  frame.loc[rows,op._POSE_COLUMNS]=poses
  frame.loc[rows,'checkpoint_sha256']=recorded['checkpoint_sha256']
  meshdir=Path(tmp)/f'episode_{ep:06d}';meshdir.mkdir()
  for slot,path in enumerate(meshes):
   source=Path(path);assert source.is_file()
   (meshdir/(roster[key][slot]+source.suffix)).symlink_to(source)
  packed=pack_object_meshes(frame.loc[rows].copy(),Path(tmp),roster)
  frame.loc[rows,'object_mesh_b64']=packed.object_mesh_b64
  audit['replacements'][key]={'rollout':entry['rollout'],'rollout_sha256':sha(entry['rollout']),
   'scene_parquet':entry['scene_parquet'],'scene_sha256':sha(entry['scene_parquet']),
   'body_to_official_name':dict(zip(bodies,roster[key])),
   'meshes':[{'path':str(x),'sha256':sha(x)} for x in meshes],
   'checkpoint_sha256':recorded['checkpoint_sha256'],'rows':len(rows)}
frozen=~epcol.isin(list(map(int,manifest)))
assert frame.loc[frozen].equals(original.loc[frozen]),'An unaffected episode changed'
# Global provenance describes this actual replacement packer and producing source.
frame['code_commit_url']=a.code_commit
frame['packer_script_sha256']=sha(__file__)
frame['uploader_script_sha256']=sha(kit/'submit/v2d_submit.py')
stable=[c for c in frame if c not in ('code_commit_url','packer_script_sha256','uploader_script_sha256')]
assert frame.loc[frozen,stable].equals(original.loc[frozen,stable])
a.out.parent.mkdir(parents=True,exist_ok=True)
frame.to_parquet(a.out,index=False)
official.validate_submission(a.out,'track_3',config)
audit.update(output_sha256=sha(a.out),rows=len(frame),unaffected_episodes_identical_except_global_provenance=True,
             status='OFFICIAL_OFFLINE_VALIDATOR_PASS')
a.out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
