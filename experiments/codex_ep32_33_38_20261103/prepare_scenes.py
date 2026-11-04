"""Prepare isolated training copies; imported HF payloads remain immutable."""
from pathlib import Path
import json, shutil, struct, subprocess, sys
import numpy as np
import pyarrow.parquet as pq

base=Path('/data/v2d');inputs=base/'inputs';out=base/'scenes';out.mkdir(exist_ok=True)
went=inputs/'experiments/wentacc/scored20_lead_sprint_v2_20261103'
haisen=inputs/'experiments/haisenk/hand_fixes_wentao_v1_20261103'
sources={
 'ep32_went_v2':went/'scenes/episode_000032/scene',
 'ep33_went_v2':went/'scenes/episode_000033/scene',
 'ep38_ken_control':inputs/'night_2026-10-24/scenes/episode_000038/ken_c_scored_attach',
 'ep32_haisen_respin':haisen/'episode_000032/scene',
}
report={}
for label,source in sources.items():
 dest=out/label
 if not dest.exists():shutil.copytree(source,dest)
 if label in ('ep32_went_v2','ep33_went_v2'):
  ep=int(label[2:4]);raw=went/f'scenes/episode_{ep:06d}/bundle_rawexcl'
  raw_dest=dest/'bundles'/f'ep{ep}_rawexcl'
  if not raw_dest.exists():shutil.copytree(raw,raw_dest)
 run=subprocess.run([sys.executable,str(went/'tools/relocate_scene.py'),str(dest)],capture_output=True,text=True)
 (base/'logs'/f'{label}_relocate.log').write_text(run.stdout+run.stderr)
 assert run.returncode==0,label
 parquets=list(dest.glob('assets/**/processed/sequence_id=*/robot_name=*/*.parquet'))
 assert len(parquets)==1,(label,parquets)
 parquet=parquets[0]
 record=pq.ParquetFile(parquet).read().to_pylist()[0]
 for key in ('object_body_position','object_body_wxyz','robot_joint_pos'):
  if key in record:
   a=np.asarray(record[key]);assert np.isfinite(a).all(),(label,key)
 for key in ('raw_motion_file','object_mesh_paths','object_urdf_paths'):
  vs=record.get(key,[]);vs=[vs] if isinstance(vs,str) else vs
  for path in vs:
   assert Path(path).exists(),(label,key,path)
 report[label]={'parquet':str(parquet.parent),'parquet_file':str(parquet),'source':str(source),'frames':len(record['object_body_position']),'objects':record['object_body_names'],'fps':record.get('fps')}

# Supply the original absolute task path as an alias rather than alter policy metadata.
ckpt=inputs/'night_2026-10-24/runs/ep38_kenC_s43/train/policy_100003840.safetensors'
with ckpt.open('rb') as f:
 h=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
cfg=json.loads(h['__metadata__']['user.resolved_config_json'])
old=Path(cfg['task']['parquet']);assert str(old).startswith(('/data/user_data/haisenk/','/scratch/haisenk-v2d/')),str(old)
target=Path(report['ep38_ken_control']['parquet_file' if old.suffix=='.parquet' else 'parquet'])
if not old.exists():old.parent.mkdir(parents=True,exist_ok=True);old.symlink_to(target,target_is_directory=target.is_dir())
assert old.resolve()==target.resolve()
report['ep38_checkpoint_alias']={'saved_path':str(old),'target':str(target),'checkpoint':str(ckpt),'tensor_payload_changed':False}
(base/'receipts/prepared_scenes.json').write_text(json.dumps(report,indent=2))
(base/'receipts/control38_checkpoints.json').write_text(json.dumps({'38':str(ckpt)}))
print(json.dumps(report,indent=2))
