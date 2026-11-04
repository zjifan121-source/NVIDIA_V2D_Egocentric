"""Check weights, reference poses and copied assets behind an unequal control rerun."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
info=json.loads((HERE/'ep5_reference.json').read_text())
entry=info['entry'];source=Path(entry['run'])/'results/episode_000005.json'
cached=json.loads(source.read_text())
control=json.loads((HERE/'ep5_timing_0_eval/results/episode_000005.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=dict(cached_auc=cached['metrics']['add_auc'],control_auc=control['metrics']['add_auc'],
         cached_checkpoint_sha=cached['checkpoint_sha256'],current_checkpoint_sha=sha(cached['checkpoint']),
         current_reference_sha=sha(info['parquet']),frozen_reference_sha=info['parquet_sha256'])
old=pq.read_table(source.with_suffix('.parquet')).to_pandas().sort_values(['environment_id','reference_frame','object_id'])
new=pq.read_table(HERE/'ep5_timing_0_eval/results/episode_000005.parquet').to_pandas().sort_values(['environment_id','reference_frame','object_id'])
for kind in ('reference_position','reference_quaternion','achieved_position','achieved_quaternion'):
    cols=[c for c in old if c.startswith(kind+'_')]
    out[kind+'_max_abs_difference']=float(np.abs(old[cols].to_numpy()-new[cols].to_numpy()).max())
out['same_frame_object_grid']=np.array_equal(old[['reference_frame','object_id']].to_numpy(),new[['reference_frame','object_id']].to_numpy())
S=Path(entry['scene']);D=HERE/'ep5_timing_0_scene';differences=[];count=0
for original in (S/'assets').rglob('*'):
    if not original.is_file() or original.suffix=='.parquet':continue
    if original.suffix.lower() not in ('.urdf','.usda','.obj','.stl','.ply','.glb','.npz'):continue
    copy=D/original.relative_to(S);count+=1
    if not copy.is_file() or sha(original)!=sha(copy):differences.append(str(original.relative_to(S)))
out['asset_files_compared']=count;out['different_physics_mesh_assets']=differences
for label,table in [('cached',pq.read_table(source.with_suffix('.parquet'))),('control',pq.read_table(HERE/'ep5_timing_0_eval/results/episode_000005.parquet'))]:
    out[label+'_provenance']={k.decode():v.decode() for k,v in (table.schema.metadata or {}).items()
                            if k.startswith(b'flash_chord.') and (b'sha256' in k or b'fps' in k)}
(HERE/'ep5_reproduction_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
