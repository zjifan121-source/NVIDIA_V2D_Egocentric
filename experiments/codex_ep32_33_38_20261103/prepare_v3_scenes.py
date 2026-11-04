"""Remote-only isolated import of pinned Wentao v3-r1 target scenes."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

base=Path('/data/v2d')
prefix=base/'inputs/experiments/wentacc/scored20_lead_v3_hands_r1_20261104/scenes'
relocator=base/'inputs/experiments/wentacc/scored20_lead_sprint_v2_20261103/tools/relocate_scene.py'
report={}
assert not (base/'receipts/prepared_v3_scenes.json').exists(),'Never mutate a validated scene'
for ep in (32,33):
 source=prefix/f'episode_{ep:06d}';dest=base/'scenes'/f'ep{ep}_went_v3_r1'
 if dest.exists():
  assert '--repair-incomplete' in sys.argv,'Never mutate an existing scene implicitly'
 else:
  shutil.copytree(source/'scene',dest)
  shutil.copytree(source/'bundle',dest/'bundle')
 r=subprocess.run([sys.executable,str(relocator),str(dest)],capture_output=True,text=True)
 (base/'logs'/f'ep{ep}_went_v3_r1_relocate.log').write_text(r.stdout+r.stderr)
 assert r.returncode==0
 paths=list(dest.glob('assets/**/processed/sequence_id=*/robot_name=*/*.parquet'))
 assert len(paths)==1
 parquet=paths[0];table=pq.ParquetFile(parquet).read()
 # The upstream raw path uses bundles/epXX_rawspin instead of bundle/.
 # Explicitly map it to the accompanying immutable bundle; change paths only.
 old=table['raw_motion_file'][0].as_py();raw=str(dest/'bundle/result.npz')
 value=raw if isinstance(old,str) else [raw]
 i=table.column_names.index('raw_motion_file')
 table=table.set_column(i,table.field(i),pa.array([value],type=table.field(i).type))
 pq.write_table(table,parquet);record=table.to_pylist()[0]
 for field in ('raw_motion_file','object_mesh_paths','object_urdf_paths'):
  values=record[field];values=[values] if isinstance(values,str) else values
  assert all(Path(x).is_file() and Path(x).resolve().is_relative_to(dest.resolve()) for x in values),(ep,field,values)
 for field in ('object_body_position','object_body_wxyz'):
  assert np.isfinite(np.asarray(record[field])).all(),(ep,field)
 if 'robot_joint_pos' in record:
  assert np.isfinite(np.asarray(record['robot_joint_pos'])).all()
 report[f'ep{ep}_went_v3_r1']={'parquet':str(parquet.parent),'parquet_file':str(parquet),
  'source':str(source),'frames':len(record['object_body_position']),'objects':record['object_body_names'],
  'fps':record['fps'],'raw_bundle_sha256':hashlib.sha256(Path(record['raw_motion_file']).read_bytes()).hexdigest(),
  'source_revision':'27578b2ae0249cdf8d8c830c6841b8bd0aaa501f','status':'SCENE_CLOSURE_PASS'}
(base/'receipts/prepared_v3_scenes.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
