"""Episode19 contact-supervision ablation; preserve all object and robot trajectories."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ref=json.loads((HERE/'ep19_reference.json').read_text())
S=Path(ref['entry']['scene']);D=HERE/'ep19_contact_gap_scene'
if D.exists():raise FileExistsError(D)
p=Path(ref['parquet']);original=pq.read_table(p);table=original
keys=[f'mano_right_object_contact_{suffix}' for suffix in ('positions','normals','part_ids')]
data={k:np.asarray(table[k][0].as_py()).copy() for k in keys}
ids=data[keys[-1]];old_ids=ids.copy()
assert table['object_body_names'][0].as_py()[0].endswith('_mini_sweeper')
# Raw contact IDs are one-indexed object slots; zero denotes no contact.
valid=np.flatnonzero((ids==1).any(1)&~(ids>1).any(1))
filled=[]
for f in range(22,200):
    if (ids[f]>0).any():continue
    before=valid[valid<f];after=valid[valid>f]
    if not len(before) or not len(after):continue
    a,b=int(before[-1]),int(after[0])
    if b-a-1>24:continue
    donor=a if f-a<=b-f else b
    for key in keys:data[key][f]=data[key][donor]
    filled.append(dict(frame=f,donor_frame=donor,bracketing_frames=[a,b]))
assert filled,'No eligible short contact gaps'
for key,value in data.items():
    i=table.column_names.index(key);field=table.schema.field(i)
    table=table.set_column(i,field,pa.array([value.tolist()],type=field.type))
for key in ('object_mesh_paths','object_urdf_paths'):
    i=table.column_names.index(key);field=table.schema.field(i)
    paths=[str(D/'assets'/s.split('/assets/',1)[1]) for s in table[key][0].as_py()]
    table=table.set_column(i,field,pa.array([paths],type=field.type))
for key in original.column_names:
    if key not in keys and key not in ('object_mesh_paths','object_urdf_paths'):
        assert table[key].equals(original[key]),key
assert np.array_equal(ids[:22],old_ids[:22]) and np.array_equal(ids[200:],old_ids[200:])
D.mkdir()
for directory in ('assets','score_input'):shutil.copytree(S/directory,D/directory)
out=D/p.relative_to(S);pq.write_table(table,out)
audit=dict(episode=19,source_scene=str(S),source_parquet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
           target_parquet=str(out),changed_columns=keys,filled=filled,
           object_and_robot_trajectories_unchanged=True,physics_and_meshes_unchanged=True,
           rationale='Video-supported right brush hold; fill only all-zero rows bracketed by brush-only labels, maximum24-frame gap, within22..199')
(D/'contact_gap_repair.json').write_text(json.dumps(audit,indent=2)+'\n')
extra='task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 training.checkpoint.save_interval_environment_steps=5000192 training.replay.capacity=1048576 training.replay.observation_storage_dtype=float16'
trial=dict(name='ep19_TARGET_CONTACTFILL_s43',episode=19,account='edmondwa',partition='preempt',qos='preempt_qos',
           scene=str(D),parquet=str(out.parent),baseline_episode_auc=ref['entry']['tracking'],seed=43,total_environment_steps=100003840,
           hypothesis='Prevent intermittent no-contact supervision from penalizing legitimate brush grasp during a visually supported continuous hold; same robot/object motion and baseline exploration.',extra=extra)
(HERE/'ep19_contact_trial.json').write_text(json.dumps(trial,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
