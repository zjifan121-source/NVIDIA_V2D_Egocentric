"""Compare original and donor robot references using official zero-residual full rollouts."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
entry=json.loads((HERE/'ep5_reference.json').read_text())
source=Path(entry['checkpoint']);raw=source.read_bytes();size=struct.unpack('<Q',raw[:8])[0]
header=json.loads(raw[8:8+size]);payload=bytearray(raw[8+size:])
header['__metadata__']['user.nominal_zero_residual']='true'
header['__metadata__']['user.targeted_grasp_diagnostic']='true'
for leaf in ('state.0018','state.0020'):
    assert header[leaf]['shape']==[56] and header[leaf]['dtype']=='F32'
    a,b=header[leaf]['data_offsets'];payload[a:b]=b'\0'*(b-a)
h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
control=HERE/'ep5_original_nominal';control.mkdir(exist_ok=True)
policy=control/'policy_zero.safetensors';policy.write_bytes(struct.pack('<Q',len(h))+h+payload)
(control/'checkpoints.json').write_text(json.dumps({'5':str(policy)})+'\n')
summaries=[]
for directory in (control,HERE/'ep5_donor_grasp_nominal'):
    if not (directory/'eval.done').exists():
        with (directory/'eval.log').open('a') as log:
            subprocess.run([sys.executable,str(REPO/'assets/data/v2d_submission_kit/examples/run_policy_evaluation.py'),
                '--checkpoints',str(directory/'checkpoints.json'),'--results',str(directory/'results'),
                '--runtime-root',str(REPO/'baseline/robotic_grounding/flash_chord')],
                cwd=REPO/'baseline/robotic_grounding/flash_chord',stdout=log,stderr=subprocess.STDOUT,check=True)
        (directory/'eval.done').touch()
    result=json.loads((directory/'results/episode_000005.json').read_text())
    summaries.append(dict(variant=directory.name,metrics=result['metrics'],diagnostic_only=True,
                          checkpoint_sha256=result['checkpoint_sha256'],step_count=result['step_count']))
    print(json.dumps(summaries[-1]),flush=True)
(HERE/'ep5_grasp_comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
