"""Official unassisted evaluation of a policy that outputs zero residual actions.

Moving wrist/finger reference commands remain active. Only mean-head gain/bias
are zeroed; batch normalization and every other network leaf are preserved.
"""
import json,struct,os,hashlib,subprocess
from pathlib import Path
P=Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28');REPO=P.parents[1];runs=Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
tasks=[(25,'ep25_grip2_scene','ep25_nominal_grip2'),(25,'ep25_pinch2_scene','ep25_nominal_pinch2'),(25,'ep25_assembly2_scene','ep25_nominal_assembly2'),(3,'ep3_grip_scene','ep3_nominal_grip')]
tasks=json.loads(Path(os.environ['V2D_NOMINAL_TASKS']).read_text()) if os.environ.get('V2D_NOMINAL_TASKS') else tasks
ep,scene,name=tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])];S=P/scene;run=runs/name;run.mkdir(exist_ok=True);(run/'train').mkdir(exist_ok=True)
source=runs/'ep25_grip2_OT1M_s44/train/policy_100003840.safetensors';raw=source.read_bytes();n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);payload=bytearray(raw[8+n:]);m=header['__metadata__'];config=json.loads(m['user.resolved_config_json']);parquet=next(S.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave'));config['task']['parquet']=str(parquet);m['user.resolved_config_json']=json.dumps(config,separators=(',',':'));m['user.environment_steps']='0';m['user.nominal_zero_residual']='true';m['user.nominal_source_sha256']=hashlib.sha256(raw).hexdigest();m['user.nominal_zeroed_leaves']="params.mean.gain, params.mean_bias"
for key in ['state.0018','state.0020']:
 leaf=header[key];assert leaf['shape']==[56] and leaf['dtype']=='F32';a,b=leaf['data_offsets'];payload[a:b]=b'\0'*(b-a)
h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8);policy=run/'train/policy_0.safetensors';policy.write_bytes(struct.pack('<Q',len(h))+h+payload);(run/'checkpoints.json').write_text(json.dumps({str(ep):str(policy)})+'\n')
with (run/'eval.log').open('w') as log:
 subprocess.run(['/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python',str(REPO/'assets/data/v2d_submission_kit/examples/run_policy_evaluation.py'),'--checkpoints',str(run/'checkpoints.json'),'--results',str(run/'results'),'--runtime-root',str(REPO/'baseline/robotic_grounding/flash_chord')],stdout=log,stderr=subprocess.STDOUT,check=True)
(run/'eval.done').touch();score=json.loads((run/'results'/f'episode_{ep:06d}.json').read_text())['metrics'];print(name,score['add_auc'],score.get('add_auc_per_body'),flush=True)
