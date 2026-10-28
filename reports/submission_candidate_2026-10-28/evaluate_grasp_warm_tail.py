"""Held-state diagnostic only: partial rollouts never enter the submission inventory."""
import json,os,subprocess
from pathlib import Path
P=Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28');FC=P.parents[1]/'baseline/robotic_grounding/flash_chord';RUNS=Path('/data/user_data/edmondwa/v2d/flash_runs/diag');tasks=[('nominal',RUNS/'ep33_nominal_original/train/policy_0.safetensors'),('trained_best',RUNS/'kv_ep33_OT_s42/train/policy_100003840.safetensors'),('wide50',RUNS/'ep33_WIDE_s42/train/policy_50003968.safetensors')];name,checkpoint=tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])];out=P/'grasp_warm_tail_diagnostics'/name;out.mkdir(parents=True,exist_ok=True)
with (out/'eval.log').open('w') as log:
 subprocess.run(['/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python',str(FC/'scripts/evaluate_policy.py'),f'evaluation.checkpoint={checkpoint}','evaluation.world_count=1','evaluation.start_frame=0','evaluation.motion_start_frame=96','evaluation.reset_mode=sampled_settled',f'evaluation.metrics_output={out/"metrics.json"}',f'evaluation.object_trajectories_output={out/"trajectory.parquet"}',f'hydra.run.dir={out/"hydra"}','logging.mode=disabled'],cwd=FC,stdout=log,stderr=subprocess.STDOUT,check=True)
m=json.loads((out/'metrics.json').read_text())['metrics'];print(name,m['add_auc'],m['add_auc_per_body'],'PARTIAL diagnostic only',flush=True)
