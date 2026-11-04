"""Native loader/retarget/support consumer check for the changed episodes of the assembled sprint set
(same consumer code path as scored-priority-delivery-20261103/check_consumers.py). CPU only."""
from pathlib import Path
import json,sys,time,os,subprocess
import numpy as np
O=Path(__file__).resolve().parent;R=Path('/Volumes/E/Code/V2D');D=R/'NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1';W=R/'work/scored-scene-repair-20261028'
sys.path[:0]=[str(D/'runtime'),str(D/'team_cohort_benchmark_20261030')]
from consume_matched_case import consumer,repair_assets,parquet,one,sha
summary=json.loads((O/'SELECTION_SUMMARY.json').read_text())
eps=[int(e) for e,v in summary['episodes'].items() if v['changed']]
only=[int(a) for a in sys.argv[1:]] or eps
results=[]
for ep in only:
    b=O/f'scenes/episode_{ep:06d}/bundle';root=O/'consumer'/f'ep{ep:02d}'
    if root.exists(): import shutil; shutil.rmtree(root)
    root.mkdir(parents=True);seq=f'lead_sprint_ep{ep:02d}_v1';start=time.monotonic();before={p.name:sha(p) for p in b.iterdir() if p.is_file()}
    try:
        runs=consumer(root,b,seq,20.254119873046875,time.monotonic()+240);assets=repair_assets(root);a=parquet(one(root/'loaded'));z=parquet(one(root/'processed'));N=len(np.load(b/'result.npz')['camera_to_world_transform'])
        for k in ['object_body_position','object_body_wxyz','robot_left_frames','robot_right_frames','robot_left_finger_joints','robot_right_finger_joints']:
            v=np.asarray(z[k]);assert len(v)==N and np.isfinite(v).all(),k
        assert np.array_equal(np.asarray(a['object_body_position']),np.asarray(z['object_body_position']))
        support=root/'reconstructed_stage'/f'{seq}_sharpa_wave_support.usda';support.parent.mkdir()
        cmd=[sys.executable,str(W/'baseline/robotic_grounding/scripts/reconstruct_support_surfaces.py'),'--input_dir',str(root/'processed'),'--dataset','ego_recon','--sequence_id',seq,'--robot_name','sharpa_wave','--output',str(support)]
        env=dict(os.environ,PYTHONPATH=str(D/'runtime'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
        with (root/'support.log').open('x') as f:p=subprocess.run(cmd,cwd=W/'baseline/robotic_grounding',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        count=0
        if p.returncode==0 and support.exists():
            try:
                from pxr import Usd
                stage=Usd.Stage.Open(str(support));count=sum(x.GetTypeName() in ['Cylinder','Cube','Mesh'] for x in stage.Traverse())
            except ImportError: count=-1
        assert before=={p.name:sha(p) for p in b.iterdir() if p.is_file()}
        row=dict(episode=ep,frames=N,loader_exit_code=runs[0]['exit_code'],retarget_exit_code=runs[1]['exit_code'],schema='FULL_NATIVE_CONSUMER_PASS',sequence_id=seq,support_exit_code=p.returncode,support_shape_count=count,seconds=time.monotonic()-start,physics_or_policy_tested=False,hand_accuracy_verified=False)
    except Exception as e:
        row=dict(episode=ep,schema='CONSUMER_FAILED',error=repr(e)[:300],seconds=time.monotonic()-start)
    (root/'RECEIPT.json').write_text(json.dumps(row,indent=2)); results.append(row); print(row,flush=True)
(O/'CONSUMER_RESULTS.json').write_text(json.dumps(results,indent=2))
