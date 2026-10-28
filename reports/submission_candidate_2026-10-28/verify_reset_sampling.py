"""Measure the actual native reset kernel under baseline and early-pickup configs."""
from pathlib import Path
import json,numpy as np,warp as wp
wp.config.kernel_cache_dir='/tmp/v2d_reset_sampling_cache'
from flash_chord.lifecycle.reset import sample_reference_reset
W=16384;rows=[]
for label,prob,threshold,voc in [('baseline_strong',.1,.1,1.),('early_strong',.5,1.1,1.),('baseline_weak',.1,.1,.05),('early_weak',.5,1.1,.05)]:
 mask=wp.ones(W,dtype=wp.int32,device='cpu');count=wp.zeros(W,dtype=wp.int32,device='cpu');frame=wp.zeros(W,dtype=wp.int32,device='cpu');fingers=wp.zeros(W,dtype=wp.float32,device='cpu');steps=wp.zeros(W,dtype=wp.int32,device='cpu');applied=wp.zeros(W,dtype=wp.float32,device='cpu')
 wp.launch(sample_reference_reset,dim=W,device='cpu',inputs=[mask,348,20,0,wp.array([prob],dtype=wp.float32,device='cpu'),threshold,.7,42,wp.array([voc],dtype=wp.float32,device='cpu'),count],outputs=[frame,fingers,steps,applied]);fraction=float(np.mean(frame.numpy()==0));expected=prob+(1-prob)/347 if voc<threshold else 1/347;assert abs(fraction-expected)<.02
 row=dict(label=label,worlds=W,first_frame_fraction=fraction,expected_fraction=expected);rows.append(row);print(json.dumps(row),flush=True)
Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/reset_sampling_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
