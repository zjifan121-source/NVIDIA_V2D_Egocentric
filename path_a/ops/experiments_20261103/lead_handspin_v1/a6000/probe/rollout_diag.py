import json,sys
import numpy as np, pyarrow.parquet as pq
d=sys.argv[1]; ep=int(sys.argv[2])
m=json.load(open(f'{d}/results/episode_{ep:06d}.json'))['metrics']
print({k:m[k] for k in m if k in ('add_auc','add_auc_per_body','mppe_cm','step_count','object_position_error_per_body_m','chord_sr','maniptrans_sr','spider_sr_uncentered')})
print({k:v for k,v in m.items() if 'termination' in k or 'gated' in k})
t=pq.read_table(f'{d}/results/episode_{ep:06d}.parquet').to_pandas()
print(t.columns.tolist()[:8],len(t))
for oid,g in t.groupby('object_id'):
    g=g.sort_values('step'); a=g[['achieved_position_x','achieved_position_y','achieved_position_z']].values; r=g[['reference_position_x','reference_position_y','reference_position_z']].values
    e=np.linalg.norm(a-r,axis=1)*100
    idx=np.linspace(0,len(e)-1,12).astype(int)
    print(g.object_body_name.iloc[0][-22:],'n',len(g),'valid',float(g.environment_valid.mean()),'err cm at steps',[(int(g.step.iloc[i]),round(float(e[i]),1)) for i in idx])
    print('   z achieved',np.round(a[idx,2],3).tolist()); print('   z reference',np.round(r[idx,2],3).tolist())
