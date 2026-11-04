import sys, glob, hashlib, json
import pyarrow.parquet as pq
skip={'raw_motion_file','object_mesh_paths','object_urdf_paths','sequence_id','source_dataset'}
for s in sys.argv[1:]:
    fs=glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet")
    if not fs: print(s,"NO PARQUET"); continue
    t=pq.read_table(fs[0]); h=hashlib.sha256(); cols=[]
    for c in sorted(t.column_names):
        if c in skip or 'path' in c: continue
        v=t.column(c)[0].as_py(); b=json.dumps(v,default=str,sort_keys=True).encode(); h.update(c.encode()); h.update(b)
        cols.append((c,hashlib.sha256(b).hexdigest()[:8]))
    print(s.split('/scenes/')[-1], fs[0].split('sequence_id=')[1].split('/')[0], h.hexdigest()[:16])
    json.dump(dict(cols),open('/tmp/fp_'+hashlib.md5(s.encode()).hexdigest()[:8]+'.json','w'))
