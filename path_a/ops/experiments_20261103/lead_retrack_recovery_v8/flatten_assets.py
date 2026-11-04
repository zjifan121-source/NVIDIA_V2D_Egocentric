"""Flatten nested optimized_assets/slot_N materials into the case asset root so trimesh resolves the texture
(the container resolves map_Kd relative to the OBJ directory). Re-signs manifest/protocol/FROZEN_BATCH/PACKAGE_SHA256."""
from pathlib import Path
import json, hashlib, shutil, sys, importlib
V=Path(sys.argv[1]); runner=sys.argv[2]
sys.path.insert(0,str(V)); run=importlib.import_module(runner)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
b=read(V/'FROZEN_BATCH.json'); pk=read(V/'PACKAGE_SHA256.json'); changed=[]
for c in b['cases']:
    case=V/'cases'/c['case']; asset=case/'payload/asset'; nested=asset/'optimized_assets'
    if not nested.exists(): continue
    mtls=list(nested.rglob('*.mtl')); assert len(mtls)==1,(c['case'],mtls); mtl=mtls[0]
    for f in mtl.parent.iterdir():
        if f.is_file(): shutil.copy2(f,asset/f.name)
    obj=asset/'mesh.obj'; lines=obj.read_text().splitlines(keepends=True)
    lines=[('mtllib '+mtl.name+'\n') if ln.startswith('mtllib ') else ln for ln in lines]; obj.write_text(''.join(lines))
    shutil.rmtree(nested)
    for k in list(pk):
        if k.startswith(f"cases/{c['case']}/payload/asset/"): pk.pop(k)
    m=read(case/'manifest.json'); m['files']={k:v for k,v in m['files'].items() if not k.startswith('asset/')}
    for f in sorted(asset.rglob('*')):
        if f.is_file(): m['files']['asset/'+f.relative_to(asset).as_posix()]=sha(f); pk[f"cases/{c['case']}/payload/asset/"+f.relative_to(asset).as_posix()]=sha(f)
    put(case/'manifest.json',m)
    p=read(case/'protocol.json'); p['input_manifest_sha256']=sha(case/'manifest.json')
    if p.get('provided_initial_pose'): p['provided_initial_pose']['mesh_sha256']=m['files']['asset/mesh.obj']
    p.setdefault('amendments',[]).append('asset materials flattened to the OBJ directory (texture resolution); geometry bytes unchanged except the mtllib line')
    put(case/'protocol.json',p)
    run.load_package(case/'manifest.json',case/'protocol.json',case/'payload')
    c.update(manifest_sha256=sha(case/'manifest.json'),protocol_sha256=sha(case/'protocol.json'),mesh_sha256=m['files']['asset/mesh.obj'])
    for k in ['manifest.json','protocol.json']: pk[f"cases/{c['case']}/{k}"]=sha(case/k)
    changed.append(c['case']); print('FLATTENED',c['case'],flush=True)
b.setdefault('amendments',[]).append(dict(what='nested optimized_assets materials flattened for cases: '+', '.join(changed),why='Mesh.load in the container found no texture image for the nested layout (image None -> trimesh vertex_colors crash); flat layout is what the successful full-track cases used'))
put(V/'FROZEN_BATCH.json',b); pk['FROZEN_BATCH.json']=sha(V/'FROZEN_BATCH.json'); put(V/'PACKAGE_SHA256.json',pk); print('DONE',len(changed))
