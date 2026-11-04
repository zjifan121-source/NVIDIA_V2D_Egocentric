"""Host-side variant of flatten_assets.py: cases are named on the command line (already-flattened cases are only
re-signed); frames for load_package come from the host cache via temporary symlinks (the launcher bind-mounts
them); a stale output/ of a failed run is moved aside so the launcher can run the case again.
Usage: flatten_host.py <package> <runner_module> <case> [<case>...]"""
from pathlib import Path
import json, hashlib, shutil, sys, importlib, os, time
V=Path(sys.argv[1]); runner=sys.argv[2]; wanted=sys.argv[3:]
sys.path.insert(0,str(V)); run=importlib.import_module(runner)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
b=read(V/'FROZEN_BATCH.json'); pk=read(V/'PACKAGE_SHA256.json'); changed=[]
for c in b['cases']:
    if c['case'] not in wanted: continue
    case=V/'cases'/c['case']; asset=case/'payload/asset'; nested=asset/'optimized_assets'
    if nested.exists():
        mtls=list(nested.rglob('*.mtl')); assert len(mtls)==1,(c['case'],mtls); mtl=mtls[0]
        for f in mtl.parent.iterdir():
            if f.is_file(): shutil.copy2(f,asset/f.name)
        obj=asset/'mesh.obj'; lines=obj.read_text().splitlines(keepends=True)
        lines=[('mtllib '+mtl.name+'\n') if ln.startswith('mtllib ') else ln for ln in lines]; obj.write_text(''.join(lines))
        shutil.rmtree(nested); print('flattened',c['case'])
    else: print('already flat, re-signing',c['case'])
    for k in list(pk):
        if k.startswith(f"cases/{c['case']}/payload/asset/"): pk.pop(k)
    m=read(case/'manifest.json'); m['files']={k:v for k,v in m['files'].items() if not k.startswith('asset/')}
    for f in sorted(asset.rglob('*')):
        if f.is_file(): m['files']['asset/'+f.relative_to(asset).as_posix()]=sha(f); pk[f"cases/{c['case']}/payload/asset/"+f.relative_to(asset).as_posix()]=sha(f)
    put(case/'manifest.json',m)
    p=read(case/'protocol.json'); p['input_manifest_sha256']=sha(case/'manifest.json')
    if p.get('provided_initial_pose'): p['provided_initial_pose']['mesh_sha256']=m['files']['asset/mesh.obj']
    note='asset materials flattened to the OBJ directory (texture resolution); geometry bytes unchanged except the mtllib line'
    if note not in p.setdefault('amendments',[]): p['amendments'].append(note)
    put(case/'protocol.json',p)
    # frames: temporary symlinks to the host cache for the validation pass
    cache=Path(f"/var/tmp/v2d_scored20_dual_anchor_v1/ep{m['episode']:02d}"); links=[]
    # checked_path resolves symlinks and rejects anything outside payload -> hardlink the frame files in, remove after
    for d in ['rgb','depth','masks']:
        t=case/'payload'/d
        if t.is_dir() and not any(t.iterdir()):
            for src in cache.joinpath(d).rglob('*'):
                if src.is_file():
                    dst=t/src.relative_to(cache/d); dst.parent.mkdir(parents=True,exist_ok=True)
                    try: os.link(src,dst)
                    except OSError: shutil.copy2(src,dst)
                    links.append(dst)
    try: run.load_package(case/'manifest.json',case/'protocol.json',case/'payload')
    finally:
        for f in links: f.unlink()
        for d in ['rgb','depth','masks']:
            for sub in sorted((case/'payload'/d).rglob('*'),reverse=True):
                if sub.is_dir(): sub.rmdir()
    c.update(manifest_sha256=sha(case/'manifest.json'),protocol_sha256=sha(case/'protocol.json'),mesh_sha256=m['files']['asset/mesh.obj'])
    for k in ['manifest.json','protocol.json']: pk[f"cases/{c['case']}/{k}"]=sha(case/k)
    for stale in ['output','GPU.log','LAUNCH.json']:
        s=case/stale
        if s.exists(): s.rename(case/f"{stale}.failed_{int(time.time())}")
    changed.append(c['case']); print('SIGNED',c['case'],flush=True)
b.setdefault('amendments',[]).append(dict(what='nested optimized_assets materials flattened (host) for cases: '+', '.join(changed),why='Mesh.load in the container found no texture image for the nested layout (image None -> trimesh vertex_colors TypeError)'))
put(V/'FROZEN_BATCH.json',b); pk['FROZEN_BATCH.json']=sha(V/'FROZEN_BATCH.json'); put(V/'PACKAGE_SHA256.json',pk); print('DONE',len(changed))
