"""Publish the staged lead sprint v1 release to a NEW prefix of the team dataset (append-only commit), then read every
file back and verify SHA-256 + size, and verify that nothing outside the new prefix changed. Same discipline as the
fb11 publish script. Uses the local Hugging Face login of this machine (token never printed or copied).
Usage: publish.py --dry-run   (remote precheck only)   |   publish.py --commit"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json,hashlib,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'work/depth-model-preflight-deps'))
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def put(n,x):
    with (W/n).open('x') as f:json.dump(x,f,indent=2)
mode=sys.argv[1] if len(sys.argv)>1 else '--dry-run'; assert mode in ['--dry-run','--commit']
plan=read(W/'UPLOAD_DRY_RUN.json');S=Path(plan['stage']);repo=plan['repo'];prefix=plan['prefix'];rows=plan['files']
for r in rows:assert sha(S/r['relative'])==r['sha256'] and (S/r['relative']).stat().st_size==r['bytes'],r['relative']
api=HfApi();before=api.repo_info(repo,repo_type='dataset',files_metadata=True);assert before.private
all_before={f.rfilename:dict(blob_id=f.blob_id,size=f.size) for f in before.siblings}
inside=[k for k in all_before if k.startswith(prefix+'/')]
assert not inside,f'Target prefix already exists on HF ({len(inside)} files); refusing to overwrite'
v1_prefix='experiments/wentacc/scored20_lead_sprint_v1_20261103/'
fb11_files=[k for k in all_before if k.startswith(v1_prefix)];assert len(fb11_files)==245,'lead_sprint_v1 prefix missing or changed'
pre=dict(state='REMOTE_PRECHECK_PASS',parent=before.sha,expected_parent=plan.get('expected_parent'),parent_moved_since_dry_run=before.sha!=plan.get('expected_parent'),
         private=True,prefix=prefix,prefix_absent=True,fb11_prefix_files=len(fb11_files),repo_files_before=len(all_before),files_to_add=len(rows),bytes=plan['bytes'],utc=datetime.now(timezone.utc).isoformat())
(W/'COMMIT_PRECHECK.json').write_text(json.dumps(pre,indent=2)+'\n')
print('PRECHECK parent',before.sha,'| files to add',len(rows),'bytes',plan['bytes'],flush=True)
if mode=='--dry-run':
    for r in rows[:12]:print('ADD',r['relative'],r['bytes'])
    print('... total',len(rows),'files; no commit (dry run)');sys.exit(0)
commit=api.create_commit(repo,repo_type='dataset',parent_commit=before.sha,
    operations=[CommitOperationAdd(path_in_repo=prefix+'/'+r['relative'],path_or_fileobj=str(S/r['relative'])) for r in rows],
    commit_message='Lead sprint v2: seated support_v1 for all 20 Scored episodes + raw-hands re-attach scenes for ep05/ep32/ep33 (objects = lead_sprint_v1); new prefix',
    commit_description=('New independent prefix. Bundles byte-identical to lead_sprint_v1 (9f8d352e). support_v1 of all 20 episodes seated at frame 0 (haisenk wseat rule re-implemented in bundle world; ep10/ep38 supports carried from the historical scenes); ep05/ep32/ep33: team raw hands re-attached to the new spoon tracks with the team loader (haisenk recipe), built scenes with seated support included. No GT, no training, no official score. Append-only: no existing file modified or deleted.'))
put('UPLOAD_COMMIT.json',dict(commit=commit.oid,url=commit.commit_url,parent=before.sha,prefix=prefix));print('COMMITTED',commit.oid,flush=True)
def verify(r):
    p=hf_hub_download(repo,prefix+'/'+r['relative'],repo_type='dataset',revision=commit.oid,local_dir=W/'readback',force_download=True)
    assert sha(p)==r['sha256'] and Path(p).stat().st_size==r['bytes'],r['relative'];return r
checked=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(verify,r) for r in rows]):
        checked.append(future.result())
        if len(checked)%40==0:print('SHA_READBACK',len(checked),'/',len(rows),flush=True)
after=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True);assert after.private
actual={f.rfilename:dict(blob_id=f.blob_id,size=f.size) for f in after.siblings}
untouched=[k for k in all_before if k!='.gitattributes']
assert all(actual.get(k)==all_before[k] for k in untouched),'a file outside the new prefix changed'
assert {k for k in actual if k.startswith(prefix+'/')}=={prefix+'/'+r['relative'] for r in rows}
put('UPLOAD_VERIFIED.json',dict(state='UPLOADED_ALL_FILES_SHA_READBACK_VERIFIED',commit=commit.oid,url=commit.commit_url,parent=before.sha,repo=repo,prefix=prefix,private=True,
    tree_url=f'https://huggingface.co/datasets/{repo}/tree/{commit.oid}/{prefix}',base_revision=plan['base_revision'],
    episodes=20,
    files=len(checked),videos=sum(r['relative'].endswith('.mp4') for r in checked),bytes=plan['bytes'],
    untouched_outside_prefix_files=len(untouched),gitattributes_changed=actual.get('.gitattributes')!=all_before.get('.gitattributes'),previous_commits_preserved=True,
    official_AUC=None,policy_score=None,verified_utc=datetime.now(timezone.utc).isoformat(),sha256=sorted(checked,key=lambda r:r['relative'])))
print('VERIFIED_ALL',len(checked),'videos',sum(r['relative'].endswith('.mp4') for r in checked),flush=True)
