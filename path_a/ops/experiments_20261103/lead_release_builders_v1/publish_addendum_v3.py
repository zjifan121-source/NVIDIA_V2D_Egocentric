"""Append an add-only batch (plan file given with --plan) to the EXISTING lead_sprint_v2 prefix: every new path must be
absent on HF, parent-commit check, SHA read-back of every added file, and a check that every file that existed before is
unchanged. Usage: publish_addendum_v3.py --plan UPLOAD_DRY_RUN_ADDENDUM_V3.json [--dry-run|--commit]"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json,hashlib,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'work/depth-model-preflight-deps'))
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
mode='--commit' if '--commit' in sys.argv else '--dry-run'
PLAN=Path(sys.argv[sys.argv.index('--plan')+1]); TAG=PLAN.stem.replace('UPLOAD_DRY_RUN_','')
plan=read(PLAN);S=Path(plan['stage']);repo=plan['repo'];prefix=plan['prefix'];rows=plan['files']
for r in rows:assert sha(S/r['relative'])==r['sha256'] and (S/r['relative']).stat().st_size==r['bytes'],r['relative']
api=HfApi();before=api.repo_info(repo,repo_type='dataset',files_metadata=True);assert before.private
all_before={f.rfilename:dict(blob_id=f.blob_id,size=f.size) for f in before.siblings}
inside={k for k in all_before if k.startswith(prefix+'/')}; assert len(inside)==plan['prefix_files_expected'],f"prefix has {len(inside)} files, expected {plan['prefix_files_expected']}"
clash=[r['relative'] for r in rows if prefix+'/'+r['relative'] in inside]; assert not clash,('would overwrite',clash[:5])
pre=dict(state='REMOTE_PRECHECK_PASS',parent=before.sha,expected_parent=plan['expected_parent'],parent_moved=before.sha!=plan['expected_parent'],prefix=prefix,prefix_files_before=len(inside),files_to_add=len(rows),bytes=plan['bytes'],utc=datetime.now(timezone.utc).isoformat())
(W/f'COMMIT_PRECHECK_{TAG}.json').write_text(json.dumps(pre,indent=2)+'\n'); print('PRECHECK parent',before.sha,'| add',len(rows),'files',plan['bytes'],'bytes | prefix files before',len(inside),flush=True)
if mode=='--dry-run': print('dry run only'); sys.exit(0)
commit=api.create_commit(repo,repo_type='dataset',parent_commit=before.sha,
    operations=[CommitOperationAdd(path_in_repo=prefix+'/'+r['relative'],path_or_fileobj=str(S/r['relative'])) for r in rows],
    commit_message=plan['commit_message'],commit_description=plan['commit_description'])
(W/f'UPLOAD_COMMIT_{TAG}.json').write_text(json.dumps(dict(commit=commit.oid,url=commit.commit_url,parent=before.sha,prefix=prefix),indent=2)+'\n'); print('COMMITTED',commit.oid,flush=True)
def verify(r):
    p=hf_hub_download(repo,prefix+'/'+r['relative'],repo_type='dataset',revision=commit.oid,local_dir=W/f'readback_{TAG}',force_download=True)
    assert sha(p)==r['sha256'] and Path(p).stat().st_size==r['bytes'],r['relative'];return r
checked=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for fut in as_completed([pool.submit(verify,r) for r in rows]):
        checked.append(fut.result())
        if len(checked)%40==0:print('SHA_READBACK',len(checked),'/',len(rows),flush=True)
after=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True)
actual={f.rfilename:dict(blob_id=f.blob_id,size=f.size) for f in after.siblings}
assert all(actual.get(k)==all_before[k] for k in all_before if k!='.gitattributes'),'an existing file changed'
assert {k for k in actual if k.startswith(prefix+'/')}==inside|{prefix+'/'+r['relative'] for r in rows}
(W/f'UPLOAD_VERIFIED_{TAG}.json').write_text(json.dumps(dict(state='UPLOADED_ALL_FILES_SHA_READBACK_VERIFIED',commit=commit.oid,url=commit.commit_url,parent=before.sha,repo=repo,prefix=prefix,
    tree_url=f'https://huggingface.co/datasets/{repo}/tree/{commit.oid}/{prefix}',files_added=len(checked),bytes=plan['bytes'],episodes=plan.get('episodes'),existing_files_unchanged=True,
    verified_utc=datetime.now(timezone.utc).isoformat(),sha256=sorted(checked,key=lambda r:r['relative'])),indent=2)+'\n')
print('VERIFIED_ALL',len(checked),flush=True)
