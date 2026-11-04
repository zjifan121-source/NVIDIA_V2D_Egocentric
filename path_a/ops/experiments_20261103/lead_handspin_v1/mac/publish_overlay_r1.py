"""Publish the hands overlay and the current manifest in one add-only commit: every path must be absent on HF, the parent commit is
pinned, every file is read back and SHA-checked, and every file that existed before must be unchanged. Usage: publish_overlay_r1.py --dry-run | --commit"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json,hashlib,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'work/depth-model-preflight-deps'))
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
mode=sys.argv[1] if len(sys.argv)>1 else '--dry-run'; assert mode in ['--dry-run','--commit']
plan=read(W/'UPLOAD_DRY_RUN_overlay_r1.json');S=Path(plan['stage']);repo=plan['repo'];prefix=plan['prefix']
rows=[dict(local=S/r['relative'],hf=prefix+'/'+r['relative'],sha256=r['sha256'],bytes=r['bytes']) for r in plan['files']]
for n,hf in [('manifest/CURRENT_MANIFEST.json','experiments/wentacc/lead_current/CURRENT_MANIFEST_r1.json'),('manifest/VERSION_LEDGER.md','experiments/wentacc/lead_current/VERSION_LEDGER_r1.md')]:
    rows.append(dict(local=W/n,hf=hf,sha256=sha(W/n),bytes=(W/n).stat().st_size))
for r in rows: assert sha(r['local'])==r['sha256'],r['hf']
api=HfApi();before=api.repo_info(repo,repo_type='dataset',files_metadata=True);assert before.private
all_before={f.rfilename:dict(blob_id=f.blob_id,size=f.size) for f in before.siblings}
clash=[r['hf'] for r in rows if r['hf'] in all_before]; assert not clash,('would overwrite',clash[:5])
print('PRECHECK parent',before.sha,'| add',len(rows),'files',sum(r['bytes'] for r in rows),'bytes',flush=True)
if mode=='--dry-run': print('dry run only'); sys.exit(0)
commit=api.create_commit(repo,repo_type='dataset',parent_commit=before.sha,operations=[CommitOperationAdd(path_in_repo=r['hf'],path_or_fileobj=str(r['local'])) for r in rows],
    commit_message='Lead v3 hands overlay r1 + current manifest: false grasps removed, cup spin follows the hand, ep25 board 2 pinch repair; add-only',
    commit_description='Objects unchanged from lead_v3 and the ep28 overlay (asserted). Manifest gives the one current scene per episode. No GT, no training, no official score.')
(W/'UPLOAD_COMMIT_overlay_r1.json').write_text(json.dumps(dict(commit=commit.oid,url=commit.commit_url,parent=before.sha),indent=2)+'\n'); print('COMMITTED',commit.oid,flush=True)
def verify(r):
    p=hf_hub_download(repo,r['hf'],repo_type='dataset',revision=commit.oid,local_dir=W/'readback_overlay_r1',force_download=True)
    assert sha(p)==r['sha256'],r['hf'];return r['hf']
done=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for fut in as_completed([pool.submit(verify,r) for r in rows]):
        done.append(fut.result())
        if len(done)%100==0: print('SHA_READBACK',len(done),'/',len(rows),flush=True)
after=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True)
actual={f.rfilename:dict(blob_id=f.blob_id,size=f.size) for f in after.siblings}
assert all(actual.get(k)==all_before[k] for k in all_before if k!='.gitattributes'),'an existing file changed'
assert all(r['hf'] in actual for r in rows)
(W/'UPLOAD_VERIFIED_overlay_r1.json').write_text(json.dumps(dict(state='UPLOADED_ALL_FILES_SHA_READBACK_VERIFIED',commit=commit.oid,url=commit.commit_url,parent=before.sha,files_added=len(done),existing_files_unchanged=True,verified_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
print('VERIFIED_ALL',len(done),flush=True)
