"""Add-only: ep28 r2 scenes, KNOWN_ISSUES under the lead_v3 prefix, manifest and ledger r2. Same checks as publish_overlay_r1.py."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,hashlib,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'work/depth-model-preflight-deps'))
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mode=sys.argv[1] if len(sys.argv)>1 else '--dry-run'
plan=json.loads((W/'UPLOAD_DRY_RUN_overlay_r2.json').read_text());S=Path(plan['stage']);repo=plan['repo'];prefix=plan['prefix']
rows=[dict(local=S/r['relative'],hf=prefix+'/'+r['relative']) for r in plan['files']]
rows+=[dict(local=W/'KNOWN_ISSUES_v3.md',hf='experiments/wentacc/scored20_lead_v3_20261104/KNOWN_ISSUES.md'),dict(local=W/'manifest/CURRENT_MANIFEST.json',hf='experiments/wentacc/lead_current/CURRENT_MANIFEST_r2.json'),dict(local=W/'manifest/VERSION_LEDGER.md',hf='experiments/wentacc/lead_current/VERSION_LEDGER_r2.md')]
api=HfApi();before=api.repo_info(repo,repo_type='dataset',files_metadata=True);all_before={f.rfilename:(f.blob_id,f.size) for f in before.siblings}
clash=[r['hf'] for r in rows if r['hf'] in all_before]; assert not clash,clash[:5]
print('PRECHECK parent',before.sha,'add',len(rows),flush=True)
if mode!='--commit': sys.exit(0)
commit=api.create_commit(repo,repo_type='dataset',parent_commit=before.sha,operations=[CommitOperationAdd(path_in_repo=r['hf'],path_or_fileobj=str(r['local'])) for r in rows],commit_message='Lead v3 manifest r2: ep28 release fix, ep28 and ep33 candidates, KNOWN_ISSUES note; add-only')
print('COMMITTED',commit.oid,flush=True)
def verify(r):
    p=hf_hub_download(repo,r['hf'],repo_type='dataset',revision=commit.oid,local_dir=W/'readback_r2',force_download=True); assert sha(p)==sha(r['local']),r['hf']
with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(verify,rows))
after=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True); actual={f.rfilename:(f.blob_id,f.size) for f in after.siblings}
assert all(actual.get(k)==v for k,v in all_before.items() if k!='.gitattributes'),'an existing file changed'
(W/'UPLOAD_VERIFIED_r2.json').write_text(json.dumps(dict(state='UPLOADED_ALL_FILES_SHA_READBACK_VERIFIED',commit=commit.oid,parent=before.sha,files_added=len(rows),existing_files_unchanged=True,verified_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n'); print('VERIFIED_ALL',len(rows),flush=True)
