"""Add one known-issues note next to the hands overlay; add-only, parent pinned, read back."""
from pathlib import Path
import sys, hashlib
W=Path(__file__).resolve().parent; sys.path.insert(0,str(W.parents[1]/'work/depth-model-preflight-deps'))
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
repo='rose-tea-cafe/v2d-t3-recon'; path='experiments/wentacc/scored20_lead_v3_hands_r1_20261104/KNOWN_ISSUES.md'; local=W/'KNOWN_ISSUES_hands_r1.md'
api=HfApi(); b=api.repo_info(repo,repo_type='dataset'); assert path not in {f.rfilename for f in b.siblings},'exists'
c=api.create_commit(repo,repo_type='dataset',parent_commit=b.sha,operations=[CommitOperationAdd(path_in_repo=path,path_or_fileobj=str(local))],commit_message='hands_r1 KNOWN_ISSUES: episode_000028 folder holds two parquets, use the r2 scenes; add-only')
p=hf_hub_download(repo,path,repo_type='dataset',revision=c.oid,local_dir=W/'readback_note',force_download=True)
assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashlib.sha256(local.read_bytes()).hexdigest(); print('VERIFIED',c.oid)
