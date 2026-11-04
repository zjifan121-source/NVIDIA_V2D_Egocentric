"""Row-level merge of packed Track 3 submissions: a base file with whole episodes replaced from other packed files.

Why: a packed file holds every episode's rollout rows plus its mesh payload, and Kaggle lets the team download what it
submitted, so a teammate's winning single-episode swap can be combined without that teammate's run directory. It is also
how a new probe goes on top of a baseline whose run directories are not all ours: pack the probe, then merge its episode.

Checks, all exact (FAIL leaves no output file):
  1. every input has the base's columns and the same row_id order
  2. each swap file equals the base on every row OUTSIDE its episode (otherwise it carries a change this merge would drop)
  3. each swap really changes its episode's poses
  4. the written file, read back: other episodes equal the base, swapped episodes equal their swap file, one commit URL
Differences in code_commit_url are ignored (a repack changes it on every row); --code-commit sets it for the whole file.

Usage: merge_swaps.py --base BASE.parquet --swap EP=FILE [--swap EP=FILE ...] --out OUT.parquet
         [--code-commit URL] [--score base=0.75166 --score EP=SCORE_OF_THAT_SINGLE_SWAP ...]
Writes OUT.parquet and OUT.merge.json. No GT, no scoring, no upload."""
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
POSE = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
IGNORE = {'row_id', 'code_commit_url'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fail(msg): raise SystemExit('MERGE FAILED: ' + msg)

def episodes(df):
    e = df.row_id.str.extract(r'^track_3/e(\d+)/w\d+/f\d+/o\d+$')[0]
    if e.isna().any(): fail(f'row_id not in track_3 format: {df.row_id[e.isna()].iloc[0]}')
    return e.astype(int)

def unequal(a, b):
    return ~((a.values == b.values) | (a.isna().values & b.isna().values))

def diff(a, b, ep):
    out = {}
    for c in a.columns:
        if c in IGNORE: continue
        ne = pd.Series(unequal(a[c], b[c]), index=a.index)
        if ne.any(): out[c] = {int(k): int(v) for k, v in ne.groupby(ep.values).sum().items() if v}
    return out

ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True, type=Path); ap.add_argument('--swap', action='append', required=True)
ap.add_argument('--out', required=True, type=Path); ap.add_argument('--code-commit'); ap.add_argument('--score', action='append', default=[]); a = ap.parse_args()
if a.out.exists(): fail(f'{a.out} already exists; never overwriting')
base = pd.read_parquet(a.base); ep = episodes(base); merged = base.copy(); report = dict(base=dict(path=str(a.base), sha256=sha(a.base)), swaps={})
for item in a.swap:
    k, f = item.split('=', 1); k = int(k)
    if k not in set(ep): fail(f'episode {k} is not in the base')
    s = pd.read_parquet(f)
    if list(s.columns) != list(base.columns): fail(f'{f}: columns differ from the base')
    if len(s) != len(base) or not (s.row_id.values == base.row_id.values).all(): fail(f'{f}: row_id values or order differ from the base')
    d = diff(base, s, ep); stray = {c: {e: n for e, n in v.items() if e != k} for c, v in d.items()}; stray = {c: v for c, v in stray.items() if v}
    if stray: fail(f'{f}: differs from the base outside episode {k}: {json.dumps(stray)[:400]}')
    if not any(c in d for c in POSE): fail(f'{f}: episode {k} poses equal the base, nothing to swap')
    m = (ep == k).values
    old, new = base.checkpoint_sha256[m].unique().tolist(), s.checkpoint_sha256[m].unique().tolist()
    for c in base.columns:
        if c not in IGNORE: merged.loc[m, c] = s.loc[m, c].values
    report['swaps'][str(k)] = dict(path=str(f), sha256=sha(f), rows=int(m.sum()), pose_rows_changed=max(d[c][k] for c in POSE if c in d), meshes_changed=(d.get('object_mesh_b64') or {}).get(k, 0), checkpoint_sha256=dict(base=old, swap=new))
urls = base.code_commit_url.astype(str).str.strip().unique().tolist()
if a.code_commit: merged['code_commit_url'] = a.code_commit
elif len(urls) != 1: fail(f'base has {len(urls)} commit URLs; pass --code-commit')
a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_suffix('.tmp.parquet')
merged.to_parquet(tmp, index=False)
# read back and verify against the inputs, not against the in-memory frame
back = pd.read_parquet(tmp); swapped = [int(k) for k in report['swaps']]; err = []
if list(back.columns) != list(base.columns) or not (back.row_id.values == base.row_id.values).all(): err.append('columns or row order changed on write')
if {str(c): str(t) for c, t in back.dtypes.items()} != {str(c): str(t) for c, t in base.dtypes.items()}: err.append('dtypes changed on write')
d = diff(base, back, ep); out_of_scope = {c: {e: n for e, n in v.items() if e not in swapped} for c, v in d.items()}
if any(out_of_scope.values()): err.append(f'episodes outside the swaps changed: {json.dumps(out_of_scope)[:300]}')
for k in swapped:
    s = pd.read_parquet(report['swaps'][str(k)]['path']); m = (ep == k).values
    bad = {c: int(unequal(back.loc[m, c], s.loc[m, c]).sum()) for c in back.columns if c not in IGNORE}
    if any(bad.values()): err.append(f'episode {k} does not equal its swap file: {bad}')
if back.code_commit_url.astype(str).str.strip().nunique() != 1: err.append('more than one commit URL')
if err:
    tmp.unlink(); fail('; '.join(err))
tmp.rename(a.out)
report.update(out=dict(path=str(a.out), sha256=sha(a.out), rows=len(back), code_commit_url=str(back.code_commit_url.iloc[0])), base_code_commit_urls=urls, episodes_changed=sorted(swapped), created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
if a.score:
    sc = dict(x.split('=', 1) for x in a.score); b = float(sc['base'])
    missing = [k for k in swapped if str(k) not in sc]
    if missing: fail(f'no --score for episode(s) {missing}')
    report['expected_score'] = dict(base=b, single_swap_scores={k: float(sc[str(k)]) for k in swapped}, per_episode_delta={k: round((float(sc[str(k)]) - b) * len(set(ep)), 4) for k in swapped}, expected=round(b + sum(float(sc[str(k)]) - b for k in swapped), 5), note='swaps add because the board score is a mean over episodes')
Path(str(a.out) + '.merge.json').write_text(json.dumps(report, indent=1) + '\n')
print(json.dumps(report, indent=1))
