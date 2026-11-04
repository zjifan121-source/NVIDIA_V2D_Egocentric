"""Read-only watcher of the team's Track 3 submissions and the per-episode incumbent table for the final pack.

Each run lists all six boards, downloads every new submitted file (no quota used, nothing is uploaded), diffs it row by
row against bob1 and classifies it:
  single_swap   exactly one episode differs; its score gives that episode's delta (score - bob1) * 20
  merge_known   several episodes differ, each equal to an already verified single swap; a check on additivity
  multi         several episodes differ in ways not seen alone; its score cannot be split per episode
  same_as_base  no episode differs
The incumbents are the combination of single swaps with the lowest predicted rank-sum, among swaps measured on all six
boards (directly, or by subtracting known episodes from a merged file). A swap missing a board is listed but not eligible.

Writes under LEAD_BOARD_DIR (default /root/probes/incumbents): raw/<ref>/<ref>.raw, classified/<ref>.json,
INCUMBENTS.json, INCUMBENTS.md, EVENTS.log. No GT, no scoring, no upload calls.
Usage: board_watch.py   (one pass; run it in a loop to keep watching)"""
import sys, os, re, json, hashlib, glob, time, itertools
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
KIT = os.environ.get('LEAD_KIT', '/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit'); sys.path.insert(0, KIT)
from kaggle.api.kaggle_api_extended import KaggleApi
W = Path(os.environ.get('LEAD_BOARD_DIR', '/root/probes/incumbents'))
BOB1_PLAN = os.environ.get('LEAD_BOB1_PLAN', '/root/bob1_meta/experiments/edmond/bob1_submission_20261103/candidate_bob1.plan.json')
BOB1_REF = 57934736; N_EP = 20
BOARDS = ['auc', 'sp-sr', 'mp-sr', 'rpe', 'mppe', 'cd-o']; SLUG = 'v2d-challenge-track3-'
POSE = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
TRACKED = POSE + ['checkpoint_sha256', 'object_mesh_b64']
SLOW_S = 900; TEAM = 'Gravity Aligned'; HIGHER_BETTER = {'auc': True, 'sp-sr': True, 'mp-sr': True, 'rpe': False, 'mppe': False, 'cd-o': False}
# From MESH_SIZE_AUDIT.md. These apply if our run replaces the incumbent of that episode; none is in the base today.
MESH_CAVEAT = {25: 'our lead_v3 boards are 0.9 and 3.5 cm thick, base 2.4 and 2.4', 33: 'our lead_v3 blue cup is 18.7 cm, the other five episodes 17.1; base spoon 27.8 cm is short, ours 31.8', 19: 'our lead_v3 dust pan is 5-10% larger than the base'}
HANDOVER = [('ep6 spoon', '28.2 x 5.0 x 2.9 cm against a class median of 30.6 x 6.3 x 3.2: area -29%, volume -43%. Largest outlier. Our lead_v3 spoon, 31.5 x 6.1 x 3.4, is in class, and spoons have no public scan, so it is a candidate for re-tracking and retraining.'),
            ('ep10 beige cup', '14.2 cm long. T4 made the ep8 cup 18.9 cm and CD-O improved by 0.0108. Same physical cup. T6 is described as cup +2.8 cm to the depth-measured table, so the length comes from observation, not a scan.'),
            ('ep15 spoon', '26.3 cm long, -14%. Our lead_v3 is short too, 25.8 cm, so we have no candidate.'),
            ('ep35 spoon', 'Width +10%, area +11%, volume +23%.'), ('ep36 spoon', 'Volume +20%.'), ('ep37 spoon', 'Width +22%, thickness -13%, area +11%.'),
            ('ep15 rack', 'Area +18%; thickness +12%. The four other episodes share one mesh, 47.5 x 17.0 x 11.5 cm.'),
            ('ep20 pitcher', 'Ken\'s base pitcher is 38.8 x 21.0 cm; the fb11 and lead_v3 pitcher is 43.0 x 27.8 cm, 33% wider. Ken\'s fb11-based ep20 runs (P2, ep20 about -0.13) use the wider one, which may be part of the loss. Unverified.')]
CODE_NOTES = ['ep1 T2 run o_ep1_hold_s44 is the 100M final checkpoint, policy_100003840, not 95M.',
              'ep5 T3 run o_ep5_attcom_s42 is the 95M checkpoint, policy_95006720.',
              "Ken's training code was team repo edm_training at 1a6bae7b (gpu_queue.sh), not de66fea3.",
              'The team is split per episode, and each owner is responsible for the runs and code behind their episodes in the final commit. Owner-supplied for the final commit: Ken (ep1, ep3, ep5, ep8), Edmond (his episodes). We merge their already-scored Kaggle files and ask nothing of them.',
              "Ken's pack cited code_commit_url 1689cf63. His hand_fixes scripts for ep1 and ep5 are in the delivered package under code/tools/track3/hand_fixes/.",
              'Handoff scene parquets point at /tmp/haisenk_v2d/. Run relocate_scene.py after untar. bundle_cleaned symlinks are dead and not needed.',
              'T2 and T3 kept our objects and our support_v1: ep1 on fb11 with hand hold windows, ep5 on lead_sprint_v2 with seated support, re-attached hands and pot COM moved 8.09 cm to the shell centroid. Neither used a depth-measured table; that came with T5 and T6.',
              'Ken ran with object termination 0.1 m / 0.5 rad (the experiment default), not the 0.3 / 1.5 of the DEF recipe.',
              ]
RUN_DIRS = [('ep1 T2', 'verified', '/root/ken_handoff/ep1_ep5_20261104/episode_000001', '/root/ken_handoff/unpacked/w_hold_ep1', 'policy_100003840, sha 969b9cef'),
            ('ep5 T3', 'verified', '/root/ken_handoff/ep1_ep5_20261104/episode_000005', '/root/ken_handoff/unpacked/w7_attcom_ep5', 'policy_95006720, sha a408e294'),
            ('ep5 T8', 'row merge; owner-supplied for the final commit: Ken', '', '', 'r_ep5_rl_K_s43@95M'),
            ('ep3 T5', 'row merge; owner-supplied for the final commit: Ken', '', '', 'h_ep3_sc_K_s42@95M, sha 45c4736b'),
            ('ep8 T6', 'row merge; owner-supplied for the final commit: Ken', '', '', 'h_ep8_sc_K_s42@100M, sha 0ff463d1')]
EXCLUDED = {57971553: 'rejected by the lead: a net +1 place from CD-O at a real AUC and RPE cost; the right path is retraining ep20 on our pitcher'}  # P2
RATIO = {'ken': 0.81, 'wentao': 0.86}
RULES = ['Merging a candidate into the base needs a six-board probe, judged on rank-sum: no board may regress by more than the gain elsewhere.',
         'No mesh is copied from another episode or from the base into our scenes, because of possible scan provenance. The only legal replacement is our own reconstruction selected by the GT-free proxy.',
         'No new mesh on old-coordinate poses (work/handover/08_operations_and_access.md: 不用新mesh配旧坐标pose). A better mesh must be re-tracked and retrained, as T4 did. Mesh-only swaps are closed, whatever their CD-O gain.',
         'Uploads are paused (the user, 2026-11-04): no board gets a new submission until the user lifts this. Kaggle rejected a 21st AUC upload, so that board has a real limit. When uploads resume, one identical file goes on all six boards.',
         'Only sources measured on all six boards can enter the final. The rank-sum picker decides, and ep8 stays T4 until T6 has its five boards.']

def now(): return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def retry(fn, *a, n=3, **k):
    for i in range(n):
        try: return fn(*a, **k)
        except Exception:
            if i == n - 1: raise
            time.sleep(8)

def leaderboards(api):
    out, cache = {}, W / 'leaderboards.json'
    if cache.is_file() and time.time() - cache.stat().st_mtime < SLOW_S:  # the A100 link to Kaggle is slow; standings move slowly
        return json.loads(cache.read_text())['boards']
    try:
        for b in BOARDS:
            rows = retry(api.competition_leaderboard_view, SLUG + b, page_size=1000) or []
            out[b] = [[str(r.team_name), float(r.score)] for r in rows if r is not None and str(getattr(r, 'score', '') or '').strip()]
        atomic(cache, json.dumps(dict(read_utc=now(), boards=out)) + '\n'); return out
    except Exception as exc:
        print('leaderboard read failed', repr(exc)[:160], flush=True)
        return json.loads(cache.read_text())['boards'] if cache.is_file() else {}

def rank_on(board, score, lb):
    """1 + other teams strictly better. Ties share the better rank, so this is the best case when many teams tie."""
    hi = HIGHER_BETTER[board]
    return 1 + sum(1 for t, v in lb.get(board, []) if t != TEAM and (v > score if hi else v < score))

def st(x): return str(getattr(x, 'status', '')).split('.')[-1]

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def atomic(path, text):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(text); tmp.rename(path)

def event(msg):
    with open(W / 'EVENTS.log', 'a') as f: f.write(f'{now()} {msg}\n')
    print('EVENT', msg, flush=True)

def load(ref):
    f = W / 'raw' / str(ref) / f'{ref}.raw'
    return pd.read_parquet(f), f

def fetch(api, ref):
    d = W / 'raw' / str(ref); f = d / f'{ref}.raw'
    if f.is_file(): return f
    d.mkdir(parents=True, exist_ok=True)
    api.competition_download_submission(ref, path=str(d), quiet=True)
    if not f.is_file():
        got = [p for p in d.iterdir() if p.is_file()]
        if len(got) == 1: got[0].rename(f)
    return f

def episodes(df): return df.row_id.str.extract(r'^track_3/e(\d+)/w\d+/f\d+/o\d+$')[0].astype(int)

def rows_sha(df, mask):
    h = hashlib.sha256()
    for c in POSE: h.update(df.loc[mask, c].to_numpy('float64').tobytes())
    for c in ('checkpoint_sha256', 'object_mesh_b64'): h.update('|'.join(df.loc[mask, c].fillna('').astype(str)).encode())
    return h.hexdigest()

def classify(ref, base, ep):
    df, f = load(ref)
    if set(df.row_id) != set(base.row_id): return dict(ref=ref, klass='different_rows', file_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    df = df.set_index('row_id').reindex(base.row_id).reset_index()
    per = {}
    for c in TRACKED:
        a, b = base[c], df[c]; ne = ~((a.values == b.values) | (a.isna().values & b.isna().values))
        for e, n in pd.Series(ne).groupby(ep.values).sum().items():
            if n: per.setdefault(int(e), {})[c] = int(n)
    changed = sorted(per); out = dict(ref=ref, file_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), changed=changed, episodes={})
    for e in changed:
        m = (ep == e).values
        out['episodes'][str(e)] = dict(pose_rows=max((per[e].get(c, 0) for c in POSE), default=0), rows=int(m.sum()), meshes_changed=per[e].get('object_mesh_b64', 0),
                                       checkpoint_sha256=df.checkpoint_sha256[m].iloc[0], rows_sha256=rows_sha(df, m))
    return out

def main():
    W.mkdir(parents=True, exist_ok=True); (W / 'classified').mkdir(exist_ok=True)
    api = KaggleApi(); api.authenticate()
    boards, quota = {}, {}
    for b in BOARDS:
        boards[b] = retry(api.competition_submissions, SLUG + b, page_size=60) or []
    qf = W / 'quota.json'
    if qf.is_file() and time.time() - qf.stat().st_mtime < SLOW_S: quota = json.loads(qf.read_text())
    else:
        for b in BOARDS:
            try:
                l = api.competition_get_submission_limits(SLUG + b); quota[b] = dict(left=getattr(l, 'num_allowed_now', None), used=getattr(l, 'num_today', None))
            except Exception: quota[b] = dict(left=None, used=None)
        if all(v['left'] is not None for v in quota.values()): atomic(qf, json.dumps(quota) + '\n')
    meta = {}
    for x in boards['auc']:
        ref = int(getattr(x, 'ref', 0))
        if ref < BOB1_REF: continue
        meta[ref] = dict(ref=ref, date=str(getattr(x, 'date', ''))[:16], by=getattr(x, 'submitted_by_ref', None), status=st(x), auc=num(getattr(x, 'public_score', None)), description=str(getattr(x, 'description', '') or ''))
    for b in BOARDS[1:-1]:  # a file can go to other boards without AUC; the CD-O board holds a derived file, so never classify from it
        for x in boards[b]:
            ref = int(getattr(x, 'ref', 0)); desc = str(getattr(x, 'description', '') or '')
            if ref < BOB1_REF or any(r['description'] == desc for r in meta.values()): continue
            meta[ref] = dict(ref=ref, date=str(getattr(x, 'date', ''))[:16], by=getattr(x, 'submitted_by_ref', None), status=st(x), auc=None, description=desc)
    for r in meta.values():
        r['pending_boards'] = [b for b in BOARDS if any(str(getattr(y, 'description', '') or '') == r['description'] and st(y).upper() == 'PENDING' for y in boards[b])]
        r['boards'] = {}
        for b in BOARDS:
            hit = [y for y in boards[b] if str(getattr(y, 'description', '') or '') == r['description'] and num(getattr(y, 'public_score', None)) is not None]
            if hit: r['boards'][b] = num(getattr(hit[0], 'public_score', None))
    fetch(api, BOB1_REF); base, _ = load(BOB1_REF); base = base.sort_values('row_id', kind='stable').reset_index(drop=True); ep = episodes(base)
    seen_f = W / 'seen.json'; seen = json.loads(seen_f.read_text()) if seen_f.is_file() else {}
    for ref, r in sorted(meta.items()):
        cf = W / 'classified' / f'{ref}.json'
        if r['status'].upper() == 'ERROR': continue
        if not cf.is_file():
            try:
                fetch(api, ref); atomic(cf, json.dumps(classify(ref, base, ep), indent=1) + '\n')
            except Exception as exc:
                print('fetch/classify failed', ref, repr(exc)[:160], flush=True); continue
        r.update({k: v for k, v in json.loads(cf.read_text()).items() if k != 'ref'})
    groups = {}
    for r in meta.values():
        if r.get('episodes') is None: continue
        groups.setdefault(tuple(sorted((e, v['rows_sha256']) for e, v in r['episodes'].items())), []).append(r)
    merged_boards = {}
    for sig, rs in groups.items():
        union = {}
        for r in sorted(rs, key=lambda z: z['ref']): union.update(r['boards'])
        for r in rs: r['boards_same_rollouts'] = union
        for key in sig: merged_boards[key] = max(merged_boards.get(key, 0), len(union))
    singles = {}
    base_auc = meta[BOB1_REF]['auc']
    for ref, r in sorted(meta.items()):
        ch = r.get('changed')
        if ch is None: continue
        r['klass'] = r.get('klass') or ('same_as_base' if not ch else 'single_swap' if len(ch) == 1 else None)
        if r['klass'] == 'single_swap':
            e = ch[0]; d = None if r['auc'] is None else (r['auc'] - base_auc) * N_EP; r['delta_vs_bob1'] = d
            m = re.search(r'<-\s*(\S+)', r['description']); o = re.search(r'own-ref ([0-9.]+)', r['description'])
            r['run'] = m.group(1) if m else None; r['own_ref'] = float(o.group(1)) if o else None
            singles.setdefault(e, []).append(r)
    for ref, r in sorted(meta.items()):
        ch = r.get('changed')
        if ch is None or r.get('klass'): continue
        known = all(any(s['episodes'][str(e)]['rows_sha256'] == r['episodes'][str(e)]['rows_sha256'] for s in singles.get(e, [])) for e in ch)
        r['klass'] = 'merge_known' if known else 'multi'
        if known and r['auc'] is not None:
            parts = [next(s for s in singles[e] if s['episodes'][str(e)]['rows_sha256'] == r['episodes'][str(e)]['rows_sha256']) for e in ch]
            if all(p['auc'] is not None for p in parts): r['expected_auc'] = base_auc + sum(p['auc'] - base_auc for p in parts)
    plan = {int(k): v for k, v in json.loads(Path(BOB1_PLAN).read_text()).items()}
    bob = meta[BOB1_REF]['boards_same_rollouts']
    tag = lambda r: (re.search(r'\b(T\d+|P\d+|bob\d+)\b', r['description']) or re.search(r'(.+)', str(r['ref']))).group(1)
    src = {}
    for sig, rs in groups.items():
        if len(sig) != 1: continue
        e, h = int(sig[0][0]), sig[0][1]; u = rs[0]['boards_same_rollouts']; head = next((r for r in sorted(rs, key=lambda z: z['ref']) if '<-' in r['description'] or r['auc'] is not None), rs[0])
        src.setdefault(e, []).append(dict(key=h, ref=head['ref'], tag=tag(head), by=head['by'], run=head.get('run'), own_ref=head.get('own_ref'), description=head['description'], info=head['episodes'][str(e)], deltas={b: u[b] - bob[b] for b in u if b in bob}, derived=[], refs=[r['ref'] for r in rs]))
    for v in src.values():
        for c in v:
            if c['info']['meshes_changed'] == 0 and 'cd-o' not in c['deltas']:  # CD-O reads only the meshes, so an unchanged mesh is an exact zero
                c['deltas']['cd-o'] = 0.0; c['derived'].append('cd-o')
    for _ in range(3):  # a merged file gives one episode's delta when the other episodes are known
        for sig, rs in groups.items():
            if len(sig) < 2: continue
            u = rs[0]['boards_same_rollouts']
            for b in BOARDS:
                if b not in u or b not in bob: continue
                parts = [next((c for c in src.get(int(e), []) if c['key'] == h), None) for e, h in sig]
                if any(c is None for c in parts): continue
                miss = [c for c in parts if b not in c['deltas']]
                if len(miss) == 1:
                    miss[0]['deltas'][b] = (u[b] - bob[b]) - sum(c['deltas'][b] for c in parts if c is not miss[0]); miss[0]['derived'].append(b)
    lb = leaderboards(api)
    def total(chosen):
        t = dict(bob)
        for c in chosen:
            for b, d in c['deltas'].items(): t[b] += d
        return t
    def rsum(t): return sum(rank_on(b, t[b], lb) for b in BOARDS) if lb else None
    full = {e: [c for c in v if all(b in c['deltas'] for b in BOARDS) and not (set(c['refs']) & set(EXCLUDED))] for e, v in src.items()}
    eps = [e for e, v in full.items() if v]; best = None; n = 1
    for e in eps: n *= len(full[e]) + 1
    if lb and n <= 200000:
        for combo in itertools.product(*[[None] + full[e] for e in eps]):
            chosen = [c for c in combo if c]; t = total(chosen); key = (rsum(t), -t['auc'])
            if best is None or key < best[0]: best = (key, chosen, t)
    selected = {int(next(e for e, v in src.items() if c in v)): c for c in best[1]} if best else {}
    bm = (ep.astype(int) >= 0)
    table = {}
    for e in sorted(plan):
        m = (ep == e).values; p = plan[e]
        row = dict(episode=e, source='bob1', submission=BOB1_REF, run=p.get('name'), scene_source=p.get('source'), own_ref=round(float(p['tracking']), 4), delta_vs_bob1=0.0, checkpoint_sha256=base.checkpoint_sha256[m].iloc[0], rows_sha256=rows_sha(base, m), boards_measured='all six (bob1)', meshes_changed=0, tried=[])
        for s in singles.get(e, []):
            row['tried'].append(dict(ref=s['ref'], by=s['by'], run=s.get('run'), auc=s['auc'], delta=None if s['delta_vs_bob1'] is None else round(s['delta_vs_bob1'], 4), status=s['status'], boards=sorted(s['boards'])))
        if e in selected:
            c = selected[e]; x = c['info']; d = c['deltas']['auc'] * N_EP
            row.update(source=f"{c['tag']} {c['by']}", submission=c['ref'], run=c['run'], scene_source=None, own_ref=c['own_ref'], delta_vs_bob1=round(d, 4), checkpoint_sha256=x['checkpoint_sha256'], rows_sha256=x['rows_sha256'], meshes_changed=x['meshes_changed'], boards_measured='all six' + (' (' + ', '.join(c['derived']) + ' derived, not read from a board)' if c['derived'] else ''), description=c['description'])
        row['mesh_caveat'] = MESH_CAVEAT.get(e, '')
        table[e] = row
    swaps = [t for t in table.values() if t['submission'] != BOB1_REF]
    expected = round(base_auc + sum(t['delta_vs_bob1'] for t in swaps) / N_EP, 5)
    pending = [dict(episode=e, tag=c['tag'], ref=c['ref'], missing=[b for b in BOARDS if b not in c['deltas']], auc_delta_ep=None if 'auc' not in c['deltas'] else round(c['deltas']['auc'] * N_EP, 4)) for e, v in sorted(src.items()) for c in v if any(b not in c['deltas'] for b in BOARDS)]
    cmd = '/root/venv-kit/bin/python /root/v2d/tools_lead/merge_swaps.py --base ' + str(W / 'raw' / str(BOB1_REF) / f'{BOB1_REF}.raw') + ''.join(f" --swap {t['episode']}={W / 'raw' / str(t['submission']) / (str(t['submission']) + '.raw')}" for t in swaps) + ' --out <OUT.parquet> --code-commit <URL> --score base=' + str(base_auc) + ''.join(f" --score {t['episode']}={meta[t['submission']]['auc']}" for t in swaps)
    best_measured = max((r for r in meta.values() if r['auc'] is not None), key=lambda r: r['auc'])
    ranks = {}
    if lb:
        ours = {b: next((v for t, v in lb.get(b, []) if t == TEAM), None) for b in BOARDS}
        cur = {b: (rank_on(b, ours[b], lb) if ours[b] is not None else None) for b in BOARDS}
        ranks = dict(teams={b: len(lb.get(b, [])) for b in BOARDS}, team_best=ours, team_rank=cur, team_rank_sum=sum(v for v in cur.values() if v is not None) if all(v is not None for v in cur.values()) else None, groups=[])
        ranks['gaps'] = {}
        for b in BOARDS:
            if ours[b] is None: continue
            hi = HIGHER_BETTER[b]; others = [v for t, v in lb[b] if t != TEAM]
            ahead = sorted((v for v in others if (v > ours[b] if hi else v < ours[b])), key=lambda v: abs(v - ours[b]))
            behind = sorted((v for v in others if (v < ours[b] if hi else v > ours[b])), key=lambda v: abs(v - ours[b]))
            ranks['gaps'][b] = dict(score=ours[b], rank=cur[b], next_ahead=ahead[:3], next_behind=behind[:1])
        for sig, rs in sorted(groups.items(), key=lambda kv: -max(r['ref'] for r in kv[1])):
            u = rs[0]['boards_same_rollouts']
            if len(u) < 2: continue
            rk = {b: rank_on(b, u[b], lb) for b in BOARDS if b in u}
            ranks['groups'].append(dict(ref=max(r['ref'] for r in rs), changed=[int(e) for e, _ in sig], scores=u, ranks=rk, rank_sum=sum(rk.values()) if len(rk) == len(BOARDS) else None))
    pred = dict(scores={b: round(best[2][b], 5) for b in BOARDS}, ranks={b: rank_on(b, best[2][b], lb) for b in BOARDS}, rank_sum=best[0][0]) if best else None
    doc = dict(updated_utc=now(), prediction=pred, pending=pending, sources={str(e): [{k: v for k, v in c.items() if k != 'info'} for c in v] for e, v in src.items()}, code_notes=CODE_NOTES, ranks=ranks, rules=RULES, handover=[dict(item=a, note=b) for a, b in HANDOVER], base=dict(submission=BOB1_REF, auc=base_auc, boards=meta[BOB1_REF]['boards']), expected_auc_all_incumbents=expected, best_measured=dict(ref=best_measured['ref'], auc=best_measured['auc'], description=best_measured['description']), quota=quota, incumbents={str(k): v for k, v in table.items()}, submissions=[{k: v for k, v in r.items() if k != 'episodes'} for _, r in sorted(meta.items(), reverse=True)], merge_command=cmd)
    atomic(W / 'INCUMBENTS.json', json.dumps(doc, indent=1, default=str) + '\n')
    L = [f"# Track 3 per-episode incumbents", '', f"Generated {doc['updated_utc']} UTC by `board_watch.py` from the Kaggle board. Do not edit by hand.", '',
         f"Best scored file: {best_measured['auc']} ({best_measured['ref']}). Incumbents below are chosen by predicted rank-sum among sources measured on all six boards; expected AUC with all of them: **{expected}**. bob1: {base_auc}.", '',
         'Quota left today: ' + ', '.join(f"{b} {quota[b]['left']}" for b in BOARDS), '',
         '## Rules', ''] + [f'- {x}' for x in RULES] + ['', '## Incumbents', '',
         '| Ep | Incumbent | Run | Own-ref | Delta vs bob1 | Boards measured | Meshes changed | Checkpoint sha | Rows sha | Tried and rejected | Candidate, six-board pending | Mesh caveat |', '|---:|---|---|---:|---:|---|---:|---|---|---|---|---|']
    for e, t in table.items():
        cand_txt = '; '.join('%s %+.4f' % (x['tag'], x['auc_delta_ep']) for x in pending if x['episode'] == e and (x['auc_delta_ep'] or 0) > 0)
        rej = '; '.join(f"{x['by']} {x['ref']} {'pending' if x['delta'] is None else format(x['delta'], '+.3f')}" for x in t['tried'] if x['ref'] != t['submission']) or ''
        L.append(f"| {e} | {t['source']} | `{t['run']}` | {t['own_ref'] if t['own_ref'] is not None else ''} | {t['delta_vs_bob1']:+.4f} | {t['boards_measured']} | {t['meshes_changed']} | `{t['checkpoint_sha256'][:12]}` | `{t['rows_sha256'][:12]}` | {rej} | {cand_txt} | {t['mesh_caveat']} |")
    fmt = {'auc': '{:+.5f}', 'sp-sr': '{:+.2f}', 'mp-sr': '{:+.2f}', 'rpe': '{:+.4f}', 'mppe': '{:+.4f}', 'cd-o': '{:+.5f}'}
    L += ['', '## Per-board matrix', '', 'Change in each board score against bob1. A star marks a value not read from a board: subtracted from a merged file, or an exact zero on CD-O when no mesh changed. A dash marks a board not measured yet. Lower is better for RPE, MPPE and CD-O.', '', '| Ep | Source | Run | ' + ' | '.join(BOARDS) + ' | Meshes changed | Status |', '|---:|---|---|' + '---:|' * len(BOARDS) + '---:|---|']
    for e in sorted(src):
        for c in sorted(src[e], key=lambda z: -z['deltas'].get('auc', -9)):
            cells = [fmt[b].format(c['deltas'][b]) + ('*' if b in c['derived'] else '') if b in c['deltas'] else '-' for b in BOARDS]
            st_ = 'selected' if selected.get(e) is c else ('rejected' if set(c['refs']) & set(EXCLUDED) else 'needs boards' if any(b not in c['deltas'] for b in BOARDS) else 'not selected')
            L.append(f"| {e} | {c['tag']} {c['by']} | `{c['run'] or ''}` | " + ' | '.join(cells) + f" | {c['info']['meshes_changed']} | {st_} |")
    L += ['', '## Best source per board', '', 'Only sources that beat bob1 on that board. One file serves all six boards, so this shows the conflicts; the selection below resolves them.', '', '| Ep | ' + ' | '.join(BOARDS) + ' |', '|---:|' + '---|' * len(BOARDS)]
    for e in sorted(src):
        row_ = []
        for b in BOARDS:
            have = [c for c in src[e] if b in c['deltas'] and (c['deltas'][b] > 0 if HIGHER_BETTER[b] else c['deltas'][b] < 0)]
            row_.append((lambda c: f"{c['tag']} {fmt[b].format(c['deltas'][b])}")(max(have, key=lambda c: c['deltas'][b] if HIGHER_BETTER[b] else -c['deltas'][b])) if have else '')
        if any(row_): L.append(f'| {e} | ' + ' | '.join(row_) + ' |')
    if best:
        L += ['', '## Best one-file selection', '', f"Lowest predicted rank-sum among sources measured on all six boards, ties broken by AUC, tested over {n} combinations: {', '.join(f'ep{e} {c[chr(116) + chr(97) + chr(103)]}' for e, c in sorted(selected.items())) or 'bob1 alone'}. Predicted rank-sum **{best[0][0]}**.", '', '| Board | Predicted score | Predicted rank |', '|---|---:|---:|'] + [f"| {b} | {pred['scores'][b]} | {pred['ranks'][b]} |" for b in BOARDS]
    for ref_, why in EXCLUDED.items():
        L += ['', f'Rejected: {ref_}, {why}.']
    if pending:
        L += ['', 'Not eligible until the listed boards are in:', ''] + [f"- ep{x['episode']} {x['tag']} ({x['ref']}), AUC on that episode {x['auc_delta_ep']}: missing {', '.join(x['missing'])}" for x in pending]
    own = {}
    for f in glob.glob('/root/bob1/repo/experiments/edmond/bob1_submission_20261103/episode_*/policy/results/episode_*.json'):
        own[int(f.split('episode_')[-1][:6])] = json.load(open(f))['metrics']
    est = {}
    for e in table:
        p_ = plan[e]; v = RATIO.get(p_.get('source'), 0.81) * float(p_['tracking']) + (selected[e]['deltas']['auc'] * N_EP if e in selected else 0.0); est[e] = v
    L += ['', '## Likeliest success failures', '', 'The base has one SP-SR failure and ten MP-SR failures, and no per-episode score is public. Expected AUC is ratio times own-reference tracking, plus any measured swap gain, so it is a rough ordering. SP-SR needs 10 cm and 0.5 rad; MP-SR needs 3 cm and 30 degrees on every object. Both are judged against the hidden reference, so own-reference columns are weak hints.', '', '| Ep | Expected AUC | Own-ref tracking | Own-ref MP | Own-ref chord | Known board flips |', '|---:|---:|---:|---:|---:|---|']
    flips = {e: [f"{c['tag']}: {b} {fmt[b].format(c['deltas'][b])}" for c in v for b in ('sp-sr', 'mp-sr') if abs(c['deltas'].get(b, 0)) > 1e-9] for e, v in src.items()}
    for e, v in sorted(est.items(), key=lambda kv: kv[1])[:12]:
        m = own.get(e, {}); L.append(f"| {e} | {v:.3f} | {float(plan[e]['tracking']):.3f} | {m.get('maniptrans_sr')} | {m.get('chord_sr')} | {'; '.join(flips.get(e, []))} |")
    L += ['', '## Six boards', '', 'Same rollouts count as one entry, whichever file or board carried them. Lower is better for RPE, MPPE and CD-O.', '', '| Ref | Changed episodes | AUC | SP-SR | MP-SR | RPE cm | MPPE cm | CD-O cm | Still pending |', '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for sig, rs in sorted(groups.items(), key=lambda kv: -max(r['ref'] for r in kv[1])):
        u = rs[0]['boards_same_rollouts']; pend = sorted({b for r in rs for b in r.get('pending_boards', [])})
        if len(u) + len(pend) < 2: continue
        L.append(f"| {max(r['ref'] for r in rs)} | {','.join(str(e) for e, _ in sig) or 'bob1'} | " + ' | '.join(str(u.get(b, '')) for b in BOARDS) + f" | {','.join(pend)} |")
    if ranks:
        L += ['', '## Rank-sum', '', 'Rank = 1 + other teams strictly better on the public board, so ties count in our favour. Lower sum is better. A file missing a board has no sum.', '', f"Teams listed per board: {', '.join(f'{b} {n}' for b, n in ranks['teams'].items())}. Our best per board: {', '.join(f'{b} {v}' for b, v in ranks['team_best'].items())}. Rank sum now: **{ranks['team_rank_sum']}**.", '', '| Ref | Changed episodes | ' + ' | '.join(BOARDS) + ' | Rank-sum |', '|---:|---|' + '---:|' * (len(BOARDS) + 1)]
        L.append('| team best | | ' + ' | '.join(str(ranks['team_rank'][b]) for b in BOARDS) + f" | {ranks['team_rank_sum']} |")
        for g in ranks['groups']:
            L.append(f"| {g['ref']} | {','.join(map(str, g['changed'])) or 'bob1'} | " + ' | '.join(str(g['ranks'].get(b, '')) for b in BOARDS) + f" | {g['rank_sum'] if g['rank_sum'] is not None else ''} |")
    if ranks:
        L += ['', '## Cheapest places to gain', '', 'Gain needed is the change in that board score to pass the next team, or the next three. Margin behind is how much we can lose before the team behind passes us. Because the sum counts places, a small gain on a crowded board can beat a large gain on a sparse one.', '', '| Board | Score | Rank | Next three ahead | Gain for 1 place | Gain for 3 places | Margin behind |', '|---|---:|---:|---|---:|---:|---:|']
        for b, g in ranks['gaps'].items():
            a = g['next_ahead']; f = lambda v: f'{abs(v - g["score"]):.5f}'
            L.append(f"| {b} | {g['score']} | {g['rank']} | {', '.join(str(x) for x in a)} | {f(a[0]) if a else ''} | {f(a[2]) if len(a) > 2 else ''} | {f(g['next_behind'][0]) if g['next_behind'] else ''} |")
    L += ['', '## Run directories for the final pack', '', 'Verified means: checkpoint hash equal across policy file, results json and submitted rows; rollout rows exactly equal; scene meshes reproduce the submitted payload. Row-merged episodes need nothing further from us: their rows come from the scored Kaggle files.', '', '| Source | Status | Run dir | Scene | Checkpoint |', '|---|---|---|---|---|'] + [f'| {a} | {b} | `{c}` | `{d}` | {e_} |' for a, b, c, d, e_ in RUN_DIRS]
    L += ['', '## Code and provenance for the final commit', ''] + [f'- {x}' for x in CODE_NOTES]
    L += ['', '## Notes for episode owners', '', 'Information only, no requests. From MESH_SIZE_AUDIT.md: a flag means the episodes disagree, not which one is wrong. No mesh was changed. The legal path for any of these is re-track and retrain on the better mesh; a mesh-only swap is barred.', ''] + [f'- **{a}:** {b}' for a, b in HANDOVER]
    L += ['', '## Team submissions since bob1', '', '| Ref | Time UTC | By | AUC | Class | Changed episodes | Boards | Description |', '|---:|---|---|---:|---|---|---:|---|']
    for ref, r in sorted(meta.items(), reverse=True):
        exp = f" (expected {r['expected_auc']:.5f})" if r.get('expected_auc') else ''
        L.append(f"| {ref} | {r['date'][5:]} | {r['by']} | {r['auc'] if r['auc'] is not None else r['status']} | {r.get('klass') or 'unclassified'}{exp} | {','.join(map(str, r.get('changed') or []))} | {len(r['boards'])} | {r['description'][:110].replace('|', '/')} |")
    L += ['', '## Build the merged file', '', '```bash', cmd, '```', '']
    atomic(W / 'INCUMBENTS.md', '\n'.join(L))
    for ref, r in sorted(meta.items()):
        key = str(ref); cur = dict(status=r['status'], auc=r['auc'], klass=r.get('klass'), boards=len(r.get('boards_same_rollouts') or r['boards']))
        if key in seen and 'boards' not in seen[key]: seen[key] = dict(seen[key], boards=cur['boards'])  # older state file
        if seen.get(key) != cur:
            if key in seen or ref > 57973091 or r['status'].upper() == 'PENDING' or r['auc'] is None:
                d = r.get('delta_vs_bob1')
                event(f"SUBMISSION ref={ref} by={r['by']} status={r['status']} auc={r['auc']} class={r.get('klass')} changed={r.get('changed')} delta={None if d is None else round(d, 4)} boards={cur['boards']} | {r['description'][:120]}")
            seen[key] = cur
    atomic(seen_f, json.dumps(seen, indent=1) + '\n')
    inc_f = W / 'incumbent_ids.json'; ids = {str(t['episode']): t['submission'] for t in swaps}
    old = json.loads(inc_f.read_text()) if inc_f.is_file() else {}
    if ids != old:
        event(f"INCUMBENTS_CHANGED {ids} expected_auc={expected}"); atomic(inc_f, json.dumps(ids) + '\n')
    print(now(), 'ok; expected', expected, 'best', best_measured['auc'], 'submissions', len(meta), 'quota auc', quota['auc'], flush=True)

if __name__ == '__main__':
    main()
