"""Review a Track 3 pack plan before it is packed/submitted (single-episode Kaggle probes and final candidates).

A plan is make_bob_plan.py's output: {"<episode>": {"run": <dir with results/episode_XXXXXX.{parquet,json}>,
"scene": <dir with score_input/meshes/episode_XXXXXX/<object>.*>, "tracking": <own-reference AUC>, "source":
"ken"|"wentao", "name": ...}}. Edmond's tools/track3/pack_submission.py --plan consumes exactly these fields.

Checks (all read-only, no GT):
  1. episode set == the kit's scored roster (data/track_3_evaluation_objects.json, 20 episodes)
  2. per episode: results parquet + json exist; json metrics.add_auc == plan tracking; parquet has rows
  3. per episode: a mesh for every roster object under scene/score_input/meshes/episode_XXXXXX/
  4. scene<->policy link: the rollout's task parquet (results json / hydra / run config / train.log) lies under the
     plan's scene -> 'linked'; otherwise 'unverified' (a mismatch is a FAIL: CD-O meshes would not be the rollout's)
  5. diff against a reference plan (--ref, e.g. bob1): per episode same / relocated (same name+tracking, other paths)
     / changed; the number of changed episodes must equal --expect-changes (default 1 for a probe)
  6. expected Kaggle AUC = mean(RATIO[source] * tracking) (make_bob_plan's ratios ken 0.81 / wentao 0.86), and for
     each changed episode the implied Kaggle delta = (ratio_new*t_new - ratio_old*t_old) / 20
  7. --code-commit URL format (https://github.com/<owner>/<repo>/commit/<40 hex>) when given
Usage: check_probe_plan.py PLAN.json [--ref REF_PLAN.json] [--expect-changes N] [--roster kit_objects.json]
       [--code-commit URL] [--out REPORT.json]
Exit 0 = PASS, 1 = FAIL. Writes PLAN.check.json (or --out)."""
from pathlib import Path
import sys, os, json, re, argparse
RATIO = {'ken': 0.81, 'wentao': 0.86}
MESH_SUFFIXES = ('.glb', '.obj', '.ply', '.stl', '.off')

def load_roster(path):
    d = json.loads(Path(path).read_text())
    return {int(k): list(v) for k, v in d['episodes'].items()}

def read_json(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def parquet_rows(p):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return None

def find_task_parquet(run: Path, results_json):
    """Any recorded path of the task parquet / scene this rollout was produced on."""
    hits = []
    def scan_text(t, where):
        for m in re.findall(r"[\w./=\-]*(?:processed|sequence_id=|task\.parquet=|\.parquet)[\w./=\-]*", t):
            if 'human_motion_data' in m or 'processed' in m or m.endswith('.parquet'): hits.append((where, m.split('=')[-1] if 'task.parquet=' in m else m))
    if isinstance(results_json, dict):
        scan_text(json.dumps(results_json), 'results.json')
    for cand in [run / 'results' / 'hydra', run / 'hydra', run / '.hydra', run.parent / 'train' / '.hydra', run.parent / '.hydra']:
        if cand.is_dir():
            for f in sorted(cand.glob('*.yaml'))[:6]:
                try: scan_text(f.read_text()[:200000], str(f))
                except Exception: pass
    for f in [run / 'train.log', run.parent / 'train.log', run / 'START', run.parent / 'START']:
        if f.is_file():
            try: scan_text(f.read_text()[:200000], str(f))
            except Exception: pass
    for d in [run.parent, run]:
        for f in list(d.glob('wandb/*/files/config.yaml'))[:2] + list(d.glob('train/wandb/*/files/config.yaml'))[:2]:
            try: scan_text(f.read_text()[:200000], str(f))
            except Exception: pass
    return hits

def check(plan_path, ref_path, roster_path, expect_changes, code_commit):
    plan = {int(k): v for k, v in json.loads(Path(plan_path).read_text()).items()}
    roster = load_roster(roster_path)
    ref = {int(k): v for k, v in json.loads(Path(ref_path).read_text()).items()} if ref_path else None
    problems, warnings, rows = [], [], {}
    missing = sorted(set(roster) - set(plan)); extra = sorted(set(plan) - set(roster))
    if missing: problems.append(f'episodes missing from plan: {missing}')
    if extra: problems.append(f'episodes not scored: {extra}')
    for ep in sorted(set(plan) & set(roster)):
        e = plan[ep]; stem = f'episode_{ep:06d}'; run = Path(e['run']); scene = Path(e['scene']); row = dict(name=e.get('name'), source=e.get('source'), tracking=e.get('tracking'))
        pq_file, js_file = run / 'results' / f'{stem}.parquet', run / 'results' / f'{stem}.json'
        if not pq_file.is_file(): problems.append(f'ep{ep}: missing {pq_file}')
        else:
            n = parquet_rows(pq_file); row['parquet_rows'] = n
            if n == 0: problems.append(f'ep{ep}: empty results parquet {pq_file}')
        js = read_json(js_file) if js_file.is_file() else None
        if js is None: problems.append(f'ep{ep}: missing/unreadable {js_file}')
        else:
            auc = (js.get('metrics') or {}).get('add_auc'); row['json_add_auc'] = auc
            if auc is None: warnings.append(f'ep{ep}: results json has no metrics.add_auc')
            elif e.get('tracking') is None: warnings.append(f'ep{ep}: plan has no tracking; json add_auc={auc:.4f}')
            elif abs(float(auc) - float(e['tracking'])) > 5e-4: problems.append(f"ep{ep}: plan tracking {e['tracking']} != results add_auc {auc}")
            elif abs(float(auc) - float(e['tracking'])) > 1e-6: warnings.append(f"ep{ep}: plan tracking {e['tracking']} is add_auc {auc:.6f} rounded")
        if e.get('source') not in RATIO: problems.append(f"ep{ep}: source {e.get('source')!r} not in {list(RATIO)} (needed for the expected-Kaggle rule)")
        mdir = scene / 'score_input' / 'meshes' / stem; found = {}
        for name in roster[ep]:
            hit = [p for p in mdir.glob(name + '.*') if p.suffix in MESH_SUFFIXES] if mdir.is_dir() else []
            if not hit: problems.append(f'ep{ep}: no mesh for {name} in {mdir}')
            else: found[name] = hit[0].name
        row['meshes'] = found
        stray = sorted(p.stem for p in mdir.glob('*') if p.suffix in MESH_SUFFIXES and p.stem not in roster[ep] and not p.name.startswith('._')) if mdir.is_dir() else []
        if stray: warnings.append(f'ep{ep}: extra mesh files in {mdir}: {stray}')
        hits = find_task_parquet(run, js); row['task_parquet_hits'] = hits[:3]
        if isinstance(js, dict):
            row['checkpoint'] = js.get('checkpoint'); row['checkpoint_sha256'] = js.get('checkpoint_sha256'); row['environment_steps'] = js.get('environment_steps')
        # link: the rollout's recorded task parquet must be the plan scene's parquet. Paths differ across machines (Babel vs
        # ours), so match on the scene's own partition: same sequence_id=<seq> and, if possible, the same parquet file name.
        proc = scene / 'assets/human_motion_data/ego_recon/processed'
        scene_parts = sorted(d.name for d in proc.glob('sequence_id=*')) if proc.is_dir() else []
        if not scene_parts:  # HF scene bundles omit the processed parquet; the loader's report.json records its partition
            rep = read_json(scene / 'report.json') if (scene / 'report.json').is_file() else None
            mm = re.search(r'(sequence_id=[^/]+)', str((rep or {}).get('processed', '')))
            if mm: scene_parts = [mm.group(1)]; row['scene_sequence_from'] = 'report.json'
        scene_files = {f.name for d in proc.glob('sequence_id=*') for f in d.glob('robot_name=*/*.parquet')} if proc.is_dir() else set()
        linked = None
        for where, h in hits:
            m = re.search(r'(sequence_id=[^/]+)', h); base = h.rsplit('/', 1)[-1]
            if base.endswith('.parquet') and base in scene_files: linked = True; break
            if m and scene_parts:
                if m.group(1) in scene_parts: linked = True; break
                linked = False; row['rollout_sequence'] = m.group(1); row['scene_sequences'] = scene_parts
        row['scene_policy_link'] = 'linked' if linked else ('MISMATCH' if linked is False else 'unverified')
        if linked is False: problems.append(f"ep{ep}: rollout was produced on {row['rollout_sequence']} but the plan scene holds {scene_parts} (CD-O meshes would not be the rollout's)")
        elif linked is None: warnings.append(f'ep{ep}: scene<->policy link unverified (no recorded task parquet path, or scene has no processed partition)')
        if ref is not None:
            r = ref.get(ep)
            if r is None: row['vs_ref'] = 'new_episode'
            else:
                same_track = r.get('tracking') is not None and e.get('tracking') is not None and abs(float(r['tracking']) - float(e['tracking'])) <= 1e-6
                same_name = (r.get('name') == e.get('name')) if (r.get('name') and e.get('name')) else None
                if same_track and same_name is not False and r.get('source') == e.get('source'):
                    row['vs_ref'] = 'same' if (r['run'] == e['run'] and r['scene'] == e['scene']) else 'relocated'
                else:
                    row['vs_ref'] = 'changed'; row['ref'] = dict(name=r.get('name'), source=r.get('source'), tracking=r.get('tracking'))
                    row['implied_kaggle_delta'] = round((RATIO.get(e.get('source'), 0) * float(e.get('tracking') or 0) - RATIO.get(r.get('source'), 0) * float(r.get('tracking') or 0)) / len(roster), 5)
        rows[ep] = row
    # provenance of OUR scenes in every changed episode (and of all 'wentao' episodes when there is no reference): the
    # path_a/ops + work scripts the submission depends on, for the --code-commit push list
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from scene_provenance import provenance
    except Exception as exc:
        provenance = None; warnings.append(f'scene_provenance unavailable ({exc!r}); scripts_to_push not listed')
    scripts_to_push = set()
    for ep, r in rows.items():
        if provenance and (r.get('vs_ref') in ('changed', 'new_episode') or r.get('source') == 'wentao'):
            try:
                pv = provenance(Path(plan[ep]['scene'])); r['provenance'] = dict(type=pv['type'], hands=pv['inputs'].get('hands'), objects_hf_prefix=pv['inputs'].get('objects_hf_prefix'), markers=pv['markers'], scripts=pv['scripts'], notes=pv['notes'])
                fg = pv['inputs'].get('false_grasp_hands') or []; r['false_grasp_hands'] = fg; r['false_grasp_check'] = pv['inputs'].get('false_grasp_check')
                if fg: warnings.append(f"ep{ep}: FALSE GRASP hand(s) in the plan scene: " + ', '.join(f"{x['hand']}->{x['object']} ({x['pre_attach_median_cm']} cm pre-attach median)" for x in fg) + f" [{r['false_grasp_check']}] — the policy trains on a hand glued to an object it only passed; prefer an attach_hold.py scene")
                moved = {k: v for k, v in (pv['inputs'].get('hold_hand_moved_max_cm') or {}).items() if v > 12}
                r['hold_hand_moved_max_cm'] = pv['inputs'].get('hold_hand_moved_max_cm')
                if moved: warnings.append(f'ep{ep}: per-frame hold moved a hand far from its reconstruction (max cm {moved}); check gaps_filled in ATTACH_HOLD.json')
                rg = pv['inputs'].get('hold_release_gaps') or []; r['hold_release_gaps'] = rg
                if rg: warnings.append(f"ep{ep}: hold keeps a hand on a placed object: " + ', '.join(f"{x['hand']} f{x['frames'][0]}-{x['frames'][1]} (object net {x['object_net_cm']} cm, wrist {x['wrist_median_cm']} cm away, tracked)" for x in rg))
                if pv['type'] != 'unknown': scripts_to_push |= set(pv['scripts'])
            except Exception as exc:
                warnings.append(f'ep{ep}: provenance failed ({exc!r})')
    exp = [RATIO[v['source']] * float(v['tracking']) for v in plan.values() if v.get('source') in RATIO and v.get('tracking') is not None]
    summary = dict(episodes=len(plan), expected_kaggle_auc=round(sum(exp) / len(exp), 5) if len(exp) == len(roster) else None,
                   mean_tracking=round(sum(float(v['tracking']) for v in plan.values() if v.get('tracking') is not None) / max(1, len(plan)), 5))
    summary['scripts_to_push'] = sorted(scripts_to_push)
    summary['false_grasp_episodes'] = sorted(int(ep) for ep, r in rows.items() if r.get('false_grasp_hands'))
    summary['hold_release_gap_episodes'] = sorted(int(ep) for ep, r in rows.items() if r.get('hold_release_gaps'))
    if ref is not None:
        rexp = [RATIO[v['source']] * float(v['tracking']) for v in ref.values() if v.get('source') in RATIO and v.get('tracking') is not None]
        summary['ref_expected_kaggle_auc'] = round(sum(rexp) / len(rexp), 5) if rexp else None
        changed = [ep for ep, r in rows.items() if r.get('vs_ref') in ('changed', 'new_episode')]
        summary['changed_episodes'] = changed; summary['relocated'] = sum(1 for r in rows.values() if r.get('vs_ref') == 'relocated')
        if expect_changes is not None and len(changed) != expect_changes:
            problems.append(f'{len(changed)} episodes differ from the reference ({changed}); expected {expect_changes}')
    if code_commit is not None and not re.fullmatch(r'https://github\.com/[\w.\-]+/[\w.\-]+/commit/[0-9a-f]{40}', code_commit):
        problems.append(f'--code-commit must be https://github.com/<owner>/<repo>/commit/<40 hex sha>, got {code_commit!r}')
    return dict(plan=str(plan_path), ref=str(ref_path) if ref_path else None, verdict='FAIL' if problems else 'PASS', problems=problems, warnings=warnings, summary=summary, episodes={str(k): v for k, v in rows.items()})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('plan'); ap.add_argument('--ref'); ap.add_argument('--expect-changes', type=int, default=1)
    ap.add_argument('--roster', default=os.environ.get('LEAD_KIT_ROSTER', str(Path(os.environ.get('V2D_ROOT', '/Volumes/E/Code/V2D')) / 'NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit/data/track_3_evaluation_objects.json')))
    ap.add_argument('--code-commit'); ap.add_argument('--out'); a = ap.parse_args()
    rep = check(a.plan, a.ref, a.roster, a.expect_changes if a.ref else None, a.code_commit)
    out = Path(a.out) if a.out else Path(a.plan).with_suffix('.check.json'); out.write_text(json.dumps(rep, indent=1) + '\n')
    for ep, r in rep['episodes'].items():
        print(f"ep{int(ep):>3} {str(r.get('source')):6} track {r.get('tracking') if r.get('tracking') is None else round(float(r['tracking']), 4)!s:<7} meshes {len(r.get('meshes', {}))} link {r.get('scene_policy_link'):<10} {r.get('vs_ref', ''):<9} {('dKaggle %+.4f' % r['implied_kaggle_delta']) if 'implied_kaggle_delta' in r else ''}{' FALSE-GRASP(' + ','.join(x['hand'] for x in r['false_grasp_hands']) + ')' if r.get('false_grasp_hands') else ''} [{r.get('name')}]")
    print(json.dumps({k: v for k, v in rep['summary'].items() if k != 'scripts_to_push'}))
    if rep['summary'].get('scripts_to_push'):
        print('  scripts the changed episodes\' scenes depend on (push before --code-commit cites our commit):')
        for sline in rep['summary']['scripts_to_push']: print('    -', sline)
    for w in rep['warnings']: print('  warn:', w)
    for p in rep['problems']: print('  FAIL:', p)
    print(rep['verdict'], '->', out); sys.exit(0 if rep['verdict'] == 'PASS' else 1)
