"""Assemble final/ep28_wentao_oldpot_hold_com_kenterm_v1 for the team HF dataset (append-only, new sub-prefix).
Layout follows final/ep20_wentao_objects_ken_hands_v1 (bundle files flat) plus scene/, run_kenf16_s203/, pack/.
Writes stage/ and UPLOAD_DRY_RUN.json (relative path, sha256, bytes per file)."""
from pathlib import Path
import json, shutil, hashlib, re, datetime
import numpy as np
W = Path(__file__).resolve().parent; P = W / 'pull' / 'x'; S = W / 'stage'
ROOT = Path('/Volumes/E/Code/V2D')
REPO = 'rose-tea-cafe/v2d-t3-recon'; PREFIX = 'final/ep28_wentao_oldpot_hold_com_kenterm_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
HOSTPATHS = [r'/root/v2d/[^\s"\']*', r'/root/runs/[^\s"\']*', r'/root/probes/[^\s"\']*', r'/root/[^\s"\']*', r'/mnt/v2d/[^\s"\']*', r'/data/user_data/[^\s"\']*', r'/tmp/[^\s"\']*', r'/Volumes/E/Code/V2D/[^\s"\']*']
def relocal(text):
    for pat in HOSTPATHS:
        text = re.sub(pat, lambda m: '<local>/' + m.group(0).split('/')[-1], text)
    return text
def sanitize_tree(d):
    for f in d.rglob('*'):
        if f.is_file() and f.suffix in ('.json', '.log', '.yaml', '.txt', '.md', '.sh'):
            t = f.read_text(errors='replace'); t2 = relocal(t)
            if t2 != t: f.write_text(t2)
if S.exists(): shutil.rmtree(S)
S.mkdir()
# 1. bundle (flat, like the ep20 entry)
B = P / 'ep28_hold'
for f in B.iterdir():
    if f.is_dir(): shutil.copytree(f, S / f.name)
    else: shutil.copy2(f, S / f.name)
# 2. scene as trained (processed assets, score_input, checks); drop the relocation marker so the consumer re-runs relocate
sc = P / 'ep28_hold_oldpot'; shutil.copytree(sc, S / 'scene', ignore=shutil.ignore_patterns('.relocated', '*.generated_by_loader'))
# 3. run essentials
R = P / 'runs' / 'ep28_holdold_kenf16_s203'; RO = S / 'run_kenf16_s203'; RO.mkdir()
for n in ['checkpoint_selection.json', 'checkpoint_evaluations.log', 'SCENE', 'START', 'TRAIN_EXITED']: shutil.copy2(R / n, RO / n)
shutil.copytree(R / 'early_90005504', RO / 'early_90005504', ignore=shutil.ignore_patterns('.evaluation.lock'))
shutil.copy2(R / 'train' / 'policy_90005504.safetensors', RO / 'policy_90005504.safetensors')
# 4. pack
PK = P / 'probes' / 'ep28_holdold_kenf16_s203_90005504'; PO = S / 'pack'; PO.mkdir()
for f in PK.iterdir():
    if f.is_file(): shutil.copy2(f, PO / f.name)
# 5. tools
(S / 'tools').mkdir(); shutil.copy2(ROOT / 'work/lead-v3-release-20261104/stage/tools/relocate_scene.py', S / 'tools/relocate_scene.py')
sanitize_tree(S)
# 6. judge lines
L = (ROOT / 'work/reports/20261103-lead-sprint/CHECKPOINT_LEDGER.md').read_text().splitlines()
hdr = [l for l in L if l.startswith('| Ep ')][0]; rows = [l for l in L if 'ep28_holdold_kenf16_s203' in l]
(RO / 'JUDGE.md').write_text('# GT-free judge lines (board emulator with the incumbent reference + camera proxy)\n\nFrom work/reports/20261103-lead-sprint/CHECKPOINT_LEDGER.md (review session, 2026-11-04). Two-check = emulator and camera proxy both above the incumbent (bob1 ep28, own-ref 0.906).\n\n' + hdr + '\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n' + '\n'.join(rows) + '\n')
# 7. validation facts
sel = json.loads((RO / 'checkpoint_selection.json').read_text()); best = max(sel, key=lambda r: r['auc'])
npz = np.load(S / 'result.npz', allow_pickle=False)
frames = int(npz['object_to_world_transform'].shape[0])
pq = sorted((S / 'scene/assets/human_motion_data/ego_recon/processed').rglob('*.parquet'))[0]
val = dict(episode=28, objects=['plastic_dish_rack', 'white_pot'], frames=frames, result_sha256=sha(S / 'result.npz'),
           scene_parquet=str(pq.relative_to(S)), scene_parquet_sha256=sha(pq), policy='run_kenf16_s203/policy_90005504.safetensors', policy_sha256=sha(RO / 'policy_90005504.safetensors'),
           rollout='run_kenf16_s203/early_90005504/results/episode_000028.parquet', rollout_sha256=sha(RO / 'early_90005504/results/episode_000028.parquet'),
           own_ref_by_checkpoint={f"{r['step']//1000000}M": dict(auc=round(r['auc'], 4), per_body=[round(x, 3) for x in r['per_body']]) for r in sorted(sel, key=lambda r: r['step'])},
           selected_step=best['step'], selected_own_ref=round(best['auc'], 4), incumbent_bob1_own_ref=0.906,
           judge_90M=dict(gated_gain=0.012, emulator_gain=0.026, camera_gain=0.012, emulator_auc=0.927, two_check=True),
           six_board_pack='pack/ep28_holdold_kenf16_s203_90005504_merged.parquet', six_board_pack_sha256=sha(PO / 'ep28_holdold_kenf16_s203_90005504_merged.parquet'),
           single_swap_sha256=sha(PO / 'ep28_holdold_kenf16_s203_90005504.parquet'), uploaded_to_kaggle=False,
           finite_numeric_arrays=all(np.isfinite(npz[k]).all() for k in npz.files if npz[k].dtype.kind == 'f'))
(S / 'VALIDATION.json').write_text(json.dumps(val, indent=2) + '\n')
# 8. manifest (bundle files, like ep20's entry)
files = {str(f.relative_to(S)): sha(f) for f in S.rglob('*') if f.is_file() and not str(f.relative_to(S)).startswith(('scene/', 'run_kenf16_s203/', 'pack/', 'tools/')) and f.name not in ('manifest.json',)}
man = dict(schema='ego_recon_scene_bundle', episode=28, objects=['plastic_dish_rack', 'white_pot'], units='metres', state='POLICY_TRAINED_AND_JUDGED_NOT_UPLOADED', files=files, matching_mesh_pose_required=True, result_npz='result.npz',
           objects_source='lead_v3 plastic_dish_rack + the lead_v3/alignment_mix white_pot (same pot as the team base bob1 ep28); NOT the v8 donor pot',
           hands_source='team raw hands -> loader clean -> per-frame hold rule (ATTACH_HOLD.json)', scene='scene/ (support seated, inertial origins at shell centroids, njmax 1024)',
           built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(S / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n')
rows = [dict(relative=str(f.relative_to(S)), sha256=sha(f), bytes=f.stat().st_size) for f in sorted(S.rglob('*')) if f.is_file()]
(W / 'UPLOAD_DRY_RUN.json').write_text(json.dumps(dict(stage=str(S), repo=REPO, prefix=PREFIX, files=rows, bytes=sum(r['bytes'] for r in rows)), indent=2) + '\n')
print('staged', len(rows), 'files', sum(r['bytes'] for r in rows), 'bytes; selected', best['step'], round(best['auc'], 4), 'frames', frames)
