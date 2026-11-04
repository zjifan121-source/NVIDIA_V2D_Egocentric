"""Second add-only addendum for the lead_sprint_v2 prefix ("hand stage" over all episodes): for every episode that has no
scene/ in the prefix yet, ship the scene the ledger recommends (ours: rawexcl / kenexcl / control), plus the control
checks; for episodes whose scene is already shipped, ship an alternative variant only when the ledger names one that
passes the gate (as scene_alt_<variant>/). Also ships HAND_LEDGER.{json,md} and the full comparison. Writes
stage_addendum_v3/ and UPLOAD_DRY_RUN_ADDENDUM_V3.json."""
from pathlib import Path
import json, shutil, hashlib, datetime, re
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'; L=OPS/'experiments_20261103'
W=Path(__file__).resolve().parent; S=W/'stage_addendum_v3'; V1S=ROOT/'work/lead-sprint-release-20261103/stage'; HAND=L/'lead_hand_reattach_v1'; SEAT=L/'lead_support_seating_v1'
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_sprint_v2_20261103'
v2=json.loads((W/'UPLOAD_VERIFIED.json').read_text()); a1=json.loads((W/'UPLOAD_VERIFIED_ADDENDUM.json').read_text())
existing={r['relative'] for r in v2['sha256']}|{r['relative'] for r in a1['sha256']}; PARENT=a1['commit']
shipped_scene={int(m.group(1)) for rel in existing for m in [re.match(r'scenes/episode_(\d{6})/scene/',rel)] if m}
ledger={r['episode']:r for r in json.loads((HAND/'HAND_LEDGER.json').read_text())}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
LOCAL_ROOTS=[str(HAND),str(SEAT),str(V1S),str(ROOT),'/private/tmp/claude-501']
def relocal(x):
    if isinstance(x,dict): return {k:relocal(v) for k,v in x.items()}
    if isinstance(x,list): return [relocal(v) for v in x]
    if isinstance(x,str):
        for r in LOCAL_ROOTS:
            if x.startswith(r): return '<local>'+x[len(r):]
    return x
if S.exists(): shutil.rmtree(S)
S.mkdir()
def ship(ep,variant,dest_name):
    sd=S/'scenes'/f'episode_{ep:06d}'; sc=HAND/'scenes'/f'ep{ep:02d}_{variant}'; assert (sc/'OUTPUT_CHECK.json').exists(),(ep,variant,'no scene')
    shutil.copytree(sc,sd/dest_name,symlinks=False,ignore=shutil.ignore_patterns('*.generated_by_loader'))
    for f in list((sd/dest_name).rglob('*.json')): put(f,relocal(read(f)))
    if variant in ('rawexcl','kenexcl'):
        b=HAND/'bundles'/f'ep{ep:02d}_{variant}'; shutil.copytree(b,sd/f'bundle_{variant}' if dest_name=='scene' else sd/f'bundle_{variant}_alt',symlinks=False)
        for f in list((sd/(f'bundle_{variant}' if dest_name=='scene' else f'bundle_{variant}_alt')).rglob('*.json')): put(f,relocal(read(f)))
        b0=np.load(V1S/f'scenes/episode_{ep:06d}/bundle/result.npz',allow_pickle=False); b1=np.load(b/'result.npz',allow_pickle=False)
        keys=[k for k in b0.files if k.endswith('_to_world_transform') and not k.startswith('hand_')]
        assert all(np.allclose(b0[k],b1[k],atol=1e-6) for k in keys),(ep,variant,'object/camera world poses changed')
summary={}
for ep,r in sorted(ledger.items()):
    rec=r['recommendation']; m=re.search(r'ours: (rawexcl|kenexcl|control)',rec); alt=re.search(r'备选 ours: (rawexcl|kenexcl)',rec)
    if ep in shipped_scene:
        if alt: ship(ep,alt.group(1),f'scene_alt_{alt.group(1)}'); summary[ep]=dict(action=f'alternative scene_alt_{alt.group(1)}',recommendation=rec)
        else: summary[ep]=dict(action='already shipped; no alternative',recommendation=rec)
        continue
    variant=m.group(1) if m else (alt.group(1) if alt else 'control')
    if not (HAND/'scenes'/f'ep{ep:02d}_{variant}'/'OUTPUT_CHECK.json').exists(): variant='control'
    ship(ep,variant,'scene')
    sd=S/'scenes'/f'episode_{ep:06d}'
    if (HAND/'scenes'/f'ep{ep:02d}_control'/'OUTPUT_CHECK.json').exists() and variant!='control':
        put(sd/'CONTROL_OUTPUT_CHECK.json',relocal(read(HAND/'scenes'/f'ep{ep:02d}_control'/'OUTPUT_CHECK.json'))); put(sd/'CONTROL_HAND_CONTACT.json',read(HAND/'scenes'/f'ep{ep:02d}_control'/'HAND_CONTACT.json'))
    put(sd/'HANDS_ADDENDUM.json',dict(episode=ep,release='lead_sprint_v2 addendum v3 (hand stage over all episodes)',objects='identical to lead_sprint_v1 / v2 bundle/',support='seated support_v1 of v2 installed in the scene',shipped_variant=variant,
        hands={'rawexcl':'team raw hands, loader clean+smooth, exclusive attach to our objects','kenexcl':"Ken's manually corrected raw hands (their bundle/ hand arrays), loader clean+smooth, exclusive attach to our objects",'control':'fb11 hands as delivered (no raw hands on HF for this episode, or re-attach failed the contact gate)'}[variant],
        ledger_recommendation=rec,why=r['why'],gate=r['gate'],contact=relocal(r['contact']),policy_tested=False))
    summary[ep]=dict(action=f'scene = {variant}',recommendation=rec)
    print('SHIP',ep,variant,'|',rec,flush=True)
shutil.copy2(HAND/'HAND_LEDGER.md',S/'HAND_LEDGER.md'); put(S/'HAND_LEDGER.json',relocal(read(HAND/'HAND_LEDGER.json')))
shutil.copy2(HAND/'HAND_REATTACH_COMPARISON.md',S/'HAND_REATTACH_COMPARISON_ALL.md'); put(S/'HAND_REATTACH_COMPARISON_ALL.json',read(HAND/'HAND_REATTACH_COMPARISON.json'))
for f in ['prep_kenhands.py','hand_ledger.py']: shutil.copy2(HAND/f,S/'tools'/f) if (S/'tools').exists() else (S/'tools').mkdir() or shutil.copy2(HAND/f,S/'tools'/f)
readme=f'''# lead_sprint_v2 追加 v3（手阶段覆盖全部 20 段，2026-11-04）

只新增文件。每段 `scenes/episode_XXXXXX/scene/`（本次新增的段）或 `scene_alt_<variant>/`（已交付段的备选）：物体与 v1/v2 的 `bundle/` 相同，桌面为 v2 坐实版，双手按台账 `HAND_LEDGER.md` 选择：

- `rawexcl`：团队原始双手 → loader 清理平滑 → 排他式 attach（每只手只贴它握的物体）。
- `kenexcl`：Ken 人工修正后的原始双手（ep3 左手 HaMeR 重跑、ep14 手性修正）→ 同上。
- `control`：fb11 双手原样（HF 上无该段原始双手：35/36/37/38；或重贴未过接触门槛：ep5）。

门槛与证据：机器人最近连杆到物体面在举起段 ≤1 cm 的帧占比，对照与候选并列；物体没改的 1/3/6/10/14/20 推荐直接用 Ken 的修复场景（有 own-ref policy 证据），这里的 `scene_alt_*` 只是备选。`HAND_LEDGER.md` 一表看全：bob1 用谁、Ken own-ref、四种做法的接触率、门槛判定、建议。

{chr(10).join(f"- ep{ep}: {v['action']}（{v['recommendation']}）" for ep,v in sorted(summary.items()))}

用法同 v2：下载后 `python tools/relocate_scene.py scenes/episode_0000XX/scene`。未做物理回放与 policy 训练；不是正式提交；无 GT。
'''
(S/'README_ADDENDUM_V3.md').write_text(readme); put(S/'SUMMARY_ADDENDUM_V3.json',dict(parent_commit=PARENT,episodes={str(k):v for k,v in summary.items()},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files=[p for p in sorted(S.rglob('*')) if p.is_file()]
ALLOWED=['.json','.csv','.md','.npz','.obj','.mtl','.png','.jpg','.jpeg','.py','.usda','.mp4','.parquet','.stl','.urdf','.log','.txt']
for p in files:
    assert p.suffix.lower() in ALLOWED,p; rel=str(p.relative_to(S)); assert rel not in existing,('would overwrite',rel)
    if p.suffix in ['.json','.md'] and 'scene' not in p.parts and not p.name.startswith('scene_alt') and 'tools' not in p.parts: assert not any(s in p.read_text() for s in ['/Volumes/','/Users/','185.216.','/private/tmp']),('local path leak',p)
manifest=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
(S/'SHA256SUMS_ADDENDUM_V3.txt').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in manifest)); manifest.append(dict(relative='SHA256SUMS_ADDENDUM_V3.txt',sha256=sha(S/'SHA256SUMS_ADDENDUM_V3.txt'),bytes=(S/'SHA256SUMS_ADDENDUM_V3.txt').stat().st_size))
put(W/'UPLOAD_DRY_RUN_ADDENDUM_V3.json',dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,expected_parent=PARENT,prefix_files_expected=len(existing),stage=str(S),files=manifest,bytes=sum(x['bytes'] for x in manifest),episodes={str(k):v['action'] for k,v in summary.items()},
    commit_message='Lead sprint v2 addendum v3: hand-stage scenes for all remaining Scored episodes + per-episode hand ledger; add-only',commit_description='Appends trainable scenes (team loader on lead_sprint_v1 objects + seated support_v1; hands = exclusive re-attach of team raw hands or Ken-corrected raw hands, else fb11 hands) with control checks, and HAND_LEDGER with contact metrics, bob1 tracking and Ken own-ref side by side; existing files untouched; no GT, no training, no official score.',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('DRY_RUN v3 files',len(manifest),'bytes',sum(x['bytes'] for x in manifest))
