"""Build the P0 addendum for the lead_sprint_v2 prefix (append-only: new files, nothing existing changed).
Adds, for the episodes Edmond flagged (1, 6, 20, 25, 37) and ep5: a training scene built with the team loader on our
objects + the seated support, with hands chosen by the robot-link contact comparison (exclusive re-attach when it keeps
every contact hand within 0.05 of the control; otherwise the control scene = v1 objects + fb11 hands + seated support),
plus the control checks and the comparison table. Writes stage_addendum/ and UPLOAD_DRY_RUN_ADDENDUM.json."""
from pathlib import Path
import json, shutil, hashlib, datetime, sys
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'; L=OPS/'experiments_20261103'
W=Path(__file__).resolve().parent; V1S=ROOT/'work/lead-sprint-release-20261103/stage'
HAND=L/'lead_hand_reattach_v1'; SEAT=L/'lead_support_seating_v1'
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_sprint_v2_20261103'
v2=json.loads((W/'UPLOAD_VERIFIED.json').read_text()); V2_COMMIT=v2['commit']; existing={r['relative'] for r in v2['sha256']}
for extra in W.glob('UPLOAD_VERIFIED_ADDENDUM*.json'): existing|={r['relative'] for r in json.loads(extra.read_text())['sha256']}
STAGE_NAME=sys.argv[1] if len(sys.argv)>1 else 'stage_addendum'
EPISODES=[int(a) for a in sys.argv[2:]] if len(sys.argv)>2 else [1,6,20,25,37,5]; TOL=0.05
S=W/STAGE_NAME
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
LOCAL_ROOTS=[str(HAND),str(SEAT),str(V1S),str(ROOT)]
def relocal(x):
    if isinstance(x,dict): return {k:relocal(v) for k,v in x.items()}
    if isinstance(x,list): return [relocal(v) for v in x]
    if isinstance(x,str):
        for r in LOCAL_ROOTS:
            if x.startswith(r): return '<local>'+x[len(r):]
    return x
def strip_tag(k):
    for pre in ('control_','rawhands_','rawclean_','rawexcl_'):
        if k.startswith(pre): return k[len(pre):]
    return k
def contact(ep,mode): return {strip_tag(k):v for k,v in read(HAND/'scenes'/f'ep{ep:02d}_{mode}'/'HAND_CONTACT.json')['objects'].items()}
if S.exists(): shutil.rmtree(S)
S.mkdir()
summary={}; md=['| 段 | 交付的双手 | 判据（对照 → 交付，机器人最近连杆 ≤1 cm 帧占比，按物体/手） |','|---|---|---|']
for ep in EPISODES:
    sd=S/'scenes'/f'episode_{ep:06d}'; ctrl=contact(ep,'control')
    cands=['rawexcl','control'] if (HAND/'scenes'/f'ep{ep:02d}_rawexcl'/'HAND_CONTACT.json').exists() else ['control']
    chosen='control'; detail=[]
    if 'rawexcl' in cands:
        ex=contact(ep,'rawexcl'); worst=0.0
        for obj,c in ctrl.items():
            for side in ('left','right'):
                cv=(c.get(side) or {}).get('within_1cm'); ev=(ex.get(obj,{}).get(side) or {}).get('within_1cm')
                if cv is None or ev is None: continue
                detail.append(f'{obj}/{side} {cv:.2f}→{ev:.2f}')
                if cv>=0.3: worst=min(worst,ev-cv)
        chosen='rawexcl' if worst>=-TOL else 'control'
    sc=HAND/'scenes'/f'ep{ep:02d}_{chosen}'
    shutil.copytree(sc,sd/'scene',symlinks=False,ignore=shutil.ignore_patterns('*.generated_by_loader'))
    for f in list((sd/'scene').rglob('*.json')): put(f,relocal(read(f)))
    if chosen=='rawexcl':
        shutil.copytree(HAND/'bundles'/f'ep{ep:02d}_rawhands',sd/'bundle_rawhands'); shutil.copytree(HAND/'bundles'/f'ep{ep:02d}_rawexcl',sd/'bundle_rawexcl',symlinks=False)
        for f in list((sd/'bundle_rawhands').rglob('*.json'))+list((sd/'bundle_rawexcl').rglob('*.json')): put(f,relocal(read(f)))
        b0=np.load(V1S/f'scenes/episode_{ep:06d}/bundle/result.npz',allow_pickle=False); b1=np.load(sd/'bundle_rawexcl/result.npz',allow_pickle=False)
        keys=[k for k in b0.files if k.endswith('_to_world_transform') and not k.startswith('hand_')]
        assert all(np.allclose(b0[k],b1[k],atol=1e-6) for k in keys),(ep,'object/camera world poses changed')
    put(sd/'CONTROL_OUTPUT_CHECK.json',relocal(read(HAND/'scenes'/f'ep{ep:02d}_control'/'OUTPUT_CHECK.json'))); put(sd/'CONTROL_HAND_CONTACT.json',read(HAND/'scenes'/f'ep{ep:02d}_control'/'HAND_CONTACT.json'))
    rec=dict(episode=ep,release='lead_sprint_v2 addendum (P0, 2026-11-03)',objects='identical to lead_sprint_v1 / v2 bundle/',support='seated support_v1 of v2 installed in the scene',
             hands='exclusive re-attach of the team raw hands (bundle_rawexcl/ is the scene input)' if chosen=='rawexcl' else 'fb11 hands as delivered (control): re-attach did not keep every contact hand within %.2f of the control'%TOL,
             shipped_variant=chosen,contact_rule=f'ship rawexcl only if, for every (object, hand) with control within_1cm >= 0.3, rawexcl within_1cm >= control - {TOL}',contact_detail=detail,
             scene='scene/ (processed parquet + meshes + URDF + seated support USD; absolute build paths: run tools/relocate_scene.py)',policy_tested=False)
    put(sd/'HANDS_ADDENDUM.json',rec); summary[ep]=rec
    md.append(f"| {ep} | {'排他式重 attach（原始双手）' if chosen=='rawexcl' else 'fb11 双手（对照）'} | "+'；'.join(detail)+' |')
    print('ADDENDUM',ep,chosen,detail,flush=True)
TAG=STAGE_NAME.replace('stage_addendum','P0').replace('stage_','')
shutil.copy2(HAND/'HAND_REATTACH_COMPARISON.md',S/f'HAND_REATTACH_COMPARISON_{TAG}.md'); put(S/f'HAND_REATTACH_COMPARISON_{TAG}.json',read(HAND/'HAND_REATTACH_COMPARISON.json'))
readme=f'''# lead_sprint_v2 追加（{TAG}，2026-11-03/04）：{'Edmond 点名段' if TAG=='P0' else '其余段'}的可训练场景

本次只新增文件，v2 原有文件未动（提交 {V2_COMMIT[:8]} 之后追加）。针对群里点名的 1/6/20/25/37（我们场景上 tracking 0.55–0.74 的段）和 ep5（bob1 里用的是我们 fb11 版），每段新增 `scenes/episode_XXXXXX/scene/`：物体与 v1/v2 的 `bundle/` 相同，桌面用 v2 坐实后的 support_v1，双手按机器人连杆接触率对照选择：

| 段 | 交付的双手 | 判据 |
|---|---|---|
{chr(10).join(md[2:])}

规则：原始双手经团队 loader 清理平滑，再做排他式 attach（每只手只贴它握的那个物体）；只有当每一只在对照里有接触的手（≤1 cm 帧占比 ≥0.3）在重 attach 后不低于对照 −{TOL:.2f} 时才交付重 attach 版，否则交付对照版（v1 物体 + fb11 双手 + 坐实桌面）。四种做法的完整对比在 `HAND_REATTACH_COMPARISON_P0.md`；每段 `HANDS_ADDENDUM.json` 记录选择与数字；`CONTROL_*` 是对照的检查。

用法与 v2 相同：下载后 `python tools/relocate_scene.py scenes/episode_0000XX/scene`，用 scene 训练。请用同一配方跑这些段的 tracking，与 Ken 场景的数字对照；ep33 建议先做单段 Kaggle 探针（bob1 只换 ep33）。

未做物理回放与 policy 训练；不是正式提交；无 GT。
'''
(S/f'README_ADDENDUM_{TAG}.md' if TAG!='P0' else S/'README_ADDENDUM.md').write_text(readme); put(S/f'SUMMARY_ADDENDUM_{TAG}.json' if TAG!='P0' else S/'SUMMARY_ADDENDUM.json',dict(base_commit=V2_COMMIT,episodes=summary,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files=[p for p in sorted(S.rglob('*')) if p.is_file()]
ALLOWED=['.json','.csv','.md','.npz','.obj','.mtl','.png','.jpg','.jpeg','.py','.usda','.mp4','.parquet','.stl','.urdf','.log','.txt']
for p in files:
    assert p.suffix.lower() in ALLOWED,p; rel=str(p.relative_to(S)); assert rel not in existing,('would overwrite an existing v2 file',rel)
    if p.suffix in ['.json','.md'] and 'scene' not in p.parts: assert not any(s in p.read_text() for s in ['/Volumes/','/Users/','185.216.']),('local path leak',p)
manifest=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
SUMS=f'SHA256SUMS_ADDENDUM_{TAG}.txt' if TAG!='P0' else 'SHA256SUMS_ADDENDUM.txt'
(S/SUMS).write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in manifest)); manifest.append(dict(relative=SUMS,sha256=sha(S/SUMS),bytes=(S/SUMS).stat().st_size))
put(W/(f'UPLOAD_DRY_RUN_ADDENDUM_{TAG}.json' if TAG!='P0' else 'UPLOAD_DRY_RUN_ADDENDUM.json'),dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,expected_parent=V2_COMMIT,stage=str(S),files=manifest,bytes=sum(x['bytes'] for x in manifest),episodes={str(k):v['shipped_variant'] for k,v in summary.items()},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('DRY_RUN addendum files',len(manifest),'bytes',sum(x['bytes'] for x in manifest),{k:v['shipped_variant'] for k,v in summary.items()})
