"""Build the HF stage for lead sprint v2 = lead_sprint_v1 objects (bundles byte-identical to the v1 upload) + seated
support_v1 for all 20 episodes (ep10/ep38 gain a support) + raw-hands re-attach scenes for ep05/ep32/ep33 (team loader,
haisenk's recipe) with their control builds' checks. New independent prefix; no GT; nothing on HF is modified.
Writes stage/ and UPLOAD_DRY_RUN.json. Re-runnable."""
from pathlib import Path
import json, csv, shutil, hashlib, datetime, re, sys
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'; L=OPS/'experiments_20261103'
W=Path(__file__).resolve().parent; S=W/'stage'
V1=ROOT/'work/lead-sprint-release-20261103'; V1S=V1/'stage'; SEAT=L/'lead_support_seating_v1'; HAND=L/'lead_hand_reattach_v1'
HTOOLS=Path('/private/tmp/claude-501/-Volumes-E-Code-V2D/9e418383-d795-43bf-92a7-b5721bcd24f8/scratchpad/haisenk_tools/experiments/haisenk/hand_fixes_wentao_v1_20261103/tools')
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_sprint_v2_20261103'
V1_PREFIX='experiments/wentacc/scored20_lead_sprint_v1_20261103'; FB11_PREFIX='experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103'
HAISEN_PREFIX='experiments/haisenk/hand_fixes_wentao_v1_20261103'
REATTACH=[32,33]; SHIP_MODE='rawexcl'
KEPT={5:'fb11 hands kept: the control already holds the new spoon (spoon/right <= 1 cm on 99.5% of lifted frames) and the re-attached raw hands do slightly worse (93.4%); see HAND_REATTACH_COMPARISON.md'}  # variant shipped as scene/: exclusive attach (each hand rides only the object it holds)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
v1=read(V1/'UPLOAD_VERIFIED.json'); v1sha={r['relative']:r['sha256'] for r in v1['sha256']}; V1_COMMIT=v1['commit']
LOCAL_ROOTS=[str(HAND),str(SEAT),str(V1S),str(ROOT)]
def relocal(x):
    """Replace this machine's absolute paths in JSON records by stage-relative markers."""
    if isinstance(x,dict): return {k:relocal(v) for k,v in x.items()}
    if isinstance(x,list): return [relocal(v) for v in x]
    if isinstance(x,str):
        for r in LOCAL_ROOTS:
            if x.startswith(r): return '<local>'+x[len(r):]
    return x
if S.exists(): shutil.rmtree(S)
(S/'scenes').mkdir(parents=True)
seat_summary={e['episode']:e for e in read(SEAT/'SEATING_SUMMARY.json')['episodes']}
rows=[]; hand_summary={}
for ep_dir in sorted((V1S/'scenes').iterdir()):
    ep=int(ep_dir.name.split('_')[-1]); sd=S/'scenes'/ep_dir.name; run=read(ep_dir/'run.json')
    shutil.copytree(ep_dir/'bundle',sd/'bundle')
    for p in (sd/'bundle').rglob('*'):
        if p.is_file(): assert v1sha.get(str(p.relative_to(S)))==sha(p),('bundle must equal the v1 upload',ep,p.name)
    if (ep_dir/'videos').exists(): shutil.copytree(ep_dir/'videos',sd/'videos')
    # seated support
    src=SEAT/'out'/ep_dir.name/'support_v1'; assert src.exists(),('no seated support',ep)
    shutil.copytree(src,sd/'support_v1')
    spec=read(sd/'support_v1/support_in_bundle_world.json'); val=read(sd/'support_v1/VALIDATION.json')
    for n,h in spec['bundle_sha256'].items(): assert sha(sd/'bundle'/n)==h,('seated spec vs staged bundle',ep,n)
    if (ep_dir/'support_v1').exists(): shutil.copytree(ep_dir/'support_v1',sd/'support_v1_fb11_unseated')
    rec=dict(run); rec.update(release='lead_sprint_v2',base=dict(repo=REPO,prefix=V1_PREFIX,revision=V1_COMMIT,note='bundle bytes identical to lead_sprint_v1 (objects) = fb11 except ep05/ep32/ep33 objects'),
        support_v1=dict(state=val['state'],origin=spec['seating']['origin'],table_top_before=spec['seating']['table_top_before'],table_top_after=spec['seating']['table_top_after'],lowered=spec['seating']['lowered'],disks=spec['seating']['disks'],skipped_disks=spec['seating']['skipped_disks'],warnings=spec['seating']['warnings'],frame0_penetration=val['frame0'],max_depth_first10_cm=val['max_depth_first10_cm'],collision_shapes=val['collision_shapes'],previous_unseated='support_v1_fb11_unseated/' if (ep_dir/'support_v1').exists() else None))
    # hand re-attach scenes
    if ep in REATTACH:
        for mode in [SHIP_MODE,'control']:
            sc=HAND/'scenes'/f'ep{ep:02d}_{mode}'; chk=read(sc/'OUTPUT_CHECK.json'); assert chk['frames']==rec['frames'],(ep,mode,'frames')
            if mode==SHIP_MODE:
                dest=sd/'scene'; shutil.copytree(sc,dest,symlinks=False,ignore=shutil.ignore_patterns('*.generated_by_loader'))
                for f in list(dest.rglob('*.json')):
                    put(f,relocal(read(f)))
                shutil.copytree(HAND/'bundles'/f'ep{ep:02d}_rawhands',sd/'bundle_rawhands'); put(sd/'bundle_rawhands/HANDFIX_PREP.json',relocal(read(sd/'bundle_rawhands/HANDFIX_PREP.json')))
                if SHIP_MODE=='rawexcl': shutil.copytree(HAND/'bundles'/f'ep{ep:02d}_rawexcl',sd/'bundle_rawexcl',symlinks=False)
                for f in list((sd/'bundle_rawhands').rglob('*.json'))+list((sd/'bundle_rawexcl').rglob('*.json') if (sd/'bundle_rawexcl').exists() else []): put(f,relocal(read(f)))
                shutil.copy2(HAND/'HAND_REATTACH_COMPARISON.json',S/'HAND_REATTACH_COMPARISON.json'); shutil.copy2(HAND/'HAND_REATTACH_COMPARISON.md',S/'HAND_REATTACH_COMPARISON.md')
                b0=np.load(sd/'bundle/result.npz',allow_pickle=False); b1=np.load(sd/'bundle_rawhands/result.npz',allow_pickle=False)
                assert all(np.array_equal(b0[k],b1[k]) for k in b0.files if not k.startswith('hand_')),(ep,'rawhands bundle changed non-hand arrays')
            else:
                put(sd/'CONTROL_OUTPUT_CHECK.json',relocal(chk))
            contact=read(sc/'HAND_CONTACT.json')
            if mode=='rawhands': shutil.copy2(sc/'HAND_CONTACT.json',sd/'scene/HAND_CONTACT.json')
            else: put(sd/'CONTROL_HAND_CONTACT.json',contact)
            hand_summary.setdefault(ep,{})['shipped' if mode==SHIP_MODE else mode]={**{k:chk[k] for k in ['cleaning','task_error_cm','penetration_frames','frame0_seating_cm','frame0_support_penetration_cm','hand_object_proximity']},'contact':contact['objects']}
        rec['hands']=dict(state='RAW_HANDS_REATTACHED_TO_NEW_OBJECT_TRACKS',shipped_variant=SHIP_MODE,recipe=('raw hands (night_2026-10-24 ken_c_scored_attach/bundle_merged, re-expressed in our world) -> team loader cleaning --no-rest-hold --no-object-clean --no-level --smooth-hands 2 (fill + smooth, no attach) -> tools/attach_exclusive.py (loader attach math, each hand assigned to the one object it holds) -> team loader build with no further cleaning' if SHIP_MODE=='rawexcl' else 'team loader --no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2 (haisenk recipe)'),
            scene='scene/ (processed parquet + meshes + URDF + seated support USD + bundle_cleaned with attached hands)',bundle_dir_hands='bundle/ keeps the fb11 hands (unchanged, attached to the OLD object tracks); bundle_rawhands/ is the loader input',checks=hand_summary[ep])
    elif ep in KEPT:
        rec['hands']=dict(state='FB11_HANDS_KEPT_AFTER_COMPARISON',note=KEPT[ep])
    else:
        rec['hands']=dict(state='FB11_HANDS_UNCHANGED',note='objects identical to fb11; haisenk hand fixes exist for ep1/3/6/10/14/20 in '+HAISEN_PREFIX)
    put(sd/'run.json',rec)
    rows.append(dict(episode=ep,frames=rec['frames'],objects=rec['objects'],objects_selection=run['selection'],selected_full_proxy=rec.get('selected_full_proxy'),
        support_state=val['state'],table_lowered_cm=[l['by_cm'] for l in spec['seating']['lowered']],seat_disks=len(spec['seating']['disks']),hands='reattached_scene' if ep in REATTACH else ('fb11_kept_after_comparison' if ep in KEPT else 'fb11')))
    print('STAGED',ep,'support',val['state'],'hands','REATTACH' if ep in REATTACH else 'fb11',flush=True)
put(S/'inventory.json',rows)
with (S/'inventory.csv').open('w') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
(S/'tools').mkdir(); shutil.copy2(V1S/'tools/install_support.py',S/'tools/install_support.py')
shutil.copy2(SEAT/'seat_supports.py',S/'tools/seat_supports.py'); shutil.copy2(HAND/'prep_rawhands.py',S/'tools/prep_rawhands.py'); shutil.copy2(HAND/'build_scene.py',S/'tools/build_scene.py'); shutil.copy2(HAND/'attach_exclusive.py',S/'tools/attach_exclusive.py'); shutil.copy2(HAND/'hand_contact.py',S/'tools/hand_contact.py')
shutil.copy2(HTOOLS/'relocate_scene.py',S/'tools/relocate_scene.py')  # haisenk's path relocator for shipped scenes (unchanged copy, credited in README)
put(S/'SEATING_SUMMARY.json',relocal(read(SEAT/'SEATING_SUMMARY.json'))); shutil.copy2(SEAT/'SEATING_TABLE.md',S/'SEATING_TABLE.md'); put(S/'SEAT_AUDIT_V1.json',relocal(read(SEAT/'SEAT_AUDIT.json')))
put(S/'HAND_REATTACH_SUMMARY.json',hand_summary)
def strip_tag(k):
    for pre in ('control_','rawhands_','rawclean_','rawexcl_'):
        if k.startswith(pre): return k[len(pre):]
    return k
def fmt_hand(ep):
    h=hand_summary[ep]; out=[]; ctrl={strip_tag(k):v for k,v in h['control']['contact'].items()}
    for obj,v in {strip_tag(k):v for k,v in h['shipped']['contact'].items()}.items():
        c=ctrl[obj]
        for side in ('left','right'):
            if v.get(side) and c.get(side): out.append(f"{obj}/{side}: 举起 {v['lifted_frames']} 帧，最近连杆到物体面中位 {c[side]['closest_link_to_surface_cm_median']} → {v[side]['closest_link_to_surface_cm_median']} cm，≤1 cm 帧占比 {c[side]['within_1cm']} → {v[side]['within_1cm']}")
    te=lambda m:f"右 {h[m]['task_error_cm']['right']['mean']}/{h[m]['task_error_cm']['right']['max']}，左 {h[m]['task_error_cm']['left']['mean']}/{h[m]['task_error_cm']['left']['max']}"
    pen=lambda m:f"{h[m]['penetration_frames']['any']} / {h[m]['penetration_frames']['over_1cm']} / {h[m]['penetration_frames']['over_2cm']}"
    return out,te,pen
seat_rows=''
for r in rows:
    e=seat_summary[r['episode']]; seat_rows+=f"| {r['episode']} | {'fb11' if e['origin'].startswith('fb11') else '历史场景（新）'} | {e['table_top_before']:+.4f} → {e['table_top_after']:+.4f} | {', '.join(str(x) for x in e['lowered_cm']) or '0'} | {', '.join(f'{d['object']} {d['gap_cm']} cm / r {d['radius_cm']} cm' for d in e['disks']) or '—'} | {', '.join(s['object'] for s in e['skipped']) or '—'} | {max(e['max_depth_first10_cm'].values()):.2f} |\n"
hand_rows=''
for ep in REATTACH:
    out,te,pen=fmt_hand(ep); h=hand_summary[ep]
    rc=read(HAND/'scenes'/f'ep{ep:02d}_rawclean/OUTPUT_CHECK.json')['cleaning'] or {}; ax=read(HAND/'bundles'/f'ep{ep:02d}_rawexcl/ATTACH_EXCLUSIVE.json')
    cl=f"补帧 左 {rc.get('hand_left_refilled_frames')} / 右 {rc.get('hand_right_refilled_frames')}，平滑 σ {rc.get('hand_smooth_sigma_frames')}；排他 attach：" + '，'.join(f"{side}→object_{o} 帧 {a}–{b}（偏移 {d} cm）" for o,lst in ax['attached'].items() for side,a,b,d in [(x[0],x[1],x[2],x[3]) for x in lst] for o in [o.split('_')[-1]])
    hand_rows+=f"| {ep} | {te('control')} | {te('shipped')} | {pen('control')} | {pen('shipped')} | {h['shipped']['frame0_seating_cm']} | "+'；'.join(out)+f" | {cl} |\n"
readme=f'''# Scored20 lead sprint v2：桌面坐实（20 段）+ 新勺轨迹上的双手重 attach（ep32/33）

新的独立前缀，只新增文件。物体层与 `{V1_PREFIX}`（提交 {V1_COMMIT[:8]}）**逐字节相同**：每段 `bundle/` 就是 v1 的 bundle（构建脚本逐文件断言等于 v1 上传回执）。本次只做两件场景级的事：

1. **桌面坐实**：全部 20 段的 `support_v1/support_in_bundle_world.json` 重新生成（schema 2）。规则与 haisenk 的 `wseat.py`（2026-11-03）相同：桌面顶 T = frame 0 各物体最低点 − 2 mm，只降不升；悬空 > 5 mm 的物体加一片薄垫（高 1 cm，顶 = 物体底 − 2 mm，半径取其最低 1 cm 的脚印、限 3–15 cm）。本实现另加：叠放在其他物体上的不加垫（ep01 锅盖在锅上、ep03 勺在锅上、ep28 架子与锅、ep30 扫帚靠架子）；垫片若切到邻居则缩小或放弃；逐段 frame 0 穿透检查写在 `VALIDATION.json`（20 段 frame 0 穿透均为 0，前 10 帧最大 0.23 cm）。ep10、ep38 在 fb11 里没有 support_v1，这里用同段历史场景的支撑经同一共同变换带入 bundle 世界后再坐实（`seating.origin`）。旧的未坐实 spec 保留在 `support_v1_fb11_unseated/`。
2. **双手重 attach**：ep05/ep32/ep33 的勺轨迹在 v1 里换了，fb11 的双手是贴在旧轨迹上的。从团队原始双手（`night_2026-10-24/.../ken_c_scored_attach/bundle_merged`，相机系参数在我们的世界里重新表达，`bundle_rawhands/`）出发，先用团队 loader 做手部清理与平滑（`--no-rest-hold --no-object-clean --no-level --smooth-hands 2`，不 attach），再用 `tools/attach_exclusive.py` 做**排他式 attach**（loader 自带 `attach_hands` 的数学，但每只手只贴它握着的那一个物体：左手→杯/锅，右手→勺），最后用 loader 不再清理地构建 `scene/`（processed parquet、mesh/URDF、坐实后的 support USD；attach 后的双手 bundle 就是 `bundle_rawexcl/`，即 scene 的输入）。只对 ep32、ep33 交付：ep05 的 fb11 双手本来就贴着新勺（右手 ≤1 cm 帧占比 0.995），重 attach 反而略差（0.934），保留 fb11 双手。为什么不用 loader 默认的 `--attach-hands`：它把每只进入 12 cm 的手都贴到每个被举起的物体上，勺在杯里搅时握杯的左手会被改贴到勺上、离开杯子（ep32 左手到杯的 ≤1 cm 帧占比 0.99 → 0.28）。四种做法的机器人连杆接触率见 `HAND_REATTACH_COMPARISON.md`。对照组（v1 bundle 原样、无清理）只保留检查结果 `CONTROL_OUTPUT_CHECK.json` / `CONTROL_HAND_CONTACT.json`。

## 坐实结果

| 段 | 来源 | 桌面顶 前 → 后 (m) | 降低 (cm) | 新增垫片 | 不加垫（叠放） | 前 10 帧最大穿透 (cm) |
|---|---|---|---|---|---|---|
{seat_rows}
frame 0 的物体底与桌面顶之差 ≥ 2 mm（无穿透）在 20 段全部成立。注意两处疑点随坐实带入：ep19 簸箕 frame 0 比桌面低 6 cm、前 10 帧上升 2.9 cm，ep32 勺 frame 0 低 3.2 cm、前 10 帧上升 2.6 cm——这两条轨迹的起始段本身可疑，坐实只保证 frame 0 不穿模。

## 双手重 attach 结果（loader 输出检查；`任务误差` = 重定向后腕部任务误差 cm 均值/最大；`穿透` = 机器人连杆进入物体的帧占比 任意/>1 cm/>2 cm）

| 段 | 对照 任务误差 | 重 attach 任务误差 | 对照 穿透 | 重 attach 穿透 | frame0 落座 (cm) | 举起段机器人手最近连杆到物体面（对照 → 重 attach） | loader 清理记录 |
|---|---|---|---|---|---|---|---|
{hand_rows}
`bundle/` 里的双手仍是 fb11 的（与 v1 相同），训练请用 `scene/`。scene 的 parquet/URDF 里有构建机器的绝对路径，下载后先 `python tools/relocate_scene.py scenes/episode_0000XX/scene`（haisenk 的脚本，原样收录）。

## 其余说明

- 代理分数、选择规则、校准、视频与 v1 相同（见 v1 前缀 README）。本次不改任何物体 mesh 或轨迹。
- 其余 17 段的双手与 fb11 相同；其中 ep1/3/6/10/14/20 有 haisenk 的手部修复（`{HAISEN_PREFIX}`），可直接用其 scene（物体相同）；但其 support 未按本次 20 段统一坐实，二者取舍见团队决定。
- 工具（原样收录，内含构建机器的绝对路径，按需改）：`tools/seat_supports.py`（坐实）、`tools/prep_rawhands.py`、`tools/attach_exclusive.py`、`tools/build_scene.py`、`tools/hand_contact.py`（重 attach 构建与检查）、`tools/install_support.py`（装支撑）、`tools/relocate_scene.py`（haisenk）。
- 未做物理回放与 policy 训练；`sha256sum -c SHA256SUMS.txt`。不是正式比赛提交。
'''
(S/'README.md').write_text(readme)
summary=dict(state='LEAD_SPRINT_V2_SEATED_SUPPORTS_AND_HAND_REATTACH_NOT_POLICY_VALIDATED',episodes=20,base=dict(repo=REPO,prefix=V1_PREFIX,revision=V1_COMMIT),bundles='byte-identical to lead_sprint_v1',
    supports_seated=20,supports_new=[10,38],hand_reattach_episodes=REATTACH,hand_reattach_compared=[5,32,33],hands_kept=list(KEPT),seating_rule=read(SEAT/'out/episode_000005/support_v1/support_in_bundle_world.json')['seating']['rule'],
    hand_recipe='run_load_retarget.py --no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2 on raw hands re-expressed in our world',official_AUC=None,policy_score=None,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
put(S/'SUMMARY.json',summary)
ALLOWED=['.json','.csv','.md','.npz','.obj','.mtl','.png','.jpg','.jpeg','.py','.usda','.mp4','.parquet','.stl','.urdf','.log','.txt']
files=[p for p in sorted(S.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt']
for p in files:
    assert p.suffix.lower() in ALLOWED,p
    if p.suffix in ['.json','.csv','.md','.py'] and 'scene' not in p.parts and 'tools' not in p.parts:  # tools/ are reference scripts with build-machine paths (like haisenk's)
        t=p.read_text(); assert not any(s in t for s in ['/Volumes/','/Users/','185.216.']),('local path leak',p)
leaky=[str(p.relative_to(S)) for p in files if p.suffix in ['.json','.log','.urdf','.usda','.py','.md'] and any(s in p.read_text(errors='ignore') for s in ['/Volumes/','/Users/'])]
manifest=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
(S/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in manifest))
manifest.append(dict(relative='SHA256SUMS.txt',sha256=sha(S/'SHA256SUMS.txt'),bytes=(S/'SHA256SUMS.txt').stat().st_size))
plan=dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,base_revision=V1_COMMIT,stage=str(S),files=manifest,bytes=sum(x['bytes'] for x in manifest),files_with_build_machine_paths=leaky,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
put(W/'UPLOAD_DRY_RUN.json',plan); print('DRY_RUN files',len(manifest),'bytes',plan['bytes'],'| files still holding build-machine paths (scene internals, relocate_scene.py handles them):',len(leaky))
