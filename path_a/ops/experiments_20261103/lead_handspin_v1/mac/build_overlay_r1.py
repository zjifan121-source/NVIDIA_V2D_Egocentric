"""lead_v3 hands overlay r1: corrected hands (and cup spin) for the episodes where v3 was wrong or could be better. Objects unchanged:
every non-cup object array and every mesh equals its parent (lead_v3, or the ep28 pot overlay); a cup's position and tilt are unchanged,
only its unobservable axial spin differs. Writes stage/ + UPLOAD_DRY_RUN.json."""
from pathlib import Path
import json, shutil, hashlib, datetime, sys
import numpy as np
W=Path(__file__).resolve().parent; SRC=W/'overlay_r1_src'; S=W/'overlay_r1_stage'; V3=W/'stage/scenes'
ROOT=Path('/Volumes/E/Code/V2D'); P28=ROOT/'work/lead-v3-release-20261104/overlay_r1_src/ep28pot_parent'
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_v3_hands_r1_20261104'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
ITEMS=[ # episode, source tag, folder name, parent bundle, why
 (8,'ep08_rawspin','episode_000008','v3','杯自旋修正（蓝本 v3 手保持）'),
 (15,'ep15_hold','episode_000015','v3','去掉右手假抓握'),
 (16,'ep16_hold','episode_000016','v3','去掉左手假抓握；右手空缺帧由上一个持有者补'),
 (25,'ep25_hold','episode_000025','v3','木板 2 右手捏合修复（Edmond 的工具，加转动），左手保持到松手帧'),
 (28,'ep28_hold_newpot','episode_000028','ep28pot','新锅 + 去掉左手假抓握'),
 (28,'oldpot','episode_000028_oldpot','v3','旧锅 + 去掉左手假抓握'),
 (32,'ep32_rawspin','episode_000032','v3','杯自旋修正'),
 (33,'ep33_rawspin','episode_000033','v3','杯自旋修正'),
 (35,'ep35_hold','episode_000035','v3','逐帧持有重贴（无 raw 手，用交付手）'),
 (36,'ep36_hold','episode_000036','v3','逐帧持有重贴（无 raw 手，用交付手），左手不再硬绑'),
 (37,'ep37_hold','episode_000037','v3','逐帧持有重贴（无 raw 手，用交付手）')]
only=[a for a in sys.argv[1:]]
if S.exists(): shutil.rmtree(S)
(S/'scenes').mkdir(parents=True); rows=[]
LOCAL=['/root/v2d/lead_v3/handspin28','/root/v2d/lead_v3/handspin25','/root/v2d/lead_v3/handspin','/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1','/root/v2d','/Volumes/E/Code/V2D/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1','/Volumes/E/Code/V2D']
def scrub(t):
    for r in LOCAL: t=t.replace(r,'<local>')
    return t
for ep,tag,name,parent,why in ITEMS:
    if tag=='oldpot': sb,ss=SRC/'oldpot/bundles/ep28_hold',SRC/'oldpot/scenes/ep28_hold'
    elif tag=='ep25_hold': sb,ss=SRC/'ep25/bundles/ep25_hold',SRC/'ep25/scenes/ep25_hold'
    else: sb,ss=SRC/'bundles'/tag,SRC/'scenes'/tag
    if not (sb/'result.npz').exists(): print('SKIP',tag,'not available'); continue
    parent_b=(V3/f'episode_{ep:06d}/bundle') if parent=='v3' else P28/f'scenes/episode_{ep:06d}/bundle'
    a=np.load(sb/'result.npz',allow_pickle=False); b=np.load(parent_b/'result.npz',allow_pickle=False)
    names=[str(x) for x in b['object_names']] if 'object_names' in b.files else [None]
    keys=['object_to_world_transform']+[f'object_{i}_to_world_transform' for i in range(1,len(names))]
    cup=[]
    for k,n in zip(keys,names):
        if n in ('blue_cup','beige_cup'):
            cup.append(n); assert np.allclose(a[k][0],b[k][0],atol=1e-6),(ep,k,'frame 0')
        else: assert np.allclose(a[k],b[k],atol=1e-6),(ep,k,'object changed')
    assert np.allclose(a['camera_to_world_transform'],b['camera_to_world_transform'],atol=1e-6),(ep,'camera')
    for m in [f for f in parent_b.iterdir() if f.suffix in ('.obj','.mtl','.png')]:
        q=sb/m.name
        assert q.exists() and sha(q)==sha(m),(ep,m.name)
    d=S/'scenes'/name; shutil.copytree(sb,d/'bundle',symlinks=False); shutil.copytree(ss,d/'scene',symlinks=False,ignore=shutil.ignore_patterns('*.generated_by_loader','relocate.log','.relocated','*.orig'))
    for f in list(d.rglob('*.json'))+list(d.rglob('*.log')): f.write_text(scrub(f.read_text()))
    shutil.copytree((V3/f'episode_{ep:06d}/support_v1') if parent=='v3' else P28/f'scenes/episode_{ep:06d}/support_v1',d/'support_v1')
    att=next((read(x) for x in [sb/'ATTACH_HOLD.json',sb/'ATTACH_RESPIN.json'] if x.exists()),{})
    rep=read(sb/'contact_repair.json') if (sb/'contact_repair.json').exists() else None
    chk=read(ss/'OUTPUT_CHECK.json'); ct=read(ss/'HAND_CONTACT.json') if (ss/'HAND_CONTACT.json').exists() else {}
    prov=dict(episode=ep,change=why,parent=('lead_v3 commit 5e7601d1' if parent=='v3' else 'lead_v3 ep28 overlay'),objects_unchanged=True,cup_spin_replaced=cup,holds=att.get('holds_final',att.get('assignment')),gaps_filled=att.get('gaps_filled'),hand_moved_cm=att.get('hand_moved_cm'),respin=att.get('respin'),pinch_repair=({k:rep[k] for k in ('side','object','hold','anchor','translation_cm','pinch_rotation_deg','pinch_normal_alignment') if k in rep} if rep else None),
        scene_check={k:chk.get(k) for k in ('task_error_cm','penetration_frames','frame0_seating_cm')},robot_contact=ct.get('objects'),policy_tested=False)
    put(d/'PROVENANCE.json',json.loads(scrub(json.dumps(prov,ensure_ascii=False))))
    rows.append(dict(episode=ep,folder=name,parent=parent,change=why)); print('OK',name,cup)
for t in ['install_support.py','relocate_scene.py']:
    (S/'tools').mkdir(exist_ok=True); shutil.copy2(W/'stage/tools'/t,S/'tools'/t)
shutil.copy2(W/'attach_tools/attach_hold.py',S/'tools/attach_hold.py'); shutil.copy2(W/'attach_tools/repair_grip_align.py',S/'tools/repair_grip_align.py')
put(S/'inventory.json',rows)
(S/'README.md').write_text('''# lead_v3 手修正覆盖 r1（2026-11-04）

只含 lead_v3 里需要修正的段，其余段继续用 lead_v3（`experiments/wentacc/scored20_lead_v3_20261104`）。**用哪一段、哪个 scene，以同目录上一级的 `lead_current/CURRENT_MANIFEST_r1.json` 为准。**

为什么要修：lead_v3 的排他式贴手只要手靠近过物体一次就整段绑上去，有几段的手其实离物体很远（ep28 左手贴之前中位 40.7 cm），之后量到的贴合率是循环的。新规则（`tools/attach_hold.py`）用 Edmond de66fea 的无 mask 规则：手只在够得着的帧持有物体，一只手同一时间只持有一个物体；物体被抬起而没有手的帧交给上一个持有者。

杯子：蓝杯和米杯的轴向自旋官方 AUC 不计，参考里的自旋是 FoundationPose 噪声。按 Ken 的做法改成跟随握杯手，杯子位置和倾斜不变。

ep25：木板 2 被抬起时没有手够得着。用 Edmond 的 `repair_hand_contact.py`（握持窗口 150–333，他逐帧看过视频）把右手捏到木板上，并增加了转动，因为我们的右手朝向不同（`tools/repair_grip_align.py`）。

物体未改：构建脚本断言了非杯物体与相机轨迹、全部 mesh 与父版本一致；杯子只断言第 0 帧一致。每段 `PROVENANCE.json` 记录手的持有窗口、手被移动的距离、机器人接触和 loader 检查。全部 `policy_tested: false`，真分以账本为准。无 GT。
''')
ALLOWED=['.json','.csv','.md','.npz','.npy','.obj','.mtl','.png','.jpg','.py','.usda','.parquet','.stl','.urdf','.log','.txt']
files=[p for p in sorted(S.rglob('*')) if p.is_file()]
for p in files: assert p.suffix.lower() in ALLOWED,p
leak=[str(p) for p in files if p.suffix in ('.json','.md','.csv','.log') and any(x in p.read_text(errors='ignore') for x in ['/Volumes/','/root/v2d','185.216.','223.109.'])]
assert not leak,leak[:5]
manifest=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
(S/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in manifest)); manifest.append(dict(relative='SHA256SUMS.txt',sha256=sha(S/'SHA256SUMS.txt'),bytes=(S/'SHA256SUMS.txt').stat().st_size))
put(W/'UPLOAD_DRY_RUN_overlay_r1.json',dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,stage=str(S),files=manifest,bytes=sum(x['bytes'] for x in manifest),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('DRY_RUN files',len(manifest),'bytes',sum(x['bytes'] for x in manifest))
