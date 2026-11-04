# lead_sprint_bundle_v1 — Scored20 冲刺候选集（最终版 2026-11-03 13:55 UTC）

> **2026-11-03 15:45 UTC 更新：** 已上传 HF 新前缀 `experiments/wentacc/scored20_lead_sprint_v1_20261103`（提交 `9f8d352e`，`work/lead-sprint-release-20261103/UPLOAD_VERIFIED.json`）。上传集只含 ep05 / ep32 / ep33 的改动（4 个物体）；本目录里 ep10 蓝杯与 ep19 扫帚/簸箕的 +0.0023 改动**未上传**——Public20 校准（`work/reports/20261103-lead-sprint/METRIC_CALIBRATION.json`）表明代理差 < 0.01 不可区分，交付门槛改为 0.01。上传集 19 段代理均值 0.76998 → 0.77746。对照视频：`videos/episode_XXXXXX/`（render_videos.py）。

基线：HF fb11（`work/scored11-fulltrack-20261103/release`）。只替换按预注册规则通过且全帧代理增益 ≥ 0.002 的物体（mesh 与位姿成套），其余 NPZ 字段、双手、相机逐字节不变。**未上传 HF。**

## 改动（7 个物体，5 段）

| 段 | 物体 | 来源 case | 全帧代理 前 → 后 |
|---|---|---|---:|
| ep05 | wooden_spoon | lead_retrack_recovery_v5/ep05_wooden_spoon__recovery_cont_donor36 | 0.7452 → 0.7909 (+0.0457) |
| ep10 | blue_cup | lead_retrack_recovery_v6/ep10_blue_cup__recovery_cont_donor20 | 0.5702 → 0.5725 (+0.0023) |
| ep19 | mini_sweeper | lead_retrack_recovery_v3/ep19_mini_sweeper__recovery_cont_own | 0.7194 → 0.7218 (+0.0023) |
| ep19 | mini_dust_pan | lead_retrack_recovery_v4/ep19_mini_dust_pan__recovery_cont_own | 0.8170 → 0.8193 (+0.0023) |
| ep32 | wooden_spoon | lead_retrack_recovery_v7/ep32_wooden_spoon__recovery_cont_donor37 | 0.6712 → 0.6911 (+0.0199) |
| ep33 | blue_cup | lead_retrack_recovery_v2/ep33_blue_cup__recovery_last | 0.7374 → 0.7552 (+0.0178) |
| ep33 | wooden_spoon | lead_retrack_recovery_v3/ep33_wooden_spoon__recovery_cont_donor36 | 0.4572 → 0.6579 (+0.2006) |

19 段代理均值 **0.7700 → 0.7776（+0.0077）**；逐段：ep05 0.7586→0.7815；ep10 0.6653→0.6665；ep19 0.7682→0.7705；ep32 0.7065→0.7164；ep33 0.5973→0.7066。

## 来源与规则

- 7 个 GPU 批次共 103 个评价 case（`ALL_CASES.md` / `ALL_CASES.csv`；各批次 `lead_retrack_recovery_v{1..7}/RESULTS*.md`）。
- 选择规则（FROZEN_BATCH amendments 记录修订）：代理严格更高；最大旋转步 ≤ 90°；>30° 的步 ≤ 10% 帧；平移 p95 ≤ 50 mm/帧、最大 ≤ 120 mm/帧；无无效帧；杯用轴倾斜步、平移用质心。交付门槛：增益 ≥ 0.002。无 GT。
- 消费链：改动段 [5, 10, 19, 32, 33] 全部 `FULL_NATIVE_CONSUMER_PASS`（loader/retarget 退出 0，`.venv-bundle-cpu`）；support 重建退出 1，与此前交付一致。失败：无。CONSUMER_RESULTS.json。
- 叠加图（人工看过）：ep33 勺、ep33 杯、ep05 勺、ep32 勺 的 `run_v1/cases/<case>/overlay_montage.png`。

## 结论

1. 三个实质增益都来自 **换 mesh**：ep33 勺（Ken mesh 短 12% → ep36 勺 mesh，+0.20）、ep05 勺（ep36 mesh，+0.046）、ep32 勺（ep37 mesh，+0.020）。跟踪器变体（plain recovery、粒子、连续性 recovery、晚锚点）在其余物体上都是 ±0.003 的噪声；donor mesh 用在杯类和 ep03/15/16/35/37 的勺上无益或退步。
2. 仍未突破：ep10 蓝杯 0.57（套叠拿起段）、ep28 锅 0.70（形状/常量偏差）、ep19 扫帚 0.72。手-物刚性先验（lead_grasp_prior_v1）被否定。

## 注意

1. ep33、ep05 的勺 mesh 替换为 ep36 的勺 mesh，ep32 的勺 mesh 替换为 ep37 的；提交 CD-O 时这些段的 mesh 随之改变（同一实物，自家重建）。
2. 7 个交付包（ep01/03/06/14/15/19/35）原本缺材质文件，这里以 `lead_fb11_repaired` 的中性材质副本为基底（几何、位姿不变）；ep10/ep32 旧 schema 补了次物体 camera 位姿/validity 字段。
3. 代理不是官方 AUC；policy 训练效果未测。GPU 窗口回执：WINDOW_RECEIPT.json。
