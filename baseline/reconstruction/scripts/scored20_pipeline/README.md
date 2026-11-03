# Scored20：候选实验与场景导出

这是 `5454d57` 三个优化器和带桌面接触约束 FP 的统一入口。显式按对象启用模块，
运行期间自动完成数据适配、冻结、优化、匹配新模型的跟踪、时序修正和完整 bundle 导出。
每个对象可显式选择已存在的固定米制 FP、适用的研究优化分支，或保留基线。
不强制每段启用新优化。保留基线与普通 FP 分别记账，不会标成新优化成功。

**实际整批证据：2026-11-02 的 contact-shape→FP 实验尝试20段，9段导出、11段在形状数值验收被拒绝。**
它没有证明20段通用适用或9段质量提高。该入口是实验编排工具，不是经过完整验证的20段优化方案。
CPU接通测试只能证明接口、状态和数据传递；本页不得用测试数量代替实测适用性与质量验收。

输入是**已有前端输出和匹配的场景 bundle**。本入口保留相机、手、桌面和未指定修改的物体；
不重新生成深度、mask、手或相机，也不训练 policy。普通重建入口的默认行为不变。

## 一条完整运行命令

在仓库根目录、CPU 优化依赖环境中运行：

```bash
python baseline/reconstruction/scripts/scored20_pipeline/pipeline.py run \
  --config scored20.json --interactive-reviews
```

固定范围：`1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38`。
先对全部输入做 CPU 检查；`INPUT_CONTRACTS_VALID` 仅表示输入格式/依赖满足，不表示方法适用或保证导出。
有缺项则不启动任何优化/GPU，并列出所有 episode 的问题。
通过后顺序执行。生成的新形状和 FP 锚点会在同一命令内提示查看实际叠加图；检查后输入
`ACCEPT` 或 `REJECT`，无需另开脚本串联。新形状必须先通过原有数值验收，再进入此审阅。
运行应保留交互终端。FP 每个对象的原有 15 分钟窗口包含锚点审阅时间。

只检查输入（输出目录须与正式运行分开）：

```bash
python baseline/reconstruction/scripts/scored20_pipeline/pipeline.py validate \
  --config scored20.json --output runs/scored20_preflight
```

## 环境与最小配置

CPU 使用 Python 3.11；CPU 环境与 FP/CUDA 环境可以是不同的解释器，但必须能看到同一套文件。
CPU 依赖安装：

```bash
python -m pip install -r baseline/reconstruction/scripts/metric_scene_refinement/requirements.txt
```

需要跟踪时，`gpu_python` 指向已安装原生重建依赖的 Linux/CUDA Python；沿用已有
FoundationPose TensorRT 权重/引擎，不在批处理中编译引擎或下载权重。
`runtime_info` 是对应原生 `FoundationPoseTracker.runtime_info` JSON（已有 FP 锚点 receipt
中的同名对象），用于核对 backend、模型、TensorRT 和设备能力。

`scored20.json`（所有路径相对该配置文件）：

```json
{
  "format_version": 1,
  "input_root": "inputs/scored20",
  "output_root": "runs/scored20_refined_v1",
  "runtime": {
    "gpu_python": ".venv-fp/bin/python",
    "source_root": "baseline/reconstruction/modules",
    "weights_dir": "weights",
    "runtime_info": "inputs/fp_runtime.json"
  }
}
```

纯 CPU 的时序修正配方可使用 `"runtime": {}`。有 shape 或 tracking 时需要 FP 环境。

准备目录：

```text
inputs/scored20/episode_000001/
  bundle/                  # result.npz、mesh.obj、mesh_1.obj…、材质、纹理、桌面
  recipe.json              # 每个对象的模块、标注及参数
  objects/slot_0/
    observations/          # RGB、米制深度、该对象 mask、K
    landmarks.npz          # 启用 temporal_pose 时需要
    measurement.json       # 对应标记的输入可观测性验收
  ...
```

其他 19 段使用同样布局。可让入口按已有 bundle 的实际物体槽位生成配方骨架和全局配置：

```bash
python baseline/reconstruction/scripts/scored20_pipeline/pipeline.py init \
  --input-root inputs/scored20 --config-out scored20.json \
  --output-root runs/scored20_refined_v1
```

骨架中的 `accepted=false`，且不默认给所有对象启用某个优化器。
必须明确选择实验分支并填写对应输入，或将对象条目替换成带原因的 `preserve`；不会伪造标注或自动通过验收。
可直接复用已有标注文件，或用下面示例的内联 JSON。修改配置即可，不需要修改 Python。

## 按对象启用和衔接

`recipe.json` 中每个原生物体槽位必须出现一次；`slot=0` 对应 `mesh.obj/object_*`，
`slot=1` 对应 `mesh_1.obj/object_1_*`。不改的配对物体明确写
`{"slot":1,"preserve":"本轮固定配对物体"}`。允许整段全部明确 `preserve`，输出记为
`BASELINE_PRESERVED`；这提供基线交付路径，不计作新方法成功。

只需重新跟踪既有米制资产时，可设置 `tracking:{"anchor_frame":已审核帧号}`，
省略 shape、temporal 和 tracking.contact。仍需匹配场景、观测和输入审核，输出明确标为
`method="fixed metric FP"`、`new_optimization_executed=false`。
不要为了满足入口要求而给本来不需要或不适用的对象强加形变。

| 配置 | 实际执行及传递 | 需要的额外输入 |
|---|---|---|
| `shape.kind=part_shape` | 共享部件形变及辅助姿态 → 验收 → **新 mesh** 的 FP → 新轨迹 | 匹配旧 mesh 的两个部件顶点和中心顶点；支持/留出帧 |
| `shape.kind=contact_shape` | 16 节点形变及接触拟合 → 验收 → **新 mesh/新接触点** 的 FP | 世界系桌面平面、可见接触像素及其支持帧、有效深度支持 |
| `tracking.contact` | 匹配的 provided pose → 锚点单次校正 → 双向 FP；仅已观察静置帧使用支撑约束 | 同 mesh 接触点、静置帧、桌面平面；最大修正 ≤6cm |
| `temporal` | 当前匹配 mesh/完整轨迹 → 双目标记＋轮廓＋接触＋时序修正 → 完整场景 | 已验收的相机系米制标记及不确定性、标记分组、帧角色、接触点/平面 |

形变后强制重新跟踪，不把旧轨迹挂到新 mesh 上。普通 FP 路径重新 register；接触 FP 路径
使用形变输出中**同一 mesh、同一锚点**的辅助位姿。接触形变输出的局部接触点自动接入 FP 和
temporal；其他形变通过稳定顶点编号绑定接触点。原模型的任意局部点不能直接套到变形后模型。
所有尺寸已烘焙到米制 mesh，FP 不再搜索或乘第二次 scale。

配方示例：

- [最小整段时序修正](examples/temporal.recipe.json)
- [部件形变 → FP → 时序修正](examples/part_shape.recipe.json)
- [接触形变 → 支撑 FP → 时序修正](examples/contact_shape.recipe.json)

示例中的帧号/语义顶点仅表示格式，**不是 Scored 的预选标注**。
选择适用于实际结构的 shape 分支，不要给所有类别强套两个部件或桌面接触。
数值权重与迭代规则沿用 `5454d57`；不根据 GT 或 policy 成败改帧角色。

## 必要输入和自动适配

- **场景**：原生 `result.npz`，共同相机和所有对象完整有限的 SE(3)，
  `T_world_object = T_world_camera @ T_camera_object`；物体 scale=1，尺寸已烘焙。
  手部字段和真实 validity 原样保留。OBJ 的材质/纹理依赖必须齐备。
- **观测**：逐帧 RGB、该对象二值 mask、同相机像素网格上的 depth、固定 K。
  默认路径 `rgb/{frame:06d}.png`、`depth/{frame:06d}.png`、`masks/{frame:06d}.png`；
  可通过 `rgb/depth/mask` 字符串指定其他文件名格式。帧号须对应 bundle 的 `0..N-1`。
- **深度编码**：显式声明 `native_inverse_u16`（米=`65535/code-1`）、`millimetres_u16`
  或 `metres_npy`。入口自动转原生编码，记录量化与输入 SHA。0/65535 为原生无效值。
  validity 默认仅表示有效数值，**不冒充左右一致性置信度**；可提供 `validity` 文件名格式。
- **K/尺寸**：默认 `K_fp.json={fx,fy,cx,cy,width,height}`；也可指定 3×3 JSON/NPY，
  同时声明 `K_image_size:[width,height]`。`resize_to_depth:true` 才允许同相机 RGB/mask 缩放和
  半像素约定的 K 适配，深度不缩放。不自动猜测畸变、裁剪、相机桥接或时间同步。
- **桌面**：世界系平面 `[nx,ny,nz,d]`，单位米。refinement 输入可以是 NPY 路径或内联数组；
  内联数组自动归一化。tracking.contact 使用已归一化的 NPY。保持同一平面/坐标系。
- **语义/接触**：`semantic_vertices={parts:[i,j],center:k}`；
  `contact_annotation={frame:t,mask_boundary_pixel:[u,v],source:"输入证据"}`。
  必须观察对象真实接触；不能用任意最低顶点代替。`contact_point.vertex` 须是审核过的物体局部接触顶点；
  另一方式是 `{"file":"point.npy","mesh_sha256":"..."}`，严格绑定未形变的 mesh。
- **时序标记**：NPZ 或内联 JSON 包含 `frame_ids[T]`、`points_camera[T,L,3]`（米）、
  `qualified[T,L]`（bool）、`sigma_z[T,L]`（米）。对应 `measurement_receipt` 需是已有
  `OBSERVABILITY_PASS` 记录。必须来自校准、同步的输入测量；入口只适配格式，不生成虚假匹配或验收。
  calibration 是 fit 的子集，fit/validation 分开，每个标记至少两个合格 calibration 观测。
  `contact_frames` 是真实接触帧，`landmark_groups` 给每个标记一个整数分组。

桌面 `table*.npy/npz/json/ply/stl/obj` 和 `metadata.json` 自动带入；其他辅助文件在
`extra_files` 中按 bundle 相对路径声明。不会重新造桌面。目标物体材质按槽位隔离，避免覆盖配对物体的纹理。

## 输出、失败和验收

`PREFLIGHT.json` 列出 20 段实际执行图与缺项；每段 `EXECUTION.json` 保存阶段状态/耗时，
`episode_XXXXXX/bundle/` 包含完整原生场景和 `run.json`，其中记录各 mesh 与结果文件 SHA。
原始输入的 SHA 在运行前后复核。中间优化、FP 配对、叠加图和失败证据保留在各自阶段目录。

输出目录必须全新，禁止覆盖或按“目录非空”恢复。数值验收拒绝记为 `CANDIDATE_REJECTED`，
附具体失败指标；运行错误记为 `FAILED_PRESERVED`。两者都停止该段并保留证据，继续其他独立段；
不自动重试、缩小参数、调权重或回退旧包。`BATCH.json` 只有 20 段全部完成才写
`ALL_EPISODES_EXPORTED_UNVALIDATED`（退出0），缺项/拒绝或失败分别退出2/1。
批次分别列出候选导出、明确保留基线、数值拒绝和运行失败的数量；`verified_improvement_count=null`，
绝不把“9个候选＋11个基线”写成“20段新方法成功”。已经失败的实验不会自动换成基线；
基线保留必须是运行前配置中明确的选择，原失败记录不改写。

任何新候选（包括只完成shape＋FP、未运行temporal）均记为 `CANDIDATE_EXPORTED_UNVALIDATED`，
并标记 `input_gate_pending=true`、`full_sequence_quality_validation=NOT_RUN`、`production_adopted=false`。
少量支持/留出帧的形状检查以及锚点目视审核，不等于跟踪整段的验收。
本入口不声称已验证 Scored AUC、手物接触改善或 policy 收益，也不自动上传或提交比赛。

## 接通验证

```bash
python -m unittest discover -s baseline/reconstruction/scripts/scored20_pipeline/tests -v
```

测试覆盖完整 20 段清单的 CPU 预检、真正的 CPU 时序优化/导出、实际 FP 冻结器与 CPU 输入加载、
新 mesh/姿态/接触点的阶段传递、桌面及手/相机/配对物体保留、失败停止与缓存拒绝。
GPU 跟踪仅在测试中使用显式替身验证接线；本次发布未执行 GPU 或完整 Scored20，
不把接口测试当成实际 GPU 数值或整批质量验证。

原接通发布验证：20项集成测试、原优化器11项、FP15项、R7/R8 13项均通过。
已有 ep21 双对象186帧缓存通过完整重新封装检查：目标 mesh 顶点/面及位姿一致，
31个受保护字段不变，源文件 SHA 不变；3帧真实 RGB/depth/mask 适配前后数值一致。
另检查了已有 Scored ep19 的235帧原生输入结构。以上没有重新跟踪、优化真实序列或评分。

## 本次整批失败后的发布约束

- 逐对象说明采用普通跟踪、哪项优化或保留基线，不能凭参数表能填完就认定通用适用。
- 分别公布输入检查、实际运行覆盖、最终产物覆盖、全段质量对照；“所有段已尝试”不等于完整交付。
- 全段对照包含RGB/depth对齐、运动连续性和手—物/物—物关系。手和相机数组未改，只说明未重估这些字段；
  物体改变后相对关系仍会改变，不能以字段一致替代交互验收。
- 新模块先形成有代表性的完整轨迹证据，再扩大适用范围；未运行验证的范围必须明确写出。
- 保留负结果，不根据本批失败放宽门槛。不把代理指标的通过、完成导出或CPU测试通过当成官方提分。
- 交付说明必须同时报告哪些对象未改、哪些模块未运行、哪些段未通过及仍需补齐的验证。

本次20段中：ep1/3/6/14/15/19/32/35/38导出；ep5/8/10/16/20/25/28/30/33/36/37被拒绝。
3段更新边界/提议几何检查未通过，其余8段仅未通过留出指标。验收项的失败计数有重叠：深度median6、p90 6、
IoU3、覆盖3、边界3、较大IoU退步2。20段接触距离和最终几何检查均通过，所以不能归因为
“桌面约束没执行”。例如ep10留出深度median从4.58mm变为15.15mm，确有退步；ep8的
median仅从3.59mm变为3.78mm，也说明不同失败的程度不同，不应统一归因或统一放宽阈值。

发布修正增加了拒绝原因传播、显式基线/普通FP分流及状态语义的回归测试，26项入口测试通过。
修正没有改变优化损失、数值门槛、FP或任何既有预测，不能使原先11个被拒候选自动变好。
原批次记录保留；已有9个导出包须完成整段质量对照后才能讨论采用。
