# Scored20：输入约束优化 → FoundationPose → 完整场景

这是 `5454d57` 三个优化器和带桌面接触约束 FP 的统一入口。显式按对象启用模块，
运行期间自动完成数据适配、冻结、优化、匹配新模型的跟踪、时序修正和完整 bundle 导出。
没有启用新增优化的 episode 会报错，不会把旧默认流程标记成新版成功。

输入是**已有前端输出和匹配的场景 bundle**。本入口保留相机、手、桌面和未指定修改的物体；
不重新生成深度、mask、手或相机，也不训练 policy。普通重建入口的默认行为不变。

## 一条完整运行命令

在仓库根目录、CPU 优化依赖环境中运行：

```bash
python baseline/reconstruction/scripts/scored20_pipeline/pipeline.py run \
  --config scored20.json --interactive-reviews
```

固定范围：`1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38`。
先对全部输入做 CPU 检查；有缺项则不启动任何优化/GPU，并列出所有 episode 的问题。
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

骨架中的 `accepted=false`、接触点和帧角色留空，必须按输入证据填写；不会伪造标注或自动通过验收。
可直接复用已有标注文件，或用下面示例的内联 JSON。修改配置即可，不需要修改 Python。

## 按对象启用和衔接

`recipe.json` 中每个原生物体槽位必须出现一次；`slot=0` 对应 `mesh.obj/object_*`，
`slot=1` 对应 `mesh_1.obj/object_1_*`。不改的配对物体明确写
`{"slot":1,"preserve":"本轮固定配对物体"}`。每段至少启用一项新增优化。

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

输出目录必须全新，禁止覆盖或按“目录非空”恢复。单段失败后记录失败，继续其他独立段；
不自动重试、缩小参数、调权重或回退旧包。`BATCH.json` 只有 20 段全部完成才写
`COMPLETE_SCORED20_EXPERIMENTAL`（退出0），缺项/失败分别退出2/1。

时序优化完成表示**有界候选已导出**，仍标记 `input_gate_pending=true`、`production_adopted=false`。
本入口不声称已验证 Scored AUC、手物接触改善或 policy 收益，也不自动上传或提交比赛。

## 接通验证

```bash
python -m unittest discover -s baseline/reconstruction/scripts/scored20_pipeline/tests -v
```

测试覆盖完整 20 段清单的 CPU 预检、真正的 CPU 时序优化/导出、实际 FP 冻结器与 CPU 输入加载、
新 mesh/姿态/接触点的阶段传递、桌面及手/相机/配对物体保留、失败停止与缓存拒绝。
GPU 跟踪仅在测试中使用显式替身验证接线；本次发布未执行 GPU 或完整 Scored20，
不把接口测试当成实际 GPU 数值或整批质量验证。

本次发布验证：新增20项集成测试、原优化器11项、FP15项、R7/R8 13项均通过。
已有 ep21 双对象186帧缓存通过完整重新封装检查：目标 mesh 顶点/面及位姿一致，
31个受保护字段不变，源文件 SHA 不变；3帧真实 RGB/depth/mask 适配前后数值一致。
另检查了已有 Scored ep19 的235帧原生输入结构。以上没有重新跟踪、优化真实序列或评分。
