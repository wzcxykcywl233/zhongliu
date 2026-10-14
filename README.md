# CoTracker3 for Tumor Tracking 

### Teacher-only single-factor experiments (no Mamba)

`diagnostics/teacher-single-factor-v1/run_teacher_experiments.ps1` compares existing
offline teacher weights and earlier inference mechanisms in 32 one-factor profiles.
Start with `-Stage plan`, `-Stage smoke`, then `-Stage validation -Group weights`;
other groups are explicit and reuse verified parent controls. No teacher/student
training, image build, pull or network is performed. A frozen validation shortlist
and explicit `-Stage test` produce a separate 38-case **teacher-generator** report,
not student performance. First-frame boundary queries only; arbitrary-time training
adaptation is deliberately not assumed. See [design and commands](diagnostics/teacher-single-factor-v1/README.txt).

If the evaluator fails to write `/output/metrics.json` with `PermissionError`, use
the separate [evaluation permission repair](diagnostics/teacher-eval-permission-fix-v1/README.txt)
entry `diagnostics/teacher-eval-permission-fix-v1/run_repair.ps1 -Stage validation -Group weights`.
It preserves the original frozen package, manifest and completed predictions.
The evaluator keeps its original image user and dependencies and computes into
private scratch output; a separate directory-owner writer publishes byte-identical
metrics with SHA256 checks. This also fixes `psutil` lookup after UID changes.
No chmod/chown, dependency installation, image build or retraining.

### Dynamic anchors and frozen-Transformer feature sidecars

Two independent 40/10/38 studies are ready in `scripts/run_anchor_and_sidecar_queues.ps1`.
Use `-Stage smoke`, then `-Stage validation`, then explicitly `-Stage test`.
The anchor study tests per-point reliability drops and rollback; the sidecar study
compares MLP, temporal convolution, short-reset and long Mamba-style reference SSMs
while freezing all pretrained CoTracker tensors. Local Docker runtimes are reused
without downloading dependencies. See [design, controls and commands](cotracker-algorithm/experiments/ANCHOR_AND_FEATURE_SIDECAR.zh-CN.md).
Only test-38 official metrics belong in the final performance table.

### Sidecar speed, degradation controls and diagnosis

The separate `scripts/run_sidecar_speed_safety_resumable.ps1` study starts with
`-Stage profile`, not another 1000-step training run. It benchmarks an equivalent
parallel scan and backward on the remote GPU, then gates a real 5-step smoke.
Validation reuses completed weights to test residual gains and a coordinate
correction cap; test-38 requires a frozen validation-qualified selection.
Old training/results remain read-only. See [scope, guards and commands](cotracker-algorithm/experiments/SIDECAR_SPEED_SAFETY.zh-CN.md).

### Six independent query-memory refinements

Run `scripts/run_memory_refinement_resumable.ps1` for the diverse-memory control
and six fixed inference-only ablations, on validation-10 then test-38.
The queue freezes source/weight/data hashes and resumes completed cases after interruption.
See [the protocol and commands](cotracker-algorithm/experiments/MEMORY_REFINEMENT.zh-CN.md).

Run from the root of this repository:
```
git submodule init
git submodule update

export TORCH_HOME=${PWD}/cotracker-algorithm/torch
cd cotracker-algorithm
pixi run python download_model.py
```

## Audit the TrackRAD2025 dataset

The PowerShell audit is read-only with respect to the dataset. It checks case
structure, metadata JSON files, MHA headers, frame counts, and image/label
dimensions, then writes CSV and JSON summaries:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\audit_trackrad_dataset.ps1 `
  -DatasetDir "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data" `
  -OutputDir ".\dataset-audit"
```

The generated files are `dataset-summary.json`, `dataset-cases.csv`,
`dataset-mha-files.csv`, and `dataset-issues.csv`.

## Run real-mask-assisted CoTracker training

The ground-truth-mask study keeps the original one-random-teacher-per-batch
training path and adds only a differentiable target-membership loss from the
40-case training masks. It evaluates all weights on validation-10, freezes the
selected weight, and evaluates only that candidate plus its paired control on
test-38. Its second protocol version derives clip and query randomness from the
epoch, sample, and global step, so training and evaluation resume without
changing the paired sequence after interruption.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_ground_truth_mask_supervision_resumable.ps1
```

The design and result locations are documented in
`cotracker-algorithm/experiments/GROUND_TRUTH_MASK_SUPERVISION.zh-CN.md`.

### Download and evaluate the 38-case public test set

The public test release consists of the 8-case preliminary set and the 30-case
final set with privacy-restricted cohort D excluded. On the remote Windows
machine, first download and audit both subsets. The script resumes interrupted
downloads and creates a combined 38-case directory using NTFS hard links, so it
does not duplicate the MHA payloads:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\download_trackrad_public_test_resumable.ps1
```

Then evaluate the original baseline and the eleven pre-registered directions
selected from the completed 50-case ablation table. The frozen queue is stored
in `cotracker-algorithm/experiments/public-test-38-profiles.json`. Per-case
outputs and per-profile metrics are checkpointed, so the same command safely
resumes after interruption:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_public_test_38_resumable.ps1
```

The queue may also be started before the download finishes. It waits for the
combined 38-case audit to pass and then launches the same resumable evaluation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\queue_public_test_38_when_ready.ps1
```

Results are written to `public-test-38-results`, including
`public-test-38-results.csv` and
`public-test-38-deltas-vs-baseline.csv`. This public test set must not be used
for further parameter selection or training.

### Establish the 40/10/38 protocol

Create a deterministic patient-level 40/10 split from the 50 labeled training
cases while keeping the 38 public test cases separate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_trackrad_40_10_38.ps1
```

The split preserves fixed A/B/C validation quotas (5/3/2) and approximately
stratifies anatomical regions within each cohort. It uses hard links and writes
the complete assignment to `trackrad-40-10-split.json`.

The existing non-trainable inference profiles do not consume the 40-case
training split. Re-evaluate their fixed queue on the 10-case validation split
with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_validation_10_resumable.ps1
```

Future trainable methods such as a Mamba adapter must train only on the 40-case
split, select settings only on the 10-case validation split, and use the
38-case set only for a final locked evaluation.

### Run the conservative and aggressive Mamba experiments

The dedicated queue trains two exploratory Mamba-style variants from the same
frozen CoTracker checkpoint: a conservative trajectory residual adapter and an
aggressive replacement of UpdateFormer temporal-attention blocks. Both use one
fixed offline teacher, hard labels, no auxiliary teacher, and the 40/10/38
protocol. Training is checkpointed every 25 steps and evaluation is committed
per case:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_mamba_40_10_38_resumable.ps1 `
  -NumSteps 1000 `
  -SaveEverySteps 25
```

Rerun the same command after an interruption. The 38-case table and direct
deltas against `hierarchical_full_grid0` are written below
`protocol-40-10-38\mamba-experiments\test-38`. The frozen design and reporting
rules are documented in
`cotracker-algorithm/experiments/MAMBA_EXPERIMENTS.zh-CN.md`.

The second Mamba study keeps CoTracker intact and learns only a full-case
hierarchical/global fusion gate from the 40-case ground-truth masks. It includes
an Oracle upper-bound audit and a per-frame MLP control so that any improvement
can be attributed to temporal modeling:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_long_fusion_mamba_resumable.ps1 `
  -TrainingSteps 500 `
  -SaveEverySteps 25
```

See `cotracker-algorithm/experiments/LONG_FUSION_MAMBA.zh-CN.md` for the frozen
design and output locations.

### Run the parameter-free dynamic query-memory study

This study keeps the CoTracker backbone and the established
`hierarchical_full_grid0_iterations2` configuration fixed. It permanently
retains the first query anchor and compares a latest reliable anchor, a
reliability-ranked Top-K memory, and a reliability-plus-diversity Top-K memory.
No model training, teacher model, or soft labels are used.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_dynamic_query_memory_resumable.ps1
```

The queue audits the 40/10/38 split, evaluates the fixed profiles on validation
10 and test 38, records memory admission/fusion diagnostics, and resumes at the
per-case boundary after interruption. See
`cotracker-algorithm/experiments/DYNAMIC_QUERY_MEMORY.zh-CN.md` for the frozen
design.

### Run mask-level appearance validation

This parameter-free follow-up keeps the validation-selected Top-K query memory
and uses frozen CoTracker frame features to validate each predicted mask against
the first-frame target appearance. It compares conservative four-pixel and
eight-pixel whole-mask translation searches while preserving mask shape and
area:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_mask_appearance_validation_resumable.ps1
```

The 10-case validation and 38-case test runs are checkpointed per case. Design
details and required activation diagnostics are in
`cotracker-algorithm/experiments/MASK_APPEARANCE_VALIDATION.zh-CN.md`.

### Run the supplementary 38-case queue

The remaining registered inference profiles and the controlled auxiliary-
teacher profiles have separate frozen manifests. Soft confidence labels are
disabled. Teacher models train on the 40-case split and are evaluated, but
never trained, on the 38-case public test set. Because the public-test results
have already been inspected, these runs are supplementary comparisons rather
than a new blind model-selection test.

The following command runs or resumes the complete queue. It retains per-step
training checkpoints, per-case prediction checkpoints, and append-only logs:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_remaining_public_test_38_resumable.ps1
```

Use `-SkipInference`, `-SkipTeacherTraining`, or `-SkipTeacherEvaluation` to
resume only a selected stage. The inference results are merged into
`public-test-38-results`; hard-label teacher training and evaluation are stored
under `protocol-40-10-38`.

### Diagnose memory features, trajectories, and masks

Run `scripts/run_memory_chain_diagnostics.ps1` for an observation-only paired
38-case diagnostic queue (control, repeated control, pointwise fusion, current
retrieval). It records sampled pyramid features, full trajectories and masks,
checks repeatability and prior outputs, and resumes per case/profile using
hashed checkpoints. It does not train or access later-frame ground-truth masks.
See [the diagnostic design and remote commands](cotracker-algorithm/experiments/MEMORY_CHAIN_DIAGNOSTICS.zh-CN.md).

### Inherit visibility/confidence across dynamic query anchors

`scripts/run_query_state_inheritance_resumable.ps1` runs six frozen inference
profiles: fixed memory control, V-only, C-only, V+C, time-decayed V+C, and
query-frame-only V+C. It uses original weights, no training, the existing
40/10/38 protocol, case-level resume, frozen inputs and mechanism audits.
Use the `test-38` CSV for performance reporting, not the validation CSV.
See [design, risks and remote commands](cotracker-algorithm/experiments/QUERY_STATE_INHERITANCE.zh-CN.md).

### Per-frame historical-offset backcheck and iteration sweep

`scripts/run_frame_backcheck_resumable.ps1` runs 2/4/6 iterations with scoring
off/on, keeping the diverse memory baseline and original weights. Each target
frame compares center, history-offset and equal-count fixed-neighborhood
evidence without changing predictions. The queue checks output equality,
resumes per case, evaluates validation 10 and test 38 separately, then runs
label-aware offline score analysis. Copy performance from
`test-38/iteration-performance-test-38.csv` (scoring-off groups).
See [the frozen experiment design](cotracker-algorithm/experiments/FRAME_BACKCHECK.zh-CN.md).
# Four-way per-frame backcheck (observation only)

Fixed two-iteration diverse-memory control; compare center, single historical
offset, four mirrored historical offsets, and fixed four-way offsets on identical
supported points. Two inference profiles compute four scores without position,
confidence, or memory feedback. Run `scripts/run_fourway_backcheck_resumable.ps1`;
results are isolated under `protocol-40-10-38/fourway-backcheck`.
See [四向回查实验说明](cotracker-algorithm/experiments/FOURWAY_BACKCHECK.zh-CN.md).
# History-guided rotation alignment (diagnostic only)

Run `scripts/run_rotation_backcheck_resumable.ps1` for fixed two-iteration
control versus history-guided angle estimation and raw-patch re-encoding.
No four-way expansion and no position/state feedback. Validation-10 and test-38
outputs stay separate under `protocol-40-10-38/rotation-backcheck`.
See [旋转对齐诊断说明](cotracker-algorithm/experiments/ROTATION_BACKCHECK.zh-CN.md).

# Best-mainline and VC-decay candidate diagnostics

Run `scripts/diagnose_best_candidates.ps1` to analyze the existing
`query-state-inheritance/test-38` outputs without training or rerunning inference.
It compares official case metrics, time/segment-offset error patterns,
ground-truth centroid-alignment counterfactuals, contour reconstruction losses,
and worst-frame overlays. CPU-only, offline, frozen-runtime, case-resumable;
source predictions are mounted read-only. Ground-truth probes are diagnostics,
not deployable improvements or official performance numbers.
See [主线与候选诊断设计及运行说明](cotracker-algorithm/experiments/BEST_CANDIDATE_DIAGNOSTICS.zh-CN.md).
# 当前最优方法参数复调（40 / 10 / 38）

入口：`scripts/run_parameter_retune_resumable.ps1`。默认仅运行验证搜索：63组首轮、最多20组交互组合、冻结入围复核；之后显式 `-Stage test` 才启动38例正式评估。断电重跑同一命令续接，不进行训练。

250点收益的后续[跟踪/重建拆分实验](cotracker-algorithm/experiments/POINT_DECOUPLING.zh-CN.md)入口为 `scripts/run_point_decoupling_resumable.ps1`：固定五组，包含稠密轨迹缓存重建、相同查询位置的稀疏跟踪及原生250点采样对照。默认 `-Stage validation`，审计通过后显式 `-Stage test` 运行38例，输出独立保存在 `protocol-40-10-38/point-decoupling-v1`。

[CoTracker主干加深实验](cotracker-algorithm/experiments/BACKBONE_GROWTH.zh-CN.md)入口为 `scripts/run_backbone_growth_resumable.ps1`，包含同条件微调base（25.39M）、时间6层（30.70M）、时间/空间各6组（46.67M）及未微调参照。新增残差层零输出初始化并保留预训练三层顺序；默认三种子、每组1000步，结构化审计教师/病例/查询/视频配对。依次运行 `-Stage smoke`、`-Stage validation`、`-Stage test`，最终38例结果在 `protocol-40-10-38/backbone-growth-v1/test-38`。

详细范围、对照、筛选规则和输出位置见 [参数复调协议](cotracker-algorithm/experiments/PARAMETER_RETUNE.zh-CN.md)。

## 支线训练耗时与退化独立诊断

[运行说明](cotracker-algorithm/experiments/SIDECAR_RUNTIME_DIAGNOSTICS.zh-CN.md)：`scripts/run_sidecar_runtime_diagnostics.ps1` 默认 `-Stage training`，只做 MLP/长 Mamba 各 3 步真实分阶段计时，保留原千步学习率计划。`-Stage profile` 为 11 组独立进程、每组 180 秒预算的合成微基准；`-Stage validation` 不再受加速门槛阻塞，固定原后端复用旧权重诊断 gain/修正上限。未启动新的千步训练；候选合格后显式 `-Stage test` 报告 38 例结果，旧目录保持不变。

真实计时定位到 Mamba 反向显存超额压力后，新增[低显存扫描调整](cotracker-algorithm/experiments/SIDECAR_COMPACT_SCAN.zh-CN.md)：`scripts/repair_sidecar_scan_memory.ps1 -Stage proof` 先验证稀疏状态/解析反向，GPU 输出、梯度、P/V/C 和显存通过后显式 `-Stage training` 仅做 3 步计时。使用新目录 `sidecar-compact-scan-v4`，不重跑旧全尺寸 reference 高显存训练图，不启动千步训练。

### 支线100步稳定性与独立退化诊断

新入口 `scripts/run_sidecar_stability_and_degradation.ps1`，新目录 `protocol-40-10-38/sidecar-stability-v5`，不要与 v4 混用。按阶段显式运行：

- `-Stage proof`：重新冻结当前源码/镜像/数据/权重，在新进程中做三组 compact GPU 正确性和显存证明。无需下载依赖。
- `-Stage stability`：MLP、长 Mamba 分别连续100步，再另起目录做50步保存、退出进程、恢复到100步，共400次真实更新。学习率仍按原1000步计划，60帧/128点/4次迭代、硬标签、无辅助教师/掩膜损失不变。主干冻结审计与教师/病例/查询/视频配对必须通过。原训练文件不修改；计时和早停在内存中插入。
- 每个进程预算1800秒；每个完成更新后检查已分配/预留峰值不超过20 GiB、驱动剩余至少2 GiB，保存后遇不合格即停止。不能预防单步中的瞬间OOM。第10–29与80–99步中位耗时比不超过2、常驻分配增长不超过256 MiB、峰值分配增长不超过1 GiB；越界要求复核，不直接断言泄漏。记录有限损失/梯度、裁剪前梯度范数和支线残差RMS。
- 恢复检查：输入严格一致；模型/AdamW浮点状态 `rtol=1e-5, atol=1e-7`，整数、调度器、教师采样器、RNG等状态严格一致；另外报告是否全部逐值一致。不以序列化文件哈希差异判断数值等价。
- `-Stage validation`：独立于稳定性结果，复用旧1000步权重、保持reference后端，只在10例验证集运行未训练参照，以及MLP/Mamba各三种子的原强度、0.5、0.25、0.5且修正上限0.5输入像素，共25个评估组。输出逐病例/队列的残差、轨迹位移、掩膜变化、DSC/D98等诊断。不会使用100步诊断权重。
- `-Stage test`：仅验证集平均DSC不低于参照且D98下降不超过0.001的候选可进入38例报告，设置在种子间一致并冻结；需要显式执行，不因速度通过自动启动。测试集已反复观察，不宣称盲测。

关键结果：`runtime/stability-summary.json`、`runtime/stability-summary.csv`、`degradation/validation-10/sidecar-safety-summary.csv`、`degradation/validation-10/sidecar-safety-case-diagnostics.csv`、`degradation/selection.json`。这些阶段没有新的1000步训练入口。部分进程失败时保留证据，换新的ResultsRoot重新验证，禁止把残缺探针拼成完成结果。

若 v5 退化验证在 `mamba_gain050_cap05/A_010` 报 `did not produce a valid output.mha`，而WSL已确认输出存在：使用 `scripts/repair_sidecar_output_paths.ps1 -Stage validation`，不要重跑训练或删除旧结果。恢复入口保留原 `degradation/frozen-study.json`，仅接受已审阅的宿主路径修复，写入独立 `path-repair-provenance.json`；模型、权重、数据、配置、旧镜像及筛选规则必须不变，不重新构建镜像。

新临时目录改为较短的 `checkpoint/.a/<case>-<6位ID>`，输出路径达到256字符时提前拒绝。旧 `.attempts` 保留：在同一个冻结的推理镜像里进行CPU检查，核对完整写入日志、实验配置/支线运行配置、MHA形状和二值内容、诊断数组哈希及轨迹缓存，校验通过后逐字节复制到短目录，沿用原日志并完成病例提交，不重复推理该病例。原临时文件不会删除。旧日志没有独立执行收据，复用还依赖原研究的冻结源码/数据/权重/镜像守卫；恢复收据记录这一限制。该例起止时间采用原runner的Running/FAILED事件边界，匹配日志修改时间并保存原始日志快照；`TimeSec`标记为恢复估计，不能作为精确端到端速度证据，原日志中的算法耗时保留。多个旧输出、时间边界歧义、缓存缺失、配置或内容不符均停止，不能随意选择旧尝试。
