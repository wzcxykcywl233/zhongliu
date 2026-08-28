# 每批次随机“课外辅导教师”实验

## 1. 实验定义

本实验修改的是 CoTracker3 的伪标签微调过程，而不是 TrackRAD 推理参数。原始
`train_on_real_data.py` 在每个 batch 从教师池随机选取一名主教师。本实验保留
该过程，再从主教师之外随机抽取一名低权重辅导教师。

设主教师损失为 `L_main`，辅导教师损失为 `L_tutor`，辅导权重为 `alpha`：

```text
L_teacher = (1 - alpha) * L_main + alpha * L_tutor
```

总教师损失强度保持为 1，因此不同组之间不会混入“增加一个损失项导致总损失
变大”的干扰。`alpha=0` 即单教师基线。两名教师使用相同的视频和查询点，
教师均处于 `eval`/`no_grad` 状态，只有学生模型更新参数。

默认教师池为：

- CoTracker2.1 online；
- CoTracker3 baseline online；
- CoTracker3 baseline offline。

上游脚本还列出了 TAPIR，但当前随仓库下载的源码不包含 TapNet 子模块及其权重，
因此默认可复现实验池不启用 TAPIR。六组实验严格共用上述三名教师，比较仍然
受控；若以后补齐 TapNet，可通过 `--teacher_types` 显式加入，但必须把整套实验
全部重跑，不能与三教师结果直接混合。

主教师每 batch 均匀随机选择；`alpha>0` 时辅导教师从剩余两名中均匀随机
选择。这里没有显式教师共识、投票或参数平均。

## 2. 实验矩阵

| Profile | 主教师权重 | 辅导教师权重 | 说明 |
|---|---:|---:|---|
| `baseline_single_teacher` | 1.000 | 0 | 原始单教师目标 |
| `random_tutor_w0025` | 0.975 | 0.025 | 极弱修正 |
| `random_tutor_w005` | 0.950 | 0.050 | 弱修正 |
| `random_tutor_w010` | 0.900 | 0.100 | 中等修正 |
| `random_tutor_w020` | 0.800 | 0.200 | 较强修正 |
| `same_teacher_control_w010` | 0.900 | 0.100 | 辅导教师强制等于主教师的实现校验 |

最后一组理论目标与单教师基线相同；若指标出现超过正常重复误差的差异，应先
排查随机性和实现，不解释为模型改进。

## 3. 远程电脑运行

在更新仓库后，先下载学生初始权重和三名教师权重：

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\prepare_random_tutor_checkpoints.ps1" `
  -RepoRoot $Repo
```

随后顺序运行六组训练：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\run_random_tutor_training_resumable.ps1" `
  -RepoRoot $Repo `
  -DatasetDir "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data" `
  -ResultsRoot "C:\zhongliu\zhongliu-tuning\random-tutor-training-results" `
  -NumSteps 2000 `
  -SaveEverySteps 25
```

脚本前台实时输出并同步追加到各组的 `training.log`。六组按表格顺序串行运行，
不会同时占用显存。

## 4. 断电恢复

每 25 个 optimizer step 原子提交一次 `.pth`，内容包括：

- 学生模型；
- optimizer 和学习率 scheduler；
- 当前 epoch、下一个 batch 和累计 step；
- Python、NumPy、PyTorch/CUDA 随机状态；
- 独立教师采样器状态。

断电后重新运行完全相同的 PowerShell 命令即可。已完成 profile 会跳过，未完成
profile 自动读取编号最大的检查点。为限制磁盘占用，每组自动保留最新三个完整
断点；不要手动删除当前 profile 目录中的 `.pth`。

## 5. 使用统一 TrackRAD 指标评估

所有训练完成后执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\evaluate_random_tutor_models.ps1" `
  -RepoRoot $Repo
```

评估使用推理 profile `baseline`，唯一变化是学生权重。每组结果位于
`C:\zhongliu\zhongliu-tuning\random-tutor-evaluation-results\<profile>\evaluation\metrics.json`，
指标为 DSC（越大越好）、HD95/MASD/CD（越小越好）和 Relative D98（越大
越好）。评估同样按病例断点恢复。

## 6. 审计要求

- 每组必须使用相同的学生初始权重、教师池、数据顺序、训练步数和随机种子；
- 比较前确认 `meta.json` 中只有 `auxiliary_teacher_weight` 或控制开关不同；
- 保留 `training.log`、自动保留的三个中间断点、最终权重和评估 `metrics.json`；
- 建议最终候选权重再用三个训练种子复跑，以区分改进与训练随机波动。
