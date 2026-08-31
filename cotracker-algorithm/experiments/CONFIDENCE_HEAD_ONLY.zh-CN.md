# 原始 CoTracker3 基线的置信度单点调优

## 为什么要重做

此前的 `confidence_hard_12` 与 `confidence_soft_6_18` 会通过共享网络参数间接改变坐标预测，因此它们不是“只调整置信度”的单点实验。本实验重新从未经 TrackRAD 微调的 `scaled_offline.pth` 出发，把可训练范围严格限制为 `updateformer.vis_conf_head` 的置信度输出行。

冻结项包括特征提取器、相关性模块、Transformer、坐标输出、可见性输出以及其他所有参数。训练损失只包含置信度 BCE，不包含坐标、不可见点坐标或可见性损失。优化器权重衰减固定为 0，避免被屏蔽的可见性行因 AdamW 权重衰减发生变化。

## 三组对照

| Profile | 是否训练 | 唯一设置 |
|---|---:|---|
| `original_baseline` | 否 | 原始 `scaled_offline.pth`，作为共同参照 |
| `confidence_head_hard_12` | 是 | 仅训练置信度行；距离不超过 12 像素为 1，否则为 0 |
| `confidence_head_soft_6_18` | 是 | 仅训练置信度行；6 像素内为 1，6–18 像素线性下降，18 像素外为 0 |

硬标签与软标签组共享同一初始权重、数据、批次顺序、训练种子、教师选择序列、优化器、学习率和训练步数。两组之间唯一变量是置信度目标标签。

需要注意：原始 TrackRAD 基线没有设置置信度阈值，但置信度会作为下一轮 CoTracker 迭代的输入。因此，单独调整置信度行仍可能在第 2–4 次迭代中间接影响最终坐标；这正是本实验允许的唯一作用通路。

## 断电可恢复训练

远程电脑拉取最新提交后执行：

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\run_confidence_head_only_experiments_resumable.ps1" `
  -RepoRoot $Repo `
  -DatasetDir "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data" `
  -ResultsRoot "$Repo\confidence-head-only-training-results" `
  -NumSteps 1000 `
  -SaveEverySteps 25
```

每 25 步原子保存一次。断电后重复同一命令会从最近检查点恢复，已经完成的组会跳过。脚本结束前执行两类审计：

1. 硬标签与软标签组的逐步教师选择序列必须完全一致；
2. 最终权重中只能有 `updateformer.vis_conf_head` 第 2 行（置信度行）发生变化。

审计文件位于训练结果目录中的 `confidence-label-audit.json`，以及各训练组内的 `confidence-head-parameter-audit.json`。

## 统一评估与汇总

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\evaluate_confidence_head_only_experiments.ps1" `
  -RepoRoot $Repo

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\summarize_confidence_head_only_experiments.ps1"
```

最终表格：

- `confidence-head-only-evaluation-results\confidence-head-only-results.csv`
- `confidence-head-only-evaluation-results\confidence-head-only-deltas.csv`

DSC、D98 越大越好；HD95、MASD、CD 越小越好。汇总同时给出硬标签相对原始基线、软标签相对原始基线，以及软标签相对硬标签的差值。
