# CoTracker3 置信度软标签实验

## 实验目的

检验将原始的 12 像素硬标签替换为分段线性软标签，是否能够改善 CoTracker3 在 TrackRAD 数据上的置信度学习与最终分割指标。

三组实验共享相同初始权重、训练数据、批次顺序、随机种子、教师模型选择序列、优化器、学习率、训练步数及损失权重。唯一变量是置信度目标标签。

## 三组设置

| Profile | 标签定义 |
|---|---|
| `confidence_hard_12` | 距离不超过 12 像素为 1，否则为 0 |
| `confidence_soft_8_16` | 8 像素内为 1，8 到 16 像素线性下降，16 像素外为 0 |
| `confidence_soft_6_18` | 6 像素内为 1，6 到 18 像素线性下降，18 像素外为 0 |

两组软标签在 12 像素处均等于 0.5，因此保留了原始 12 像素决策边界的解释。

## 训练说明

远程电脑已有 TrackRAD 数据，因此实验使用现有的单教师真实视频微调流程。三组都启用同样的置信度 BCE，教师轨迹作为监督目标，不启用辅助教师。默认训练 1000 步，每 25 步原子保存一次，断电后重复原命令即可恢复。

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\run_confidence_label_experiments_resumable.ps1" `
  -RepoRoot $Repo `
  -DatasetDir "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data" `
  -ResultsRoot "$Repo\confidence-label-training-results" `
  -NumSteps 1000 `
  -SaveEverySteps 25
```

完成后脚本会生成 `confidence-label-audit.json`，验证三组使用了完全一致的教师序列。

## 统一评估与汇总

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\evaluate_confidence_label_experiments.ps1"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\summarize_confidence_label_experiments.ps1"
```

结果位于：

- `confidence-label-evaluation-results\confidence-label-results.csv`
- `confidence-label-evaluation-results\confidence-label-deltas-vs-hard.csv`

DSC、D98 越大越好；HD95、MASD、CD 越小越好。
