# 使用软标签重新学习 CoTracker3 置信度表征

本实验从原始 `scaled_offline.pth` 出发，使用 `soft_6_18` 标签和置信度 BCE，重新训练 UpdateFormer 的共享表征及置信度输出行。坐标损失、不可见点坐标损失和可见性损失全部关闭。

冻结范围包括图像编码器、相关性模块、坐标输出 `flow_head` 和 `vis_conf_head` 的可见性行。需要注意，UpdateFormer 是坐标、可见性和置信度共享的，因此共享表征改变后，冻结的坐标头也可能输出不同坐标。本实验测试的是“更深层置信度重学”的整体效果，不再属于只有 385 个参数变化的严格输出行单点实验。

对照组为：

- `original_baseline`：未经 TrackRAD 训练的原始权重；
- `confidence_head_soft_6_18`：上一轮仅训练置信度输出行的结果；
- `confidence_updateformer_soft_6_18`：本轮重新训练共享 UpdateFormer 表征。

训练、评估和汇总命令：

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\run_confidence_updateformer_soft_resumable.ps1" `
  -RepoRoot $Repo `
  -NumSteps 1000 `
  -SaveEverySteps 25

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\evaluate_confidence_updateformer_soft.ps1" `
  -RepoRoot $Repo

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\summarize_confidence_updateformer_soft.ps1"
```

训练每 25 步原子保存，断电后重复同一命令即可恢复。结束时参数审计必须证明图像编码器、相关性模块、坐标输出头和可见性行没有变化。
