# 轮廓点数增益诊断（只读）

这一步不训练、不运行推理、不改变既有结果。它回答：250 点相对固定 1000 点的测试集收益是否广泛出现，还是由少数病例推动；DSC 增益与 D98 退化是否发生在相同病例。验证集同时列出既有 250、500、750、1000、1250 点结果，但**不会据已看过的38例重新选择参数**。

在远程电脑运行：

```powershell
git -C C:\zhongliu\zhongliu-tuning pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\diagnose_retune_points.ps1
```

默认输出到 `protocol-40-10-38\parameter-retune-gridfix-v2\point-count-diagnostics`：

- `test-38-paired-cases.csv`：38例逐病例五项正式指标及250点减1000点的差值。
- `test-38-paired-summary.csv`：总体和A/B/C/X队列的均值、方向计数、留一病例范围及探索性病例自助抽样区间。
- `validation-10-point-counts.csv`：首轮已跑完的五种点数在10例上的逐病例值。
- `summary.json`：输入指标文件哈希及DSC/D98方向冲突计数。

测试集已经被反复查看，区间不代表新的独立盲测证据；同一患者或时间帧也不宜简单视为独立样本。该诊断仍无法单独区分轨迹变化与轮廓重建变化。如果250点收益仅由极少数病例驱动，下一轮优先检查这些病例的轮廓质量；如果多病例一致，再在验证集做更窄的点数扫描。
