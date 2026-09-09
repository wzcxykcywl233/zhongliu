# Mamba 探索实验：40/10/38 固定协议

## 实验目的

本实验从当前已验证的 `hierarchical_full_grid0` 出发，只比较 Mamba 的两种
接入方式。两组均保留跨度 10、原始特征权重 0.5、遮挡层级合并、双锚点权重
0.5、关闭支撑网格和 4 次迭代，不混入软标签或辅助教师重采样。

这里的模块是按照 Mamba 选择性状态空间思想实现的纯 PyTorch 双向时序扫描，
用于先验证结构是否有效。它不依赖 `mamba-ssm` 的专用 CUDA kernel，避免远程
电脑因 PyPI/GitHub 网络或本地编译失败而无法复现实验。因此报告中应称其为
“Mamba-style 选择性状态空间模块”，不能声称使用了官方优化 kernel。

## 两条实验路线

| Profile | 改动位置 | 冻结部分 | 训练部分 | 预期与风险 |
|---|---|---|---|---|
| `hierarchical_full_grid0_mamba` | 在 CoTracker 输出轨迹上增加双向时序残差修正器 | 原 CoTracker 全部参数 | 2 个 Mamba-style 块和二维残差头 | 改动小；初始严格等价原轨迹，残差限制在每轴 8 像素内，预期修正短时抖动，风险较低 |
| `hierarchical_full_grid0_mamba_replacement` | 将 UpdateFormer 中全部时间注意力块替换为双向选择性状态空间块 | 除替换块外的原 CoTracker 参数 | 全部替换后的时间块 | 改动激进；验证 Mamba 能否直接承担时序信息传播，可能更快，也可能因破坏预训练表示而明显退化 |

保守版输出头采用零初始化，所以训练开始时输出与基础 CoTracker 完全一致；
查询帧上的原始查询点会被强制保留。激进版先严格载入同一基础权重，再替换时间
块，避免其他网络部分发生无关变化。

## 固定训练与评估条件

- 训练：固定 40 例，仅训练新增或替换的 Mamba 参数。
- 教师：固定 `offline_cotracker_three` 单教师。
- 标签：沿用原始硬置信度标签，不启用软标签。
- 辅助教师权重：0，不使用教师重采样。
- 默认训练步数：1000；每 25 步原子保存断点。
- 选择：10 例验证集只用于判断结构是否值得保留。
- 最终结果：38 例公开测试集，报告时以它为准。
- 直接对照：`hierarchical_full_grid0`；同时保留原始 `baseline` 供背景比较。

冻结的完整配置见 `mamba-40-10-38.json`。

## 断点保护与输出

训练脚本会保留最近 3 个训练断点、优化器、调度器、随机数状态和数据位置。
断电后重新执行同一条命令，会从最近的 25 步断点继续。评估按病例原子提交；
已经具有输出、元数据、完成标记和诊断记录的病例会被跳过。

主要输出：

- `protocol-40-10-38/mamba-training`：模型权重和训练日志；
- `protocol-40-10-38/mamba-experiments/queue.log`：总队列实时日志；
- `protocol-40-10-38/mamba-experiments/validation-10`：10 例结果；
- `protocol-40-10-38/mamba-experiments/test-38`：38 例最终结果；
- `mamba-test-38-results.csv`：完整指标表；
- `mamba-test-38-deltas-vs-hierarchical-full-grid0.csv`：Mamba 相对直接对照的差值。

指标方向为 DSC、D98 越大越好；HD95、MASD、CD 越小越好。最终判断必须同时
查看五项指标和运行时间，激进版若仅加速但精度明显下降，不应替代正式候选。

## 远程电脑启动

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"

git -C $Repo pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\run_mamba_40_10_38_resumable.ps1" `
  -RepoRoot $Repo `
  -NumSteps 1000 `
  -SaveEverySteps 25
```

运行中可在另一个 PowerShell 窗口查看：

```powershell
Get-Content `
  "$Repo\protocol-40-10-38\mamba-experiments\queue.log" `
  -Tail 80 -Wait
```

如果断电，Docker Desktop 恢复且 GPU 可用后，重新执行完全相同的启动命令。

## 方法依据

- Mamba 论文：<https://arxiv.org/abs/2312.00752>
- 官方实现：<https://github.com/state-spaces/mamba>
