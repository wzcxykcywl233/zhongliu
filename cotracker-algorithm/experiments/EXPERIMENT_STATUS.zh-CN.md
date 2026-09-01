# CoTracker3 实验主线与纳入状态

本文件规定后续实验筛选、结果排名和正式报告的纳入范围。历史代码与结果继续保留以保证可追溯性，但只有“正式主线”中的实验可用于选择后续组合或形成项目主结论。

## 正式主线

正式主线使用原始 `scaled_offline.pth` 权重，只调整 TrackRAD 推理、分级重锚定、原始特征记忆、支撑网格及迭代次数。当前组合实验均已在同一套 50 病例公开标注数据和统一评估脚本下完成。

指标方向：DSC、D98 越大越好；HD95、MASD、CD 越小越好。

| Profile | DSC ↑ | HD95 ↓ | MASD ↓ | CD ↓ | D98 ↑ | 时间（秒） |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.886662 | 4.620684 | 1.814792 | 2.182416 | 0.872867 | 1229.18 |
| `hierarchical_feat05_grid0` | **0.890206** | 4.531927 | 1.712859 | 2.030161 | **0.892454** | 297.67 |
| `hierarchical_full_grid0` | 0.889518 | **4.429874** | **1.689648** | **2.011340** | 0.889421 | 2557.30 |
| `hierarchical_full_grid0_iterations2` | 0.889173 | 4.433281 | 1.689715 | 2.013419 | 0.886086 | 776.67 |
| `hierarchical_feat05_grid0_iterations2` | 0.888969 | 4.494188 | 1.715699 | 2.053935 | 0.884987 | **271.37** |
| `hierarchical_feat05_iterations2` | 0.888040 | 4.632172 | 1.792322 | 2.150858 | 0.883097 | 273.80 |
| `hierarchical_full_iterations2` | 0.887760 | 4.604043 | 1.794593 | 2.167283 | 0.874494 | 866.02 |

当前正式候选分为三类：

1. **综合主候选：`hierarchical_feat05_grid0`**。取得最高 DSC 和 D98，五项精度指标均优于原始基线，运行时间约为原始基线的 24%。
2. **几何精度候选：`hierarchical_full_grid0`**。取得最低 HD95、MASD 和 CD，但运行时间约为综合主候选的 8.6 倍。
3. **轻量候选：`hierarchical_feat05_grid0_iterations2`**。时间最短，全部五项指标仍优于原始基线，但精度略低于综合主候选。

后续正式实验应从上述候选出发，并继续遵守一次只增加一个可归因变量的原则。默认优先围绕 `hierarchical_feat05_grid0` 开展局部参数验证；只有明确以边界和中心距离为主要目标时，才使用 `hierarchical_full_grid0`。

## 探索性实验：保留但不纳入正式流程

以下系列只作为方法探索，不参与正式排名、不作为后续组合的证据，也不写入正式实验报告的主结果表：

- `confidence_hard_12`、`confidence_soft_8_16`、`confidence_soft_6_18`；
- `confidence_head_hard_12`、`confidence_head_soft_6_18`；
- `confidence_updateformer_soft_6_18`；
- 随机辅助教师及其对照实验。

排除原因：

1. TrackRAD 教师伪标签训练并不等同于原始 Kubric 硬标签训练流程；
2. head-only 实验只训练置信度输出行，硬标签与软标签差异接近数值持平；
3. UpdateFormer 置信度重训练会改变共享表征，不满足“只替换原始硬标签”的严格定义；
4. 辅助教师方案训练成本较高，三种子复测未显示稳定收益。

这些实验可以在附录或研究日志中注明“尝试过但未进入正式方法”，但不得与正式推理组合放在同一排名表中。

## 结果来源

- 全量审计：`full-audit-results/audit-results.csv`；
- 六组组合：`hierarchical-combination-results/summary.json`；
- 六组组合运行器：`scripts/run_hierarchical_combinations_resumable.ps1`。

正式报告整理时，应先读取同目录的 `experiment-status.json`，过滤所有 `exploratory_excluded` 项。
