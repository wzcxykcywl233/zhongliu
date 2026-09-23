# 当前最优方法的分阶段参数复调

## 目的与边界

固定 CoTracker 权重和当前重建方法，重新检查加入动态查询记忆后的参数搭配。不训练、不使用软标签或辅助教师、不加入 Mamba。主对照 `rt_control` 是 `hierarchical_full_grid0_iterations2_memory_topk_diverse` 的别名；`rt_repeat` 为完全同配置复跑；`rt_decay` 为增加现有 V/C 衰减继承的候选对照。

40例用于固定协议和检查数据，10例验证集用于所有选择，冻结后才运行38例测试。最终报告使用 `final/test-38` 的正式指标。公开测试集已经反复查看，本轮不是新的盲测。10例验证集较小且搜索较多，存在选择过拟合风险，不应把最佳验证值等同于稳定收益。

## 首轮63组

54组单参数 + 3组对照 + 2组机制对照 + 4组匹配桥接对照。表内当前值复用对照，不重复建立同配置单参数。完整注册清单在 `parameter_retune.py`，计划时输出 `catalog.json`，包含短配置名与完整参数映射。

| 参数轴 | 候选值（含对照） | 目的 |
|---|---|---|
| span 分段跨度 | 3、5、8、10、15、20、30 | 局部适应和累积漂移平衡 |
| iters 迭代次数 | 1、2、3、4、6 | 匹配精度与计算开销 |
| slots 记忆总槽数 | 2、3、4、6、8 | 历史覆盖和过时记忆之间的平衡；包含原始锚点 |
| floor 原始记忆权重下限 | 0、0.15、0.30、0.50、0.70、1 | 身份约束强度；1为只读取原始记忆的端点 |
| diversity 多样性权重 | 0、0.10、0.25、0.50、1 | 增加记忆差异是否有益 |
| feature 记忆特征混合系数 | 0.10、0.25、0.50、0.75、1 | 混合历史记忆与当前锚点特征；不是最终位置融合系数 |
| reliability 写入可靠性阈值 | 0、0.50、0.90、0.97、0.99 | 覆盖可靠性偏高情况下的有效门槛 |
| similarity 写入相似度阈值 | −1、0.30、0.50、0.70、0.90 | −1是余弦相似度门槛下界，不放宽可靠性条件 |
| global 全局位置融合权重 | 0、0.25、0.50、0.75、1 | 全局与局部分支的贡献；端点不代表自动优化掉全部无用计算 |
| tau 状态衰减时间常数 | 0.25、0.50、1、2、4 | 以 rt_decay 为直接对照 |
| points 目标轮廓点数 | 250、500、750、1000、1250 | 精度/成本，1500仍排除 |
| grid 支持网格 | 0、3、5、7 | 检查小网格是否有益，不含成本高的15 |
| occ_visibility 遮挡可见性阈值 | 0.3、0.5、0.7 | 检查遮挡合并是否实际触发 |
| occ_fraction 遮挡点比例阈值 | 0.25、0.50、0.75 | 合并触发强度 |

衰减公式：`logit(t) = logit(boundary_probability) * exp(-offset/(span*tau))`。tau越小，回到中性概率0.5越快；默认tau=1与旧实现一致。其他轴以不继承状态的 `rt_control` 为对照。

两个机制对照：`rt_no_memory` 关闭记忆模式及槽位（明确不是单字段实验）；`rt_no_merge` 关闭遮挡跨度合并。原设计的1槽位和feature=0不符合现有记忆路径约束，不伪装成可运行单参数。4组 `rt_dglobal_*` 在 rt_decay 上单独改变全局融合权重，用来解释后续 global×tau 交互。

## 第二轮：最多20组定向组合

固定五对交互：span×iters、slots×diversity、feature×floor、reliability×similarity、global×tau。

每轴在首轮中按验证DSC降序、HD95升序、CD升序、D98降序选择至多两个非默认值。与直接对照完全相同输出的候选排除，同轴输出完全相同的候选归并，因此实际组合可能少于20组。选入组合不等于已经证明单轴有正收益；这是预先限定的交互检验。不会遍历注册表中所有92种可能组合。

每轮均跑 rt_control / rt_decay。global×tau以rt_decay为直接对照，去掉任一修改后都有对应首轮对照，不混淆“是否继承状态”和“衰减强度”。

## 最终入围与复核

只使用单参数和组合轮的验证结果，预先定义三个候选角色：

1. 精度：DSC优先，依次以HD95、CD、D98打破并列。
2. 边界：验证DSC不低于主对照0.001以上，在合格配置中优先最低HD95、其次MASD。
3. 速度：同一DSC容忍范围内优先最低total_time。

三个角色去重，不强求一定凑满三个；运行时间只是单次观测，不能当成严谨性能基准。候选可以没有正向收益，报告必须保留实际退化，不自动宣称优于对照。

`final.json`冻结最多三个候选、两个对照，以及入围组合对应的逐项撤回对照。先在10例上重新运行这些配置，逐病例输出哈希必须与筛选阶段一致，才允许进入38例测试。不会利用测试结果重新选参数，也没有自动“全部阶段包括测试”的默认入口。

## 运行

首次准备需本机已有 `python:3.11-slim` 作为标准库计划工具，不自动下载；推理/评价镜像由现有队列构建。计划模式不运行GPU任务：

```powershell
git -C C:\zhongliu\zhongliu-tuning pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\run_parameter_retune_resumable.ps1 -Stage plan
```

运行首轮、组合、入围复核（默认即validation）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\run_parameter_retune_resumable.ps1 -Stage validation
```

完成后查看 `protocol-40-10-38\parameter-retune\final.json`。确认后运行38例：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\run_parameter_retune_resumable.ps1 -Stage test
```

断电后启动Docker Desktop，再运行原命令；不是开机自动启动。正在计算但未提交的病例会重跑。运行期间不要更新仓库、替换权重、改动数据或清理镜像。显式传入 `-ResultsRoot` 可开新实验目录，不能覆盖旧结果。

## 日志、审计和文件

- 顶层 `queue.log`：整个搜索过程；每阶段保留现有实时 `runner.log`、病例 `case.log`、配置与机制诊断。
- `single.json / combination.json / final.json`：冻结队列，后两者包含用于选择的验证文件指纹。
- 各阶段 `frozen-run.json`：代码、权重、数据指纹；后续阶段强制与首轮镜像一致。
- 病例级断点提交；文件锁防止同目录重复运行。旧输出不会因为失败被直接删除。
- 所有新组固定随机种子20260923；使用对照重复检查实际可重复性，不宣称GPU绝对确定性。
- 实际配置、病例数、病例ID、输出文件SHA256及完成标记在阶段切换时审核。损坏输出会阻止选择，需检查并恢复相应病例，不能靠放宽检查通关。
- 单组失败后现有执行器继续后续组，但任何缺失组都阻止进入下一阶段。
- CSV保存DSC/HD95/MASD/CD/D98、时间、病例级差值、输出一致性；机制表记录写入接受/拒绝、融合权重累积、遮挡合并以及新组峰值CUDA张量分配字节（不是整张卡总占用）。
- `retune-matched-comparisons.csv`额外提供相对直接对照、入围组合逐项撤回对照的变化，避免只看相对主对照的汇总。

最终报告目录：`protocol-40-10-38\parameter-retune\final\test-38`。
主表：`retune-final-test-38-results.csv`；病例表：`retune-final-test-38-cases.csv`；配对表：`retune-matched-comparisons.csv`。

## 重建问题不混算

本轮不修改轮廓重建方法。先前6个重建失败帧可在 best-candidate-diagnostics/test-38/jobs/*/representation.csv 的 Valid=False 行查看 Reason。点数扫描可作为线索，但不能把真实掩膜往返Dice当作模型上限或预期提升。若后续发现重建缺陷，另设固定轨迹的实验，不并入本轮参数收益。

本地测试验证配置、选择、默认衰减一致性、模拟阶段恢复与损坏检测；完整GPU耗时和结果仍需远程运行，不预先承诺收益。
