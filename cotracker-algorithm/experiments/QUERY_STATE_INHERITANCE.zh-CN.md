# 动态查询锚点的 P/V/C 跨段继承实验

## 要回答的问题

现有层级跟踪已经将上一段末帧位置P作为新查询位置，并传递历史查询特征；但每次新调用离线CoTracker时，可见性V与置信度C的内部状态重新置为零logit（概率0.5）。这组实验只改变V/C的初始化，不增加教师、不使用软标签、不重新训练、不改变掩膜重建。

主干仍是原始权重CoTracker。固定对照为 `hierarchical_full_grid0_iterations2_memory_topk_diverse`，新名称 `pvc_control` 与其配置完全相同。仍每10帧更新查询位置（保留原有遮挡合并），不是另加自适应锚点时间选择。P与查询特征的传递方式保持不变。

## 六组冻结配置

| 配置 | V初始状态 | C初始状态 | 主要比较/预期 |
|---|---|---|---|
| pvc_control | 原样零logit | 原样零logit | 固定对照，重跑核验 |
| pvc_v | 继承并广播至整段 | 不继承 | vs control：可见性历史是否有帮助 |
| pvc_c | 不继承 | 继承并广播至整段 | vs control：置信度历史是否有帮助 |
| pvc_vc | 同时继承、整段广播 | 同左 | vs v/c：二者联合效果；可能放大过度自信 |
| pvc_vc_decay | 继承后按时间距离衰减 | 同左 | vs vc：降低远离边界时过期先验的影响 |
| pvc_vc_query | 只初始化新查询帧 | 同左 | vs vc：区分边界状态传递与整段先验广播 |

除配置字段 `query_state_inheritance` 外，各组设置相同：跨度10、迭代2、grid0、原始特征权重0.5、双锚点权重0.5、四槽Top-K多样性记忆，使用同一 `scaled_offline.pth`。不增加其他优化组合。

## 精确计算定义

使用上一段**已经提交的、局部/全局融合后的末帧**概率 `v_end, c_end`，每个点分别处理。全局分支不继承，首段也不继承，保留原始初始化。

令 `epsilon=1e-4`：

```
p = clamp(v_end 或 c_end, epsilon, 1-epsilon)
L = log(p / (1-p))
下一段第k帧的初始logit = s(k) * L
```

- 整段广播：`s(k)=1`。
- 衰减：`s(k)=exp(-k/10)`；k为新查询帧后的实际帧偏移，k=0完整继承，k=10剩原logit的约36.8%。衰减趋向概率0.5，不是趋向概率0。
- 仅查询帧：`s(0)=1`、其他帧为0。
- 首段或未启用的状态：零logit，与对照完全相同。

初始化之后仍由原Transformer更新坐标、V、C；**不是强制锁定输出，也不是在每次内部迭代反复覆盖状态**。如果遮挡合并回退到更早锚点，取那个起点当时保存的先验，不使用失败段末帧的状态。合并段的衰减尺度仍固定10帧。

P沿用原有边界位置；不额外继承运动速度、Transformer隐状态或优化器状态。概率转logit后转到模型所在设备，并检查形状、有限性，避免CPU/CUDA混用和0/1概率产生无穷。

## 风险与可解释性

这是一项推理初始化实验，不等于重新训练CoTracker。预训练模型原本从零logit开始，非零先验可能造成分布变化；高置信度也可能让错误被跨段放大。它可能改善、无效或变差，不预设收益。

V/C既参与Transformer输入又是增量更新的初始值，最后也会影响原有分支融合与记忆可靠性。这些是本机制的下游影响，并不意味着单独修改了融合公式。`pvc_vc_query`只在边界注入，影响其他帧与否由原网络交互决定。

## 数据与执行

沿用40/10/38划分，校验数量及病例ID互斥。**本轮不训练**，40例只参与划分审计；先评估10例验证集，再评估38例测试集。六组预先冻结，不根据测试结果在线调整参数。所有正式性能表取 **test-38**；validation-10单独保留。公开测试结果已多次查看，本轮属于补充实验，不宣称新的盲测。

远程PowerShell：

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"
git -C $Repo pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$Repo\scripts\run_query_state_inheritance_resumable.ps1" -RepoRoot $Repo
```

默认根目录：`C:\zhongliu\zhongliu-tuning\protocol-40-10-38\query-state-inheritance`。

可用 `-Stage validation` 或 `-Stage test` 单独运行，test模式要求该目录内验证集六组结果齐全。首个默认all命令自动串行完成两部分。不要同时运行其他GPU实验队列。

## 断电与日志

复用已有病例级原子提交、失败继续下一配置、最终汇总报错、目录独占锁、镜像冻结与数据/源代码/权重哈希冻结。每例有 `case.log` 和 `diagnostics.json`，split有追加 `runner.log`，根目录有追加 `queue.log`。不覆盖其他旧实验。

断电重启后，确认Docker与GPU恢复，重新执行同一命令：已完成病例跳过，未提交的病例从头推理。本轮没有训练步数，因而不涉及优化器断点。若代码、权重或数据变化，必须指定新的 `-ResultsRoot`，不会强行复用不同条件的旧缓存。不要在运行中拉取更新。禁止通过关闭杀软绕过安全拦截。

另开PowerShell看实时日志：

```powershell
$Root = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\query-state-inheritance"
Get-Content -LiteralPath "$Root\test-38\runner.log" -Tail 40 -Wait
```

test-38目录尚未生成时，查看 `validation-10\runner.log`。

## 输出与审核

38例性能结果：`test-38/query-state-test-38-results.csv`，包含DSC、HD95、MASD、CD、D98、TimeSec；对照差值：`query-state-test-38-deltas-vs-fixed-control.csv`。DSC/D98按现有评估口径越大越好，HD95/MASD/CD越小越好。

同目录另外保存：

- `query-state-test-38-cases.csv`：逐病例指标差值及 `OutputMatchesControl`，避免只看聚合指标判断机制是否运行。
- `query-state-test-38-diagnostics.csv` / `mechanisms.csv`：继承段数、传入状态数量、非零状态数、logit绝对值总和、高于0.99/低于0.01的边界先验点数。
- `summary.json`、每配置的 `metrics.json`、预测输出及日志。

汇总脚本核对配置名、病例数/ID、实际继承模式与诊断字段；V-only不允许C被初始化，C-only反之；候选对应状态必须至少在某病例激活，避免空跑。`state_*_values`包含零logit元素；用`nonzero_values`判断非中性初始化，计数含遮挡重算的尝试，不是唯一帧数。

解释结果时先看control是否复现此前38例表现，再比较单V/单C，最后比较VC与两个限制传播范围的版本。不能将V/C数值升高本身当成跟踪更准确，也不能将测试集与验证集的指标混入同一性能表。
