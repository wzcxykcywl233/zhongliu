# 四向镜像回查（只诊断，不修正位置）

## 固定基础与问题

基础为 `hierarchical_full_grid0_iterations2_memory_topk_diverse`，固定2次迭代和原始权重。不叠加P/V/C继承、额外教师、软标签、Mamba或掩膜外观平移，不训练。

对每个目标帧t，每个跟踪点使用该帧局部CoTracker第1次更新位置P(1)与最终双锚点融合位置P*，计算 delta=P(1)-P*。若delta=(2,1)，历史四向偏移为(2,1)、(-2,1)、(-2,-1)、(2,-1)。每个方向比较查询帧Fq(Q+delta_j)与目标帧Ft(P*+delta_j)。查询帧仍是当前分段锚点，不是逐帧改用上一帧。

这是四向回查，不是搜索新中心；评分不参与位置、可见性、置信度、融合权重、记忆写入或后续查询点更新。

## 四种评分、两次推理

| 评分字段 | 定义 |
|---|---|
| center | 仅中心余弦匹配 |
| center_history | 0.5中心 + 0.5单向历史匹配 |
| center_history_four | 0.5中心 + 0.5四向历史匹配均值 |
| center_fixed_four | 0.5中心 + 0.5固定四向匹配均值 |

固定四向为(±a,±a)，a=4/sqrt(2)，即半径4模型像素的四个对角位置。不是半径4的横竖十字对照。历史四向的半径随历史偏移变化，因此与固定四向的差异也可能来自半径，不能全部归因于历史路径。

四种评分在 `fourway_backcheck_i2` 的同一次推理中计算；`fourway_control_i2` 关闭回查用于预测哈希核验和计时对照。共两组推理，不需要跑四组相同模型。开关回查后每个病例的预测数组必须完全相同，否则离线分析失败。

## 有效性与公平比较

- 每帧使用自己的迭代历史；排除最后一次更新和初始猜测。
- 历史偏移半径在[0.5,16]模型像素，四个镜像位置两两至少相距0.5像素。
- 零偏移或接近坐标轴使镜像位置重复时，该点本轮不参与四项主评分，不冒充四个独立样本；记录退化点数。
- 四个历史位置、四个固定位置及中心在查询端和目标端均不越界，才接受该点。不会clamp到图像边缘。
- 四项评分使用完全相同的点。单向历史分数在本轮共同有效点上重新计算，不直接照搬上轮不同覆盖率的结果。
- 无有效点时记null。有效点少于该帧点数的10%时，保存记录但不进入主要相关性/AUROC统计。
- 特征为第0层编码特征，模型像素按stride转换，双线性采样后做L2归一化与余弦匹配。
- 四向均值再与中心各占一半；不因增加采样点而提高邻域的总权重。不做最大方向择优。
- 记录四向单独均值和每点四向分差的帧均值，但它们本轮不用于修正或另行挑选最优规则。
- 分段首帧不重复计分；原始首帧排除。发生遮挡合并后替换被覆盖的旧评分，保证每个非首帧恰好一条记录。

该评分基于局部近似平移。形变、旋转、背景混入与重复纹理可能使其失败，镜像样本也不是独立证据。当前分段锚点本身还可能有漂移。

## 统计与边界

沿用40训练/10验证/38测试划分（40例仅审计，不训练）。先跑10验证再跑38测试，所有设计运行前固定，不根据测试结果调整阈值。

预测完成后才读取真值计算离线诊断：优先STAPLE标签，排除首帧/空真值帧，以DSC<0.8定义低质量帧。官方DSC/HD95/MASD/CD/D98仍只从评估器metrics.json取得，诊断逐帧DSC不可替代官方指标。

主要看病例内Spearman均值、每病例配对变化及识别低质量帧AUROC；合并帧相关性是辅助描述。所有四种分数用同一帧和点，Scope为`common_four_scores`。不把相邻帧当独立样本做显著性检验。覆盖不足时如实报告，不将缺失补成高分。

由于38例测试集已被多次查看，这是补充诊断，不能声称为新的盲测确认。只有评分有效性得到支持，才另设实验研究评分到位置修正的反馈，本轮没有反馈。

## 运行、日志、断点

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"
git -C $Repo pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$Repo\scripts\run_fourway_backcheck_resumable.ps1" -RepoRoot $Repo
```

默认结果根目录为 `C:\zhongliu\zhongliu-tuning\protocol-40-10-38\fourway-backcheck`，独立于上一轮。先确认其他GPU实验已结束。总共2×(10+38)=96次病例推理，串行执行。

根目录queue.log、各split的runner.log和backcheck-analysis.log、每病例case.log/diagnostics.json实时记录。数据/配置/源码/权重与镜像冻结；断电后恢复Docker与GPU，再执行同一命令，跳过已完成病例，未完成病例重跑。可加`-Stage validation`或`-Stage test`（后者要求验证已完成）。离线统计失败也可重跑同一命令。

如源码/数据变化，拒绝混用旧缓存，应指定新的ResultsRoot。不会覆盖上一轮实验，不生成临时可执行脚本，不关闭杀软，不修改系统执行策略；安全软件阻止时保留告警排查。

## 完成后发回（仅test-38目录）

1. `fourway-score-summary.csv`：四种评分的相关性、AUROC和覆盖率。
2. `fourway-case-deltas-vs-center.csv`：病例内相对中心评分的变化。
3. `fourway-analysis-summary.json`：输出一致性、有效帧数、各类无效点数。

其他文件：`fourway-frames.csv`逐帧完整评分，`fourway-score-cases.csv`逐病例统计，`fourway-output-checks.csv`逐病例一致性；`fourway-backcheck-test-38-results.csv`为两组官方性能和运行时间，指标应完全相同。10例验证结果只留在validation-10，不混入38例正式表现表。
