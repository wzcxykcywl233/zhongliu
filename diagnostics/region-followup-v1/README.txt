区域监督后续：G1探索、D1诊断、点输出测试（2026-10-30）

目的与边界
用户已明确要求“跑G1、诊断D1，其他的跑验证”。本包把三件事隔离，旧rs-v1只读。
G1原门槛失败仍如实保留；这是门槛失败后经用户授权的额外探索，不伪造通过记录。
既不重新挑选M0/W1/D1权重，也不依据38例结果改变G1训练或D1诊断方案。

1. points：正式报告既有pretrained、M0/W1/D1点输出的test-38结果。
   pretrained一次，其余三个种子，完全复用rs-v1的已完成权重。
   这是用户要求的点输出子集，不宣称完成旧selection中的D1-region测试。
   all现在明确包含这个已授权的38例报告；不自动测试G1。
2. diagnose：固定D1三个种子，validation-10，不训练。
   每例只跑一次轨迹，所有解码对照共用轨迹、初始掩膜和权重：
   raw：原始强度区域头；percentile：仅区域特征路径p1/p99映射0..255；
   prior：去掉学习残差，仅保留轨迹平均位移平移的低分辨率先验；
   static：第一帧掩膜直接沿时间复制（负对照）。
   另存点输出，检查raw/points是否精确复现旧输出，差异体素数不会被掩盖。
   保存逐帧Dice、面积、空掩膜、质心像素距离、时间分段、特征/残差幅度；
   汇总旧D1训练早中晚损失、轨迹散度及可见率。查询帧排除，空/空Dice定义为1。
   这些是诊断指标，不是官方DSC或精确点追踪真值，不能直接替代官方排名。
   百分位公式与训练一致，但应用于整段验证视频，而训练使用采样片段。
   因此即便改善也只提示输入分布问题，不能单凭此确定全部退化原因。
   所有预测固定后才读取未来真值做离线评价；不回传梯度、不自动替换模型。
3. g1：三个种子，各2000步（1000 TrackRAD + 1000 ACDC）。
   ACDC仍78训练/20开发，排除076、082；TrackRAD仍40/10/38隔离。
   官方scaled主干冻结、全新compact Mamba与区域头，初始化与D1配对；
   AdamW lr=5e-5，weak=.05、region=.1，与D1相同。
   唯一损失改动：去掉教师坐标损失；仍使用教师可见度筛选弱监督点。
   只读取并校验rs-v1的教师标签缓存，缺失则停止，绝不悄悄重生。
   每25步原子保存优化器与记录；继续相同命令可恢复。
   完成后审计与D1逐步输入、标签、初始化一致以及冻结权重完全不变。
   只在validation-10评估G1区域输出，比较既有M0-points/D1-points/D1-region。
   不自动选型、不自动跑G1 test-38。没有新增G1点输出的正式终点。

运行（远程已有镜像、rs-v1完整结果和标签、数据；不构建、不拉取、容器不联网）
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\region-followup-v1\run_experiment.ps1" -Stage all

默认输出C:\zhongliu\rs-next-v1。可分开用-Stage points / diagnose / g1，-Stage plan只冻结计划。
all顺序为points、diagnose、g1；GPU严格串行。单分支失败记录后继续独立分支，末尾报告失败。
初始计划不通过则不开始任何分支。不要同时开两个同输出目录的队列（独占锁）。
旧脚本和旧结果不改动；新输出不得与旧结果、数据或代码目录重叠。

预期耗时
G1为6000个优化步，若继续保持2.4秒/步，训练本体约4小时，另加加载/审计/评估。
总耗时还包括10个38例推理队列、30个D1验证病例重放、3个G1验证队列；以远程日志为准。
本地只有CPU，已做单元与模拟启动器检查；未宣称远程GPU运行已验证通过。

结果位置
reports\points\test-38\summary.csv、paired-deltas.csv、case-deltas.csv
reports\diagnosis\validation-10\summary.csv、frames.csv、temporal.csv、training-phases.csv、replays.json
diagnosis\seed_N\CASE\raw.mha / percentile.mha / prior.mha / static.mha / points.mha / feature-metrics.csv
reports\g1\validation-10\summary.csv、paired-deltas.csv、case-deltas.csv
followup-plan.json绑定旧selection、权重、报告与新源码；旧selection不变。
公开test-38已经多次查看，本轮不宣称盲测、显著性或临床可靠性。
