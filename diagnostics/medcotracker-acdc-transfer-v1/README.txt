MedCoTracker：ACDC -> TrackRAD 交接烟测与单种子 pilot
====================================================

目的
先检验真实医学时序预训练是否帮助当前冻结主线，不改变教师池，不同时搜索额外技巧。
比较的是各路线训练后的学生在 TrackRAD validation-10 的官方指标，不是教师自身表现。
本包不提供 test-38、三种子确认或正式全量训练入口。

五条路线
R1：不在 ACDC 上训练；官方 scaled 主干 + 新 Mamba，作为同期对照。
A1：ACDC 仅训练 Mamba；完整继承已训练支线，官方主干不变。
B1：ACDC 仅训练 fnet；交接时冻结适配后的主干，接全新 Mamba。
B2：ACDC 仅训练 updateformer（含输出头，不含 corr_mlp）；然后冻结并接全新 Mamba。
B3：ACDC 小学习率训练整个原主干（没有 Mamba）；然后冻结并接全新 Mamba。
所有路线进入 TrackRAD 后均只更新 Mamba，冻结的是各自交接后的主干，不是强制恢复官方权重。
B1/B2/B3 与 R1 的新支线初始张量完全一致。A1 继承训练过的支线，不重新置零。

固定预算与监督
外部预训练：external-train-80，500 步，batch=1，128 查询，384x512，4 次迭代。
每个 epoch 每位患者取一次，确定性打乱；连续原生帧 min(16,T)，不重复/补帧。
A1/B1/B2/B3 学习率分别 5e-5/1e-6/1e-6/5e-7，AdamW，常数学习率。
同一步的病例、切片、时间片、查询与教师标签通过缓存哈希配对。
三位教师固定：v2(cotracker2v1, support5, iter4)、on3(baseline_online, support0, iter4)、
offscaled(scaled_offline, support100, iter4)。三位均已训练、冻结、只推理。
使用既有伪轨迹损失：可见 Huber*.05 + 不可见 L1*.01；不增加 mask、V/C 或辅助教师损失。
标签可见度/置信度不是校准后的点对应准确率。ACDC ED/ES 标注未用作轨迹真值。
ACDC external-dev-20 未参与训练或路线筛选；本轮以 TrackRAD validation-10 作为下游筛选。

TrackRAD：train-40，seed0(20261008)，250 步，60 帧/128 查询/4 次迭代，支线学习率5e-5。
复用原训练器、AdamW、1100-step OneCycle 日程，只将实际停止步数设为250；不改损失。
推理固定1000轮廓点、2次迭代、hierarchical_full_grid0_iterations2_memory_topk_diverse。
不得拿250步 R1 与过去1000步主线混为同预算对照。
外部预训练增加了数据与更新次数，不能把所有增益归因于医学数据本身（非等计算量消融）。

阶段
plan：核验源代码、已通过的外部烟测、数据清单、权重与固定镜像，无训练。
smoke：继承外部5步烟测结果，5条路线各跑 TrackRAD 10步，然后核验冻结张量、
       三个教师槽的真实配对输入/标签、严格载入与推理镜像接口。
       原拟5步不足：固定seed首次离线教师出现在step7，所以交接烟测用10步。
       推理接口烟测使用合成输入，不称为医学准确率验证。
pilot：要求交接烟测通过；从官方权重重新开始500步外部预训练，不续接烟测权重/优化器；
       再做5路线各250步 TrackRAD，最后运行6组validation-10（含未训练参照）官方评估。
all：plan -> smoke -> pilot。任一失败停止，不自动进入三种子或test-38。

筛选
只比较同期 R1：DSC严格更高，D98最多下降0.002，HD95最多上升0.02。
满足者按 DSC高/HD95低/CD低/D98高/名称排序。无合格者保留R1。
单种子只用于决定是否值得三种子确认，不报告种子标准差或统计显著性。
ACDC 心脏门控cine与肿瘤MRI-linac仍有域差异；损失下降不代表真正对应更准确。

安全、续跑和日志
新结果根默认 C:\zhongliu\mct-ab-v1，不改任何既有实验结果、原始数据或权重。
只使用本机已有固定镜像，不构建/拉取、不联网。训练容器不挂载validation/test数据。
数据审计阶段只核对train/validation文件，test仅读取原清单中的病例ID检查互斥。
源代码/数据/镜像变化会拒绝混用缓存；保持固定checkout完成本队列。
外部每25步原子保存模型/优化器/已提交记录；保留当前及前一版本。重启从最近完整状态
继续，以确定性epoch/step重新构造输入和随机状态。已有标签缓存必须逐项验证。
TrackRAD沿用已修复的loader位置、teacher RNG、优化器/调度器续跑，保留最近3个检查点。
这些轮换仅作用于本新实验的中间检查点；最终模型、审计与日志保留。
只允许一个队列占用输出根；不与另一GPU训练并发。日志写入每路线training.log与queue.log；
控制台每25步一条。单个外部更新超过120秒/显存保留超过22GiB、TrackRAD中位步时超过8秒
或显存余量不足会停止。外部每路线进程限时3小时，TrackRAD pilot每路线1小时。
中断时保留所有结果并重跑同一条命令；不要删除缓存或改SHA强行继续。

远程执行（先git更新，Docker Desktop正常运行）
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\medcotracker-acdc-transfer-v1\run_experiment.ps1" -Stage all

前置默认目录
ACDC：C:\zhongliu\acdc-official-v1
数据审计：C:\zhongliu\acdc-ab-native-v2
已完成外部GPU烟测：C:\zhongliu\acdc-ab-gpu-smoke-v1
原固定镜像/TrackRAD数据清单：C:\zhongliu\ts-v1
可通过参数修改路径，但不能更换清单内容混用旧输出。

完成后发回
C:\zhongliu\mct-ab-v1\reports\validation-10\summary.csv
C:\zhongliu\mct-ab-v1\reports\validation-10\deltas-vs-R1.csv
C:\zhongliu\mct-ab-v1\reports\validation-10\selection.json
总完成证明：C:\zhongliu\mct-ab-v1\summary.json
