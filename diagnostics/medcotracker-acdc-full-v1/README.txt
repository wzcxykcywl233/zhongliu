MedCoTracker：B1 / R1 三种子全量实验
==================================

为什么跑满
已完成的500步ACDC + 250步TrackRAD、单种子pilot中，只有B1通过预设筛选。
它相对R1五项指标全部改善，但DSC仅+0.00018133，尚未超过未训练参照。
用户决定直接使用原计划全量预算：外部3000步 + TrackRAD1000步、3个配对种子。
这不是原500/250预算的纯重复确认，同时增加了训练时长；与pilot的差异不能只归因于种子。
本轮只扩大B1和同期R1，A1/B2/B3保留pilot结果，不在本队列继续训练。

路线与不变项
R1：官方scaled主干 + 全新compact Mamba；TrackRAD train-40仅训练Mamba1000步。
B1：每个种子从官方scaled独立开始，在ACDC external-train-80仅训练fnet3000步；
    冻结适配后的整个主干，再接全新Mamba；TrackRAD train-40仅训练Mamba1000步。
同一种子R1/B1的新Mamba初始张量逐项相同，TrackRAD病例、视频、查询、教师及标签逐步配对。
3个种子不仅重复下游训练，也分别完成独立外部预训练，不共用一个ACDC适配权重。
学生/教师抽样种子为20261008/09/10；外部片段/查询/患者顺序种子为20261023/24/25。
ACDC固定80/20患者拆分保持不变，绝不随训练种子重划；20例external-dev不训练。

外部：batch1，128查询，384x512，连续原生min(16,T)帧，无重复补帧，4次迭代；
fnet学习率1e-6，AdamW，常数学习率，梯度裁剪10，仅原伪轨迹损失。
下游：60帧/128查询/4次迭代，Mamba学习率5e-5，原AdamW与1100步OneCycle日程，
实际训练到1000步。不是把pilot的250步学习率日程重新拉伸。
教师固定：v2(cotracker2v1,support5,iter4)，on3(baseline_online,support0,iter4)，
offscaled(scaled_offline,support100,iter4)。教师冻结只推理，不微调、不增加新教师策略。
不新增mask/V/C辅助损失，不把ACDC ED/ES分割标注解释为点对应真值。
下游推理仍为1000轮廓点、2次迭代、hierarchical_full_grid0_iterations2_memory_topk_diverse。
pretrained为官方权重+同一改进推理流程、无支线训练，不是完全未优化的最早原始基线。

预算与阶段
3次独立ACDC3000步 + 6次TrackRAD1000步；另有6次10步交接烟测。
不继承pilot优化器/模型。正式模型从各路线规定的初始状态重新开始。
烟测只使用已验证的5步外部B1检查点验证接口，不把它当作正式预训练结果。
plan：核验旧pilot/烟测结果、数据、权重、源代码、已有镜像，冻结新计划，不训练。
smoke：3种子各R1/B1共6次10步交接烟测、冻结/配对审计、实际推理镜像接口检查。
full：要求全部烟测已通过，然后依次训练全部正式任务，最后评估validation-10。
validation：仅恢复/执行已完成模型的validation-10评估及汇总，不训练。
all：plan -> smoke -> full -> validation。任一失败停止，不自动跑test-38。
test：显式执行；必须先有冻结的完整三种子validation报告。先核验报告及模型哈希，
      再解析/挂载test-38目录，评估固定pretrained/R1/B1，不用测试集更换胜出路线。
验证集或测试集各7组评估：1组pretrained + 3种子x2路线；pretrained不假装有3个训练种子。

确认规则
以B1-R1的3种子配对均值判定：DSC>0，D98>=-0.002，HD95<=0.02；
至少2/3种子DSC改善，任一种子DSC下降不超过0.002。失败则保留R1。
selection.json在validation-10完成时冻结。无论B1是否合格，显式test阶段均可报告
预先固定的两组对照，但不能因test结果重写validation选择。
同时报告相对pretrained的差值，不把“优于训练对照”说成“超过官方未训练参照”。
标准差是3个训练种子之间的波动，不是患者独立置信区间，无p值或盲测声明。
ACDC增加了医学数据和优化步数，非等总计算量消融；心脏cine不等同肿瘤MRI-linac。

新目录与安全
默认 C:\zhongliu\mct-full-v1（23字符），不改C:\zhongliu\mct-ab-v1及任何已有结果。
新包从已验证pilot实现固化而来，不修改旧包，以保留旧manifest的源代码哈希。
只复用本地固定镜像，不构建/拉取、不联网、不下载数据。
外部训练只挂载ACDC数据为训练输入；下游训练只挂载TrackRAD train-40为训练输入。
all/full不解析或挂载test目录，仅读取历史清单中的test病例ID检查互斥。
每25步原子保存，外部保留当前及上一优化器检查点，下游保留最近3个。
来源变化拒绝续跑；同一命令恢复时先验证已完成任务并跳过，不混用不同预算的检查点。
外部单更新>120秒/显存保留>22GiB停止；外部每种子进程最长8小时。
下游中位步时>8秒/显存保留>20GiB/余量<2GiB停止，每任务最长3小时。
上述是异常保护上限，不是运行时间预测。不要与另一个GPU训练队列并发。
日志每25步打印，完整记录在各任务training.log与总queue.log。

远程运行（先由用户推送本地提交，并在远程git更新）
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\medcotracker-acdc-full-v1\run_experiment.ps1" -Stage all

正常完成后先查看：
C:\zhongliu\mct-full-v1\reports\validation-10\summary.csv
C:\zhongliu\mct-full-v1\reports\validation-10\paired-deltas.csv
C:\zhongliu\mct-full-v1\reports\validation-10\selection.json

需要最终38例报告时，单独执行同一脚本 -Stage test；不重新训练。
结果在reports\test-38；病例级差值为paired-cases.csv，原始每种子指标为results.csv。

前置目录默认沿用pilot：
PriorStudy=C:\zhongliu\ts-v1
DataRoot=C:\zhongliu\acdc-official-v1
AuditRoot=C:\zhongliu\acdc-ab-native-v2
ExternalSmoke=C:\zhongliu\acdc-ab-gpu-smoke-v1
PilotRoot=C:\zhongliu\mct-ab-v1
OutputRoot要求独立且<=25字符，避免加上seed子目录后Windows路径过长。
