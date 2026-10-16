教师独立搜索与冻结 Mamba 学生自动组合实验 v1
==================================================

一、目的和边界

不同教师不一定适合相同的改法。本实验分别寻找各教师相对其自身默认设置的改进，
但评价标准是“它教出的固定学生在 validation-10 上的表现”，不是教师置信度或
教师自己的分割得分。通过三种子确认后，再测试教师池组合，最后自动报告 test-38。

保持用户确定的主线：冻结官方 CoTracker 主干，只训练既有 Mamba 特征支线。
不微调教师，不解冻学生主干，不增加新的支线结构，不改变损失、数据划分或推理配置。
本轮仅搜索辅助点数量与教师迭代次数，不把此前层级记忆、动态跨度等直接移植到
任意时刻查询接口。后续若扩展，需要独立协议；本轮不是穷举全部可能的教师改法。

共有三个教师槽位：
  槽位 0：CoTracker2.1，cotracker2v1.pth（v2）。
  槽位 1：在线 CoTracker3，baseline_online.pth（on3）。
  槽位 2：离线 CoTracker3，在 baseline_offline.pth（offold）与
          scaled_offline.pth（offscaled）两个权重系列中分别寻找改进，再择一。
两个离线权重不是同时添加为第四个教师；仍维持原三个槽位的抽样概率和抽样顺序。

二、固定学生和可比性

主干 25,385,700 参数冻结；Mamba 支线 3,377,325 参数可训练；总计 28,763,025。
使用 compact 扫描，gain=1，无修正上限，无分块检查点。与之前冻结主线保持一致。
训练：40 例，60 帧/片段，128 个任意时刻查询点，学生 4 次迭代，lr=5e-5，
hard confidence，无辅助教师加权损失，无真实掩膜损失。
种子：20261008 + 0/1/2。每组从同一官方学生权重及同种子支线初始化开始。
推理固定 hierarchical_full_grid0_iterations2_memory_topk_diverse：1000 点、2 次迭代。
不混入 250 点方案，不把教师辅助点数量当作学生轮廓点数量。

每步先执行原本的辅助点抽样，再使用对应数量，确保不同配置的随机数消耗一致。
逐步审计病例、视频、目标查询、教师槽位、辅助点原始抽样和 P/V/C 标签。
未改动槽位的标签必须相同；被改动槽位必须实际改变过标签。
审计主干所有权重保持完全不变，支线确实学习，推理支线实际启用。

三、独立搜索矩阵

教师系列     默认辅助点          减点候选           迭代次数候选
v2           5x5 网格=25点       0点、3x3=9点       2、6、8（默认4）
on3          5x5 网格=25点       0点、3x3=9点       2、6、8（默认4）
offold       100个随机辅助点     0点、50点          2、6、8（默认4）
offscaled    100个随机辅助点     0点、50点          2、6、8（默认4）

每个系列为默认 + 5 个单因素候选，其他槽位不变。不是辅助点与迭代次数的全交叉。
在线教师窗口固定16，离线固定60，目标查询点的时刻、顺序和数量不变。
control：v2 默认 + on3 默认 + offscaled 默认，是当前冻结学生的较优教师池参照。
oldpool：仅将 control 离线教师换为 offold 默认，提供原教师池对照。
三个系列共用 control，因此24个系列配置实际为22个不重复的教师池。

四、自动队列

smoke：
  真实 GPU 检查所有24个系列配置的教师接口，查询时刻包含0/29/59。
  四个默认接口必须与原始教师调用输出完全一致；教师权重不得变化。
  再对 control、v2_s0、on3_s0、offold_i6、offscaled_slo 做各10步学生训练，
  检查配对、冻结状态与真实学生 GPU 推理。未通过，不启动后续训练。

pilot：
  22个教师池，每组 seed0、250步，再评估10例验证集。
  每个教师系列只保留一个合格相对赢家；没有合格改进则保留该系列默认。
  offold 的相对参照是 oldpool；其他系列的相对参照是 control。

confirm：
  control、oldpool 和最多4个系列赢家，最多6个教师池，各3种子、1000步。
  重新从官方权重和相同种子支线初始化训练，不从 pilot 250步检查点继续。
  pilot/full 都使用1000步学习率计划，避免短程学习率计划制造额外差别。
  此阶段统一与 control 比较；每个槽位保留一个合格改进。
  槽位2比较 offold、offscaled 两个系列的确认结果，选择其中一个，或保留 control。

combine：
  对通过确认的槽位改进测试所有子集，包括单项、两两组合、三项组合。
  至多8个组合池（含 control），另保留 oldpool。已完整训练的相同配置直接校验复用。
  最多新增4个组合池，每个3种子、1000步，然后用10例验证集选择。
  同时冻结最佳合格多槽位组合、整体最佳方案及最终 test-38 名单。
  如果没有合格多槽位组合，明确标为未产生合格组合，并保留 control，不能强行宣称组合最优。
  整体赢家可能是单项；组合收益不一定可叠加。报告包含组合与对应单项的配对差值。

test：
  自动评估已冻结名单：control、oldpool、通过确认的单槽位方案、最佳合格组合、
  整体赢家，以及无支线的官方权重直接推理参照。相同方案去重。
  使用38例测试集，仅评估既有完整检查点，不训练、不根据测试结果重新选择。
  因此用户要求的“最终最佳组合38例测试”已包含在 -Stage all 的末尾。

五、预先固定的选择规则

相对对应参照，候选需满足：
  平均 DSC 提升 > 0.00001；
  平均 D98 下降不超过0.002（0.2个百分点）；
  平均 HD95 增加不超过0.02（官方指标单位）；
  三种子完整训练时至少2/3种子 DSC 上升，且单种子 DSC 下降不超过0.002。
合格候选按 DSC、HD95、CD、D98、配置名依次排序；无合格候选则回退参照。
这些是工程筛选约束，不是统计显著性检验。MASD/CD 和所有失败组也完整报告。
允许小幅 D98/HD95 代价，不代表全指标获胜；报告必须保留这些权衡。

六、预计开销与保护

smoke 50步；pilot 22x250=5500步；确认与组合合计最多30组1000步。
最坏共35,550个优化步。按之前1.7-2秒/步估算，仅训练约17-20小时；
不包括验证、检查点、GPU接口检查和审计，8次迭代教师也可能更慢，不能保证总耗时。
按最坏选择分支，最多73个不同的评估目录；验证为10例，最终测试为38例。
只串行运行GPU任务，不会自动并发其他训练。不要与现有占显存任务同时运行。
单组超时：smoke 600秒、pilot 3600秒、full 14400秒。
两步预热后最近5步中位数超过8秒、峰值显存超过20GiB或可用显存低于2GiB会停止。
计时包含教师和训练计算，不含所有加载/保存时间；不是纯Mamba耗时。
每25步保存原子检查点（smoke每步），保留最后3个；固定种子、训练数据顺序支持恢复。
遇到配置、来源、标签配对或哈希不一致，停止并保留证据，不跳过审计、不清空旧结果。

七、远程运行

依赖已有 C:\zhongliu\ts-v1 下的冻结清单和本地镜像记录，以及仓库 training-checkpoints。
沿用已有镜像精确ID；不构建、不拉取镜像、不联网、不安装新依赖。
新包与旧实验隔离，不改变原算法源文件和旧结果。Docker Desktop 必须已启动。
默认输出 C:\zhongliu\tps-v1；路径保持短（最多32字符），且不放进仓库或数据目录。

先更新仓库，命令逐行运行：
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

可先检查计划：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-pool-search-v1\run_experiment.ps1" -Stage plan

先做真实GPU检查：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-pool-search-v1\run_experiment.ps1" -Stage smoke

通过后自动跑完整队列，末尾自动测试38例：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-pool-search-v1\run_experiment.ps1" -Stage all

all 自带 smoke 闸门，已通过则校验复用。断电/中断后原命令重跑，已完成内容校验复用。
也可分阶段运行 -Stage pilot / confirm / combine / test；必须已有前一阶段冻结选择。
新源码或协议不应混入现有输出；需要新实验时使用新的 -OutputRoot。
若先前冻结来源不匹配，先定位差异，不要删除清单绕过检查。

八、结果位置

C:\zhongliu\tps-v1\queue.log                       简洁阶段进度
C:\zhongliu\tps-v1\manifest.json                   原始数据/权重/代码/镜像清单
C:\zhongliu\tps-v1\pilot-selection.json            各教师系列短程选择
C:\zhongliu\tps-v1\slot-selection.json             三种子确认后的槽位选择
C:\zhongliu\tps-v1\selection.json                  冻结最终组合、整体赢家、38例名单
C:\zhongliu\tps-v1\reports\pilot\validation-10\     22组短程学生比较
C:\zhongliu\tps-v1\reports\confirm\validation-10\   三种子确认
C:\zhongliu\tps-v1\reports\combine\validation-10\   组合与单项比较
C:\zhongliu\tps-v1\reports\test\test-38\            最终正式38例报告

每个报告目录包含 summary.csv（均值/种子标准差）、results.csv（逐种子）、
paired-deltas.csv、paired-cases.csv、scores.json、audit.json。
reports\阶段\label-runtime.csv 记录各槽位时间和原始标签诊断。
各训练组 training.log、paired-steps.jsonl、initialization-report.json 供追溯。
各评估组 launcher.log 保存详细病例日志，控制台不再逐例刷屏。

九、解释限制

验证集10例被多轮使用，短程seed0筛选可能漏掉收敛较慢或其他种子有效的方案。
只确认各系列一个赢家；最终是本轮有限搜索内的相对最佳，不是全局最优。
38例历史上已多次查看，本轮仍不以其选择，但不能声称全新盲测。
N=3是训练种子数，预训练参照N=1的SD=0不代表跨病例没有波动。
P/V/C标签变化证明修改生效，不等于对应点真值准确率；学生提升才是本实验目标。
本轮不证明Mamba优于MLP，也不证明教师微调有效，因为支线结构固定、教师未训练。
本地只做CPU单测和Windows队列模拟；真实GPU行为需远程 smoke 验证。
