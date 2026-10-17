冻结 Mamba 主线的教师输入单项实验 v2
2026年10月17日

目的
上轮 teacher-pool-search-v1 已在验证集选出 on3_s0，38例报告三种子均有正向变化。
本轮把它作为新对照，探索教师输入视图是否影响伪标签和最终学生表现。
教师自身不训练；仅在生成伪标签时改变输入，学生始终接收原视频和原查询。
不改变学生结构、损失、教师抽样、查询数量、推理点密度或支线幅度。
不根据38例测试结果筛选新候选。

一 本轮参照及改动数
control = 上轮 on3_s0：
  槽0 CoTracker2.1 / cotracker2v1.pth / 25辅助点 / 4次迭代。
  槽1 在线CoTracker3 / baseline_online.pth / 0辅助点 / 4次迭代。
  槽2 离线CoTracker3 / scaled_offline.pth / 100辅助点 / 4次迭代。
oldpool 仅将本轮control槽2换为 baseline_offline.pth，槽1仍为0辅助点。
注意本轮oldpool不是上轮在线25点的同名oldpool；只作离线旧权重系列的直接参照。
四个系列 v2/on3/offold/offscaled 分别试4个单项候选，共16个改动，加2个参照共18池。
离线两个权重系列仍竞争同一槽位，不是新增第四个教师。

二 四种改动
flip_x  左右翻转教师视频、目标查询和实际使用的辅助点，输出x映射回 W-1-x。
flip_y  上下翻转，输出y映射回 H-1-y。
  两者分别检验预训练特征是否存在对MRI不合适的方向偏好。全片段采用同一变换，
  不插值、不裁剪，不改变查询时刻/顺序。V/C沿用对应点，不做翻转后平均或融合。
percentile  对完整60帧片段使用同一组1%/99%灰度分位数，裁剪并线性映射至0..255。
  分位数从展平视频以固定步长抽取至多131072个值估计，步长ceil(numel/131072)。
  三通道共用范围；禁止逐帧单独归一化。平坦片段范围<=1e-6则原样回退并记录。
  目的是测试对比度和极值影响，并非断言归一化一定有益。
gamma  float32下使用255*(x/255)^0.8，轻度增强低亮度区域，不额外归一化。
  固定指数，不扫描多个Gamma，不从训练/验证真值决定变换参数。

每项只改一个教师槽位的transform字段，其余设置不变。每次教师仍只前向一次。
时间反向暂不纳入：在线模型对任意时刻查询的有效前后区间不对称，需要另立协议。
所有变换只是候选，可能产生更差的伪标签；必须依据学生验证结果判断。

三 固定学生
冻结官方主干25,385,700参数，仅训练Mamba支线3,377,325参数，总计28,763,025。
40例训练，60帧/片段，128目标查询，学生4次迭代，lr=5e-5，compact扫描。
种子20261008+0/1/2；无真实掩膜损失、无额外教师加权。教师全部eval且冻结。
推理固定hierarchical_full_grid0_iterations2_memory_topk_diverse，1000点、2次迭代。
不是250点方案，不继承上轮学生训练后权重；新旧教师池学生都配对从相同初始化训练。

四 自动队列
smoke 真实GPU检查全部20个系列设置（含共享默认），默认输出必须与原始调用完全相同。
  查询时刻包含0/29/59；所有视图映射与直接模型调用核对，教师参数必须保持不变。
  然后control及四个覆盖变换/教师类型的候选各训练10步，审计并做GPU学生推理。
pilot 18池，每池seed0训练250步，以学生10例验证结果筛选每个系列一个相对赢家。
confirm control/oldpool和至多4个赢家，各3种子重新训练1000步，统一对control确认。
combine 对合格槽位改动的所有子集测试，至多8池；同一配置复用已审计完整训练。
  另保留oldpool。多个槽位合格才有新的多项组合，不强行拼接失败候选。
test 验证集冻结名单后，自动对control、oldpool、合格单项、最佳合格组合、整体赢家
  和无支线官方参照作38例评估；仅测试，不更新权重、不根据测试结果重新筛选。
all 包含以上全部阶段；烟测失败则停止，已完成阶段重跑时校验复用。

规则沿用上轮：均值DSC增加>0.00001，D98下降<=0.002，HD95增加<=0.02；
三种子确认至少2个DSC提升、任何单种子下降不超过0.002。合格后按DSC/HD95/CD/D98/名字排序。
无合格候选回退control。规则允许微小指标代价，不是全指标胜出或统计显著性声明。
完整保留未入选结果；只有每系列一个短程赢家进入确认，可能漏掉慢收敛候选。

五 安全及预计开销
沿用C:\zhongliu\ts-v1的冻结清单、已有精确镜像ID和本地训练权重。
新结果默认C:\zhongliu\tis-v2，不混用tps-v1缓存。不改原算法代码，不构建/拉取镜像、不联网。
逐步核对原病例/视频/查询/教师抽样/辅助点随机数；未改槽位标签必须完全一致。
记录变换模式、分位数回退、变换后的查询哈希、视频抽样哈希、映射后P/V/C哈希。
视频抽样哈希最多65536个值，仅是机制证据；原训练视频的完整哈希仍由原训练审计保存。
主干必须完全冻结，支线必须产生非零更新。来源或输出变化则停止，保留证据。
GPU任务串行；超过20GiB峰值、空闲不足2GiB，或预热后近期步时中位数>8秒则停止。
超时smoke每组600秒、pilot3600秒、full14400秒；检查点每25步保存，smoke每步。
最坏50烟测步+4500pilot步+30000完整步=34550步，53次训练调用、69个唯一评估目录。
按已有约1.7至2秒/步估算，最坏纯训练约16至19小时，另加评估/保存/审计。
这不是耗时保证；翻转/分位数会增加预处理开销。GPU烟测预计约10至20分钟，仅为估计。
本机只验证CPU单测和PowerShell队列，远程真实GPU烟测通过后才会启动完整队列。

六 运行
更新仓库后，直接执行下面一行；它自带烟测，成功后自动继续到38例报告。
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-input-search-v2\run_experiment.ps1" -Stage all

仅查看计划：同一命令改为 -Stage plan。
只做烟测：改为 -Stage smoke。中断后原命令重跑，不删除旧结果。
不要与其他GPU队列并行。使用短且独立OutputRoot，最多32字符。

七 输出
C:\zhongliu\tis-v2\queue.log
C:\zhongliu\tis-v2\manifest.json
C:\zhongliu\tis-v2\pilot-selection.json
C:\zhongliu\tis-v2\slot-selection.json
C:\zhongliu\tis-v2\selection.json
C:\zhongliu\tis-v2\reports\test\test-38\summary.csv
C:\zhongliu\tis-v2\reports\test\test-38\paired-deltas.csv
C:\zhongliu\tis-v2\reports\test\test-38\paired-cases.csv
验证结果位于reports\pilot、confirm、combine各自的validation-10目录。
各阶段label-runtime.csv和训练paired-steps.jsonl用于诊断速度、标签变化及回退。
本轮新control是上轮验证集选中的on3_s0；旧结果只提供历史背景，不代替本轮配对对照。
10例验证和38例公开测试已被反复使用；N=3是训练种子，不是病例样本量，不作盲测或p值声明。
