冻结 Mamba 主线的教师多视图轨迹融合实验 v3
2026年10月18日

目的与假设
上一轮将原图直接替换为翻转图，部分候选验证集有收益，但未在38例测试中优于原对照。
本轮不重复替换实验，而保留原始教师预测，再加入翻转视图预测。
假设两路方向偏差可能互补；若分歧明显则回退原轨迹，以减少替换整个教师输出的风险。
这是假设，不是已经验证的收益。两路都错时也可能一致，平均也可能把正确轨迹拉偏。
教师仍是已训练好的冻结推理模型，不微调教师权重。

一 固定主线与参照
control 沿用 teacher-pool-search-v1 验证集选中的 on3_s0：
  槽0：CoTracker2.1，cotracker2v1.pth，25辅助点，4次迭代。
  槽1：在线CoTracker3，baseline_online.pth，0辅助点，4次迭代。
  槽2：离线CoTracker3，scaled_offline.pth，100辅助点，4次迭代。
oldpool 仅将control槽2换为baseline_offline.pth，保留槽1的0辅助点。
它不是更早在线25点版本的同名oldpool。
v2、on3、offold、offscaled是四个实验系列；两个离线系列竞争同一槽，不是同时四教师。

学生官方主干25,385,700参数完全冻结，仅训练现有Mamba支线3,377,325参数。
总计28,763,025参数。40例训练，60帧/片段，128目标查询，学生4次迭代，lr=5e-5。
compact运行，种子20261008+0/1/2；损失、教师抽样、数据增强、调度器均不改。
推理固定hierarchical_full_grid0_iterations2_memory_topk_diverse，1000点、2次迭代。
不是250点方案，也不把上轮学生权重接着训练。每组从同一官方权重及配对支线初始化开始。

二 每个教师系列的三项单独改动
xmean：原始图与左右翻转图各推理一次，映射回原坐标后对位置P等权平均。
  不设V/C或分歧门槛，是基础融合对照；仍要求各路数值有限、位置在图内。
xguard：与xmean相同，但每个点每帧都要求两路V>0.9、C>=0.5，距离<=2像素。
  不满足时保留原始P，不删除监督、不修改原始V/C。
xyguard：原始图、左右翻转图、上下翻转图共三路；每路满足质量检查，
  任意两路最大距离<=2像素时等权平均，否则保留原始P。

12项候选：
  v2_xmean          v2_xguard          v2_xyguard
  on3_xmean         on3_xguard         on3_xyguard
  offold_xmean      offold_xguard      offold_xyguard
  offscaled_xmean   offscaled_xguard   offscaled_xyguard
加control和oldpool，共14池。每项只改其直接参照的一个教师槽位的融合方式。
旧离线系列短程筛选相对oldpool，其余相对control；完整确认统一相对control。

三 坐标、标签与因果边界
整段视频与目标/辅助查询做同一翻转；x映射W-1-x，y映射H-1-y，不插值、不改变时刻/顺序。
不额外抽取随机辅助点。学生仍接收原视频、原查询，只有教师生成的位置监督可能变化。
V/C逐点逐帧始终沿用原始视图，CoTracker2.1的C沿用原协议的全1。
所有查询时刻保留原始P；在线教师仅在查询之后融合，离线教师可在查询前后融合。
2像素指训练输入坐标，不是毫米；阈值预先固定，不扫描、不按38例结果调整。
V/C与视图一致性都未校准，不能视为真实对应精度。三路一致也可能共同错误。
本轮不改损失权重、不加入逆时间追踪、不做位置后处理，也不改学生推理链路。

四 自动执行与筛选
smoke：真实GPU检查全部16个系列设置（4个默认+12个候选），验证坐标映射与原始V/C。
  再训练5组各10步，逐步配对审计并做训练后学生GPU推理。
pilot：14池，seed0各250步，用学生10例验证结果选每系列一个相对赢家。
confirm：control、oldpool与至多4个赢家，3种子各1000步，从头配对训练。
combine：对通过确认的不同槽位改动测试所有子集，至多8池，另保留oldpool。
  相同完整配置复用已审计结果；不同离线权重不可同时占同一槽位。
test：先冻结验证集名单，再对参照、合格单项、最佳合格组合与整体赢家评估38例。
  没有合格组合不强行拼接；没有合格候选则保留control。test阶段本身不会训练。
all：自动执行上述各阶段，任一烟测/审计/运行检查失败即停止。

门槛沿用上一轮，不因本轮结果放宽：
  平均DSC增加>0.00001；平均D98下降<=0.002；平均HD95增加<=0.02。
  三种子确认至少2个DSC提升，任何一个种子DSC下降不超过0.002。
  合格者按DSC、HD95、CD、D98、名称排序。均值略升不代表满足种子一致性。
所有候选保留结果；新增selection-diagnostics.csv列出每项门槛的淘汰原因。
只确认每系列一个短程赢家，因此可能漏掉慢收敛方案，并非穷举全部最优。
N=3是训练种子，不是病例数。10例验证和38例公开测试都已反复使用，不作盲测或p值声明。

五 审计、速度及恢复
沿用C:\zhongliu\ts-v1的冻结清单和本地精确镜像ID；新结果C:\zhongliu\tes-v3。
不修改已有算法/训练文件，不构建、不拉取、不联网，不混用上一轮缓存。
逐步核对病例、视频、查询、教师抽样及辅助点随机数；未改槽位标签必须一致。
原视图P/V/C、各视图映射后的P/V/C、输入查询及视频抽样均记录哈希。
视频抽样哈希最多65536个值，是辅助机制证据；原训练视频另有完整哈希。
旧离线候选也与本轮oldpool核对原视图标签，不能只与scaled对照比较。
主干权重必须完全不变，支线必须更新；来源/输出变化则停止并保留证据。

label-runtime.csv记录教师调用数、融合接受/改变/回退点帧数、修正像素及耗时。
MeanCorrectionPixels是每步合格时间区间均值的步间平均，不能当作已修改点的平均修正量。
TeacherSeconds包含视图处理、映射、融合及视图哈希开销，不是纯GPU核计时。
新增教师调用只发生在训练伪标签生成时；学生推理不调用这些教师。
所有GPU任务串行。峰值分配/保留>20GiB、显存空闲<2GiB或近期步时中位数>8秒则停止。
每组超时：smoke600秒、pilot3600秒、full14400秒；完整训练每25步保存检查点。
最坏50烟测步+3500pilot步+30000完整步=33550步，49次训练调用、65个唯一评估目录。
实际预算随筛选结果减少。多视图会增加训练耗时，应以本次smoke/pilot日志估计，不能照搬上轮步速。
本机只能验证CPU和模拟队列，真实GPU兼容性与耗时由远程烟测确认。

六 运行
先更新仓库，再运行单行命令；自带烟测，通过后自动继续完整队列：
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-ensemble-search-v3\run_experiment.ps1" -Stage all

仅看计划改为-Stage plan；仅烟测改为-Stage smoke。不要与其他GPU队列并行。
中断后原命令续跑；不要删结果、改冻结清单或复用tis-v2/tps-v1目录。
自定义OutputRoot必须独立且不超过32字符。首次默认根目录为C:\zhongliu\tes-v3。

七 结果入口
C:\zhongliu\tes-v3\queue.log
C:\zhongliu\tes-v3\pilot-selection.json
C:\zhongliu\tes-v3\slot-selection.json
C:\zhongliu\tes-v3\selection.json
C:\zhongliu\tes-v3\reports\test\test-38\summary.csv
C:\zhongliu\tes-v3\reports\test\test-38\paired-deltas.csv
C:\zhongliu\tes-v3\reports\test\test-38\paired-cases.csv
reports\pilot、confirm、combine下有selection-diagnostics.csv及validation-10结果。
各阶段label-runtime.csv与逐组paired-steps.jsonl用于判断是否真正改动标签、是否耗时过大。
