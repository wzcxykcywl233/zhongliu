冻结教师 + 最佳原CoTracker学生：全量复训 v1

冻结依据：ts-v1 的125步单种子10例验证结果。新教师候选中iter6的学生DSC最高，
其次以HD95/CD/D98/名称打破平局；D98不是最优，且所有训练组均低于未训练参照。
用户已明确要求继续全量验证。本实验记录这项决定，不篡改旧门禁、旧报告或旧权重。

教师：冻结 scaled_offline.pth，额外支持点0，教师迭代6次；只替换三教师池的离线
槽位。CoTracker2/online3不变，采样比例不变，不新增教师微调，不做EMA联动。
“冻结教师方案”不等于使用本轮125步学生权重。所有学生均重新从scaled_offline初始化。

学生：原CoTracker3离线主干25,385,700参数，无Mamba/侧支；部署配置rt_points_0，
与pd_sparse_native完全相同，即250点 + hierarchical_full_grid0_iterations2_memory_topk_diverse。
250是推理轮廓点数；训练保持10帧/384个随机时刻查询/4次学生迭代/学习率5e-5，
原vis_conf_head冻结、硬标签、无软置信度/辅助教师损失/掩膜监督。不是同时调参。
这对应此前38例DSC约0.881245的原CoTracker方案，不把教师生成器成绩当成学生成绩。

组别：old与iter6各3种子、每组1000步，共6次完整训练，另有未训练参照N=1。
同种子同病例/视频/查询/支持点随机抽样配对；未修改的两位教师原始标签必须一致。
不把125步权重续接到1000步，不复用训练结果，仅复用已验证的原不可变运行时镜像。

数据：原40训练/10验证/38测试，所有病例互斥。仅40例视频用于训练，不读38例监督。
教师选型只用已经完成的10例验证pilot。新学生统一使用最佳250点配置评价：
old/iter6/pretrained都重评，不能将新250点结果直接减旧1000点结果。
1000步完成后先评10例，再固定两组全部3种子报告38例，即使退化也如实报告。
无测试集筛选；38例是已反复观察的公开测试集，不能宣称新盲测或显著性。

远程执行，命令单行，不需要再跑旧pilot、不需要旧-AllowDegradedPilot：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\frozen-teacher-full-v1\run_full.ps1" -Stage plan
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\frozen-teacher-full-v1\run_full.ps1" -Stage all

all会完成冻结校验、6次训练、10例评价、38例评价。也可逐段-Stage train/validation/test。
默认新目录C:\zhongliu\ts-full-v1；默认输入C:\zhongliu\ts-v1。
旧ts-v1只读，旧数据和源码逐文件哈希校验；本包放diagnostics，未改动旧冻结源码。
不build、不pull、不安装依赖、不改执行策略、不要求管理员；本机原镜像须仍可用。
断电后重跑同一个all命令，完成臂先校验再跳过，未完成臂恢复模型/优化器/调度器/
教师采样器/数据位置；发现不一致则保留现场并停止。原每臂7200秒与每步中位15秒
安全预算沿用，单臂结束即检查。不通过修改预算或丢弃不利种子拼接结果。

核心输出：
selection.json：教师权重SHA、迭代/支持点、学生完整配置、验证依据、继续全量决定。
training-audit.json / training-runtime.csv：全量配对审计及按教师槽位的耗时。
validation-10/ 与 test-38/ 各含：
 frozen-teacher-student-results.csv：每个种子的官方学生指标。
 frozen-teacher-student-summary.csv：DSC/HD95/MASD/CD/D98/TimeSec的均值、样本SD。
 frozen-teacher-student-paired-deltas.csv / paired-summary.csv：同种子新教师对旧教师，
 以及新旧教师训练组分别对未训练参照；同时回答“改善训练”和“超过原有方法”。
 frozen-teacher-student-paired-cases.csv：逐种子逐病例差值。
 audit.json：官方指标与实际输出文件SHA；不把预测重算成另一套正式指标。

计时的StepSeconds不含数据加载和保存权重，TimeSec来自官方评价器，均非总墙钟耗时。
本实验验证全量训练效果，不保证改善，不能将短试验的排名视为三种子/38例结论。
