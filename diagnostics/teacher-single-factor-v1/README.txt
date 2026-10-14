教师单因素渐进实验 v1（2026-10-14）

目标与边界
本轮先改“生成伪标签的教师”，不训练学生、不更新教师权重、不挂 Mamba、
不使用软标签。用现有 CoTracker3 offline 架构，比较 baseline_offline.pth 与
scaled_offline.pth，然后在教师推理上复用此前的改进。不是把之前训练后退化的
大模型直接当强教师，也不是重新做辅助教师权重/重采样实验。
先只研究离线教师。将来进入学生配对训练时，先只替换教师池中的offline一项，
保留CoTracker2/online教师及其抽样序列，再验证是否解除当前监督限制。

每个子配置只覆盖一个字段，完整配置自动从直接父对照继承。32 个配置不是
32 种“已证明的改进”；预先登记的实验树即使某个中间步骤失败也不会自动改写。
主线用相同目标点身份，相同图像、重建、模型架构、随机支持点和评价器。
固定输入为 T,H,W（MHA 原数组为 W,H,T），默认原始 MRI 强度。
这不是复现旧训练中 T,W,H 的预处理；norm 只改变强度，不同时改变轴顺序。

分组（新配置数量；运行时自动补齐并复用父对照）
weights    2  old -> scaled：只替换权重，offline/window60/迭代4均不变。
core       5  支持点策略 uniform100 -> none 或 grid10；4->2/6次迭代；1%-99%强度归一化。
hierarchy  4  分层跨度10；原查询特征权重0.5；全局/局部分支融合0.5；遮挡合并。
memory     4  latest容量2 -> 容量4 -> topk -> topk+多样性。
state      5  独立继承V、C、VC、衰减VC、仅查询帧VC，均对照 diverse。
span       2  固定跨度10分别改为5、15。
context    1  固定250个目标点，额外750边界上下文点；只返回原250个目标点。
anchor     3  独立动态锚点支路：实现对照fixed -> endpoint/rollback。
refinement 6  recent_slot、逐点写入、逐点融合、当前帧检索、循环写入、轮廓保护。

重要的单因素处理
1. 原离线教师的100个随机时空支持点与grid10均为100点，支持策略单独测试。
   本轮随机支持固定种子；它不严格复现历史教师随机流。
2. memory_mode=none 时API slots派生为0，capacity字段仍保留休眠预设2；
   启用latest只启用该模块，不同时由2变4。容量扩大另列latest4实验。
3. adaptive_anchor支路从dual05分叉（无遮挡合并、无VC继承）。先运行独立
   fixed实现对照，再只改anchor模式。不能拿rollback与同时启用merge的配置
   比较后声称差异都来自动态跨度。回退阈值0.15，连续2帧，最大年龄30。
4. 固定250个t=0目标边界点，所有配置共用同一批点；重建不根据V/C删点，
   也不改变后处理。额外上下文点在跟踪后剔除，不进入重建。
5. 原特征/全局融合两个权重均0.5；memory接纳可靠性/相似度阈值0.5、
   原锚点权重下限0.3、多样性权重0.25；状态衰减时间常数1。均不在本轮调参。

为什么先做教师质量评估
当前学生从scaled预训练权重出发，旧教师却使用baseline权重，有必要先确认
换权重及已有推理改进是否能提供更可靠的监督。掩膜DSC上升不等于点对应更准，
更不等于训练学生后的DSC一定上升。所以同步保存目标点轨迹P、可见度V、置信度C、
相对父对照的位置改变量、越界比例、V>0.9覆盖率、轨迹二阶差分和机制计数。
高V/C与更平滑轨迹也不自动代表正确。本轮不利用未来真值选择/修正轨迹。

查询适配范围（不能跳过）
这是第一帧边界查询的教师标签生成器试验，不是完整训练接口。
旧训练允许任意查询帧；现有hierarchical函数内部会将查询时间置0。
本包在调用前拒绝非零目标查询时间，绝不静默错误对齐。
只有发现有价值的教师配置之后，才设计任意时刻查询的匹配适配对照，再做
“学生结构/初始化/输入序列固定，只换教师”的配对训练。
在线教师及CoTracker2暂不纳入：切换模型族同时改变架构，会破坏本轮单因素解释。

数据与统计
冻结原40/10/38协议的全部文件哈希和88个互斥case ID；不从测试集训练。
10例只用于教师选择，绝不混入38例正式表。指标来自冻结的TrackRAD评价器，
DSC/D98越高越好，HD95/MASD/CD越低越好。实际病例JSON元数据用于评价，不写死。
CSV名和Split列明确区分validation-10与test-38，ReportKind标明teacher-label-generator。
TimeSec为官方job时间总和（预处理/加载完成之后的跟踪+重建时间），并另外记录
纯跟踪秒数；不是教师训练耗时。
公开38例已经反复查看过，不能宣称盲测。病例/帧也未证明相互独立，不提供p值。

选择规则
先完整运行自己指定组及其父对照，再显式select同一Group。
按validation-10 DSC选最多3个高于old的候选，平局依HD95/CD/D98/名称；
始终测试old/scaled，并补入候选的直接父对照，总计最多8个测试配置。
没有候选改善时仍测试old/scaled，不偷偷扩大搜索范围。此规则只按DSC排序，
不宣称全指标胜出，D98下降必须保留。select不读取测试表现。
selection.json冻结后禁止再往同一目录添加验证配置或改候选。
因此若计划跑多组，必须都跑完才select；-Group all要求32项验证都完成。

远程运行（每条命令独立复制，先启动Docker Desktop）
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-single-factor-v1\run_teacher_experiments.ps1" -Stage plan

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-single-factor-v1\run_teacher_experiments.ps1" -Stage smoke

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-single-factor-v1\run_teacher_experiments.ps1" -Stage validation -Group weights

先看weights组CSV决定是否继续，不自动启动所有32项。后续把Group依次改为core、
hierarchy、memory、state、span、context、anchor、refinement；父对照哈希通过则跳过。
有明确需要也可以-Stage validation -Group all，但仍不自动启动test或学生训练。

全部预定组完成后，下面示例选择all组并冻结候选，再显式测试38例：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-single-factor-v1\run_teacher_experiments.ps1" -Stage select -Group all

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-single-factor-v1\run_teacher_experiments.ps1" -Stage test

如只跑weights即可select -Group weights。不可select未完成的组。

默认输入
ProtocolRoot: 仓库/protocol-40-10-38/feature-sidecar-modefix-v3
SourceStudy: 仓库/protocol-40-10-38/sidecar-stability-v5/degradation
Weights: 仓库/training-checkpoints，验证baseline_offline.pth和scaled_offline.pth
数据：trackrad2025_labeled_train_40 / labeled_validation_10 / labeled_public_test_38
可用同名参数传实际路径；必须与冻结协议内容一致。
Inference/Evaluation镜像默认读取SourceStudy/local-runtime-images.json。
只使用已安装sha256镜像，无构建、下载、联网；缺镜像时停止，不改tag。
可显式传-InferenceImage/-EvaluationImage，仍需通过冻结数值源检查。

输出与恢复
默认C:\zhongliu\teacher-single-factor-v1-results（独立短路径，不覆盖旧结果）。
manifest.json冻结包、源、镜像、权重与数据；plan.json显示每一条父子改动。
smoke/summary.json记录old/scaled/diverse/rollback的小规模23帧32点检查，
跨过两个span10边界，避免只检查初始段而没有调用后续记忆读取。
validation-10/<profile>/jobs/<case>保存tracks.npz、diagnostics.json、prediction.json、MHA、verified.json。
teacher-validation-10-weights-results.csv 等是教师验证表，不应填入学生38例表。
teacher-test-38-results.csv / deltas.csv / paired-cases.csv / label-diagnostics.csv是教师38例表。
异常保留.attempts、唯一launch日志；重跑相同命令逐case恢复，不删除结果。
每个容器默认2小时超时，内外双重限制；仅停止本次创建的唯一容器。
新代码/数据/权重/镜像改变时用新OutputRoot，禁止混用缓存。

本地测试
python -m unittest discover -s diagnostics/teacher-single-factor-v1 -p "test_*.py" -v
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File diagnostics/teacher-single-factor-v1/test_launcher.ps1
需要torch/numpy/SimpleITK/scipy/skimage/cv2；不下载依赖、不用GPU。
GPU数值与显存检查由远程smoke完成，本地CPU测试不冒充GPU验证。
