教师联动机制实验 v1（2026-10-21）

一、这次验证什么

上一轮是改变各教师各自的输入/推理配置，再交替教学生，不是同一步互相审核。
本轮固定现有control教师池及冻结主线，在同一视频、同一组查询上，让主教师提出
标签，另两位教师同时审阅。指标属于训练后的学生，不是教师单独的表现。
教师不更新参数，不进行教师间反向传播；联动发生在伪标签和坐标损失这一层。
不混入上一轮INE/SINE、多视图、强度归一化等教师改动，也不修改学生推理结构。

当前对照是on3_s0 + scaled教师池，不是最早的旧教师池：
  slot0 CoTracker2：cotracker2v1.pth；5x5支持点；4次迭代。
  slot1 CoTracker3 online：baseline_online.pth；无支持点；4次迭代。
  slot2 CoTracker3 offline：scaled_offline.pth；100个支持点；4次迭代。
每步仍按原随机序列挑一个主教师；联动组额外调用剩余两位。

二、六个正式组

control：原独立辅导，只使用主教师监督。
tutor010：可见坐标损失按主教师90%、两位审核教师各5%组合；无效审核教师的份额
          还给主教师。这是坐标损失辅导对照，不等同于历史上两次学生前向、混合
          全部损失的random-tutor。
gate2：两位审核教师彼此距离<=2像素，而主教师距二者都>2像素，该点帧的可见
       坐标监督权重从1降至0.25；不改主教师坐标、不删除点、不改变V/C监督。
gate4：同gate2，只将分歧阈值改为4像素，检验阈值敏感性。
shuffle2：计算gate2降权数量，但在每一帧相同的审核可用点集合内随机选择同等
          数量的点降权。数量、帧分布、权重总量均配对，区分审核判断与重加权效应。
fuse2：保留gate2；三者预测两两距离都<=2像素时，向两位审核教师的均值移动25%，
       位移范数上限0.5像素。不是把明显分歧的点直接拉到多数意见。

以上像素指训练裁剪/缩放后的图像坐标，不是毫米或原始MRI像素。
审核只作用于原本有效且主教师可见的坐标监督；每位审核教师还需坐标有限、在图内、
V>0.9，并严格位于查询时刻之后。查询帧及其之前不改。
两位审核教师有一位不可用时，gate/fuse回退原监督；tutor可用其中有效的一位。
不同教师的原始C不当作可比较概率，不平均V/C。
可见Huber损失用加权均值，保持原系数0.05，不声称梯度范数相等。
不可见坐标损失、可见度监督、confidence硬标签规则保持原样。
无GT掩膜参与审核/融合决策，GT只供原有官方评估使用。

三、烟测空操作对照

shadow：额外调用两位教师但不改监督，学生权重和每步loss必须与control逐值一致。
same_teacher：主教师输出模拟两个完全一致的审核意见，结果也必须逐值一致。
二者只用于烟测，不参加正式优选。CPU合成测试另外强制触发分歧、降权、融合和回退，
避免真实烟测恰好没触发机制而虚假通过。

四、学生和配对

冻结官方主干25,385,700参数，只训练compact Mamba支线3,377,325参数，总计28,763,025。
60帧、128查询、4次训练迭代、lr=5e-5、原1000步学习率计划，训练种子0/1/2。
每组独立从相同预训练主干和零输出支线开始，不在旧结果上续训。
每步学生前向只有一次；所有教师eval且requires_grad=False。
保留主教师支持点抽样与输入，额外审核支持点使用局部确定性随机数；额外调用前后
恢复Python/NumPy/Torch/CUDA随机状态。主教师P/V/C、病例、视频、查询和教师选择
必须跨组一致；审核教师标签也需在联动组之间一致。
断电重复step若输入/审核标签/监督变化会拒绝，不吞掉重复冲突。
日志auxiliary=none/alpha=0指禁用旧random-tutor路径，不代表本轮没联动；查看
paired-steps.jsonl中的label_audit.collaboration、teacher_calls。

五、自动流程和预算

plan：核对40/10/38互斥病例、源码、权重、固定本地镜像，冻结清单。
smoke：GPU三教师接口/损失梯度；8组x10步；主干冻结和空操作配对审计；所有学生
       GPU推理检查。任意失败停止，不进入完整队列。
pilot：6组x250步、seed0，仍按原1000步学习率计划，在validation-10评估。
       用于观察机制/耗时，不淘汰机制对照，六组全部进入三种子确认。
confirm：6组x3种子x1000步、从头训练，在validation-10评估并冻结优选。
test：不训练，已冻结18个学生和未训练参照在test-38报告全部结果。

总计32次训练、19,580次更新：正式18,000步、pilot1,500步、烟测80步。
共45个整集合评估目录：pilot7、confirm19、test19。
烟测后budget-estimate.json按实测单步耗时粗估训练时间，不含评估/加载/保存/审计。
不按旧队列耗时承诺本轮时间，不要并行启动其他GPU队列。
每次完成更新后检查：allocated/reserved峰值<=20GiB，剩余显存>=2GiB，最近5步
中位耗时<=8秒。单训练进程超时：smoke600秒、pilot3600秒、full14400秒。
守卫不能防止单步瞬间OOM，失败时保留证据。

采用规则仅用validation-10三种子：平均DSC增益>0.00001，D98下降<=0.002，HD95增加
<=0.02，至少2/3种子DSC提高，最差种子DSC下降<=0.002；不满足则保留control。
shuffle2不作为采用候选。六组预先锁定均报告38例，包含失败组，不按测试集改选。
联动是否有效还需检查gate2与shuffle2、tutor010的差异及D98/难例覆盖；仅胜过control
不能证明审核更准确。教师一致不等于GT，相关错误可能让多数教师一起错。
38例已反复观察，不声称盲测；种子SD不是独立病人置信区间。

六、远程运行

先启动Docker Desktop，保留C:\zhongliu\ts-v1原冻结清单与本地镜像/权重。
复用其固定源码、训练/推理/评估镜像，不构建、不下载、不访问容器网络。
不改历史实验包、主训练文件、教师权重和旧结果。

先单独执行更新，确认成功再运行下一条：
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

烟测通过后自动完成整个队列：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-collaboration-v1\run_experiment.ps1" -Stage all

仅烟测改为-Stage smoke，仅检查计划改为-Stage plan。
默认新目录C:\zhongliu\tc-v1；中断后重复原命令，校验已完成项后续跑。
输入/源码/镜像变化必须查明原因，不删除审计、不覆盖旧结果、不强行混用缓存。
依赖的旧诊断源文件也纳入哈希；改动后不能在原目录继续混跑。
同目录锁防止重复启动，但不能阻止其他目录的GPU队列。

关键结果：
reports\test\test-38\summary.csv            学生指标均值/种子SD
reports\test\test-38\paired-deltas.csv      对control/pretrained的逐种子差值
reports\test\test-38\paired-cases.csv       对control/pretrained的逐病例差值
reports\test\test-38\mechanism-deltas.csv   gate2-shuffle2、gate2-tutor010、gate4-gate2、fuse2-gate2
reports\test\test-38\mechanism-cases.csv    以上机制对照的逐病例差值
reports\confirm\label-runtime.csv          各组/种子/主教师槽的覆盖、分歧、降权、融合、耗时
selection.json                             仅validation-10冻结的采用候选
audits\smoke.json                          主干冻结与空操作学生逐值一致证明
budget-estimate.json                       烟测后的粗略训练预算

完成后先贴summary.csv、mechanism-deltas.csv、reports\confirm\label-runtime.csv。
控制台每25步显示进度，完整training.log留在各组目录，无需粘贴整段训练记录。

七、本地回归测试

python diagnostics/teacher-collaboration-v1/test_study.py -v
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File diagnostics/teacher-collaboration-v1/test_queue.ps1 -TemporaryRoot D:\yijian\tmp
后者模拟Docker和队列，不做真实训练；实际GPU验证仍需远程smoke通过。
