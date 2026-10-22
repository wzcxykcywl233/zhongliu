教师联动标签验证 v1（2026-10-22）

问题：tc-v1中降权/融合已发生，但学生没有稳定收益。先检验规则是否把错误识别对、
融合是否朝正确方向移动。本包不训练学生、不改教师权重、不更换现有主线、不读取38例影像。
所有结果都是教师标签诊断，不是学生DSC、官方指标或方案采用结果。

一、固定的两组验证

1. 已知形变（train-40，40例x3种运动=120项）
每例首帧真实MRI纹理，生成60帧视频。128个查询点中64个来自首帧标注轮廓，64个为内部网格。
三个固定条件：静止；线性平移到(12,-8)像素；平移同时旋转到12度、缩放到1.05。
每帧直接由首帧逆映射双线性采样，不反复插值。查询点真值用正向矩阵解析计算。
只评分解析真值在图内且距边缘至少1像素的位置。无额外人工遮挡或成像噪声。
这些是真实MRI纹理上的已知图像形变，不等价于真实MRI组织运动，不能据此声称临床点位准确率。

2. 真实边界（validation-10，10例x1项）
每例取前min(60,T)个连续帧，不重复补帧，不均匀抽帧，不按结果选片段。
128个查询点都来自首帧轮廓。全部教师预测完成后才读取后续标注。
评分为预测点到当帧标注边界的无符号距离；优先staple_labels，否则labels，记录使用文件。
空/全前景标注或越界点记为不可评分，不填0、不静默删掉，保留Scored/Unscored。
边界距离不能辨别沿边界滑动、错误但恰好落在边界的点，不是真实点对应GT。

两组均沿用训练数据读取器的坐标轴、整卷1/99强度归一化和384x512尺寸，不改原读取器。
轮廓查询在最大连通前景的边界像素中按固定索引均匀抽取，不是原部署的样条弧长采样。
若边界像素少于目标点数，就取全部唯一边界像素并记录实际数量，不用重复点凑数。
不足4个边界像素则停止。查询排序不用于重建掩膜。
这不是部署流程的官方掩膜评价。所有查询时刻为0，和训练中任意时刻的查询分布不同。
本轮不能直接复原历史405873个冲突点：旧日志保存的是摘要/哈希而非完整坐标数组。
因此新增少量教师推理，保留所有输出坐标与参考，之后可只读分析，不再重复推理。

二、比较对象

原封不动复用tc-v1教师池：v2支持5x5、on3支持0、scaled离线支持100，全部4次迭代。
每项仅运行三次教师推理，同一输出轮流视为三位主教师的意见。
支持点使用tc-v1确定性审核抽样器。这是固定标签探针，不声称逐步复现原训练随机支持点。
按2/4像素阈值，检查eligible、conflict、nonconflict、shuffled、agreement、fused_changed。
shuffled与conflict逐帧在同一可审核集合内匹配数量，使用原固定随机种子规则。
fuse的计算与原规则相同：一致区域向两名审核老师均值移动25%，修正上限0.5像素。
每项在CPU调用原联动可见坐标loss钩子，核对统计、权重和融合目标，不运行学生模型。
所有评分参考只进入评分，不进入教师推理、审核规则或融合。

三、看哪些结果

synthetic-summary.csv：解析点位误差EPE，单位为384x512训练图像像素，越小越好。
real-summary.csv：边界距离，单位同上，不能与EPE合并。
mechanism-summary.csv：24行重点对照（3种已知运动+自然视频，各3位主教师x2阈值）。
  MacroConflictMinusShuffledPrimaryError >0说明冲突位置主教师偏差比随机位置大；
  MacroReviewMinusPrimaryOnConflict <0说明冲突位置多数意见更接近参考；
  MacroFusedMinusPrimaryOnChanged <0说明实际融合位置改善。真实视频仅为边界代理。
  首项只用conflict/shuffled都有可评分数据的同一批病例，列出可配对病例数及双方漏评数量。
case-diagnostics.csv：逐病例、主教师、阈值、队列的分解。
jobs/.../outputs.npz：三教师P/V/C、queries，合成项另含truth和变换矩阵。
jobs/.../proof.json：实际支持点/教师配置/标签哈希、耗时、显存、GT文件。
summary.json：完整性、局限和报告哈希。

每张表保留Selected/Scored/Unscored。无样本的均值留空，不当成0。
Macro_*为每病例先求均值再等权平均，Micro_*按可评分点帧数加权。
同一原图衍生的三个条件、重复帧和网格/轮廓点都不当成独立患者。不计算p值或虚假置信区间。
每个运动条件单独汇总；all和contour/grid是重叠视图，不可相加。
PrimaryError：主教师误差。ReviewerMeanError：两位审核教师均值的误差。
ReviewerMinusPrimary <0：多数意见更接近参考；>0：反而更远。
BothReviewersBetterFraction：两位审核教师各自都比主教师好的比例。
AllThreeErrorAbove2Fraction：三位教师都偏离参考超过2像素的比例。
FusedMinusPrimary <0：融合改善，>0：融合损害。
对真实边界表，这些只能解释为更靠近/远离边界，不可解释为点对应更准确。
所有同一行比较严格使用共同可评分位置。Threshold=4的融合是诊断反事实，未曾训练fuse4。

判断顺序：
  conflict对比同帧等数量shuffled：是否确实集中于更大主教师误差？
  conflict的ReviewerMinusPrimary及BothReviewersBetterFraction：审核老师究竟是否更可信？
  agreement及fused_changed：是否多数意见一致但一起错？小幅融合是否反而增加误差？
  按三教师和40/10病例分别看方向；不能用 pooled 点帧大样本数替代病例一致性。
本轮没有自动选优，也不会因合成探针变好自动启动训练。

四、远程执行

保留ts-v1冻结清单、tc-v1完整结果和本地固定训练镜像；Docker Desktop需运行。
复用本地镜像，网络关闭，不构建、不下载，不修改旧结果。输入目录只读挂载。
默认新输出 C:\zhongliu\trd-v1。先更新仓库，再单独执行下一行：

git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-review-diagnostics-v1\run_diagnosis.ps1" -Stage all

all先烟测：首个训练病例3个形变+首个验证病例，共4项12次教师推理。
烟测通过后剩余126项继续，已有4项复用；总130项390次教师推理，不是390次训练。
单独烟测用-Stage smoke；之后-Stage full也会自动生成报告。默认-Stage plan只核对清单。
每完成一项打印一行，完整错误保存在各stage.log。中断后重复同一命令，校验已完成项后续跑。
缓存哈希错误时停止，不删除/覆盖已完成结果。源码/输入变化时使用新输出目录。
烟测进程超时20分钟，其余进程2小时；GPU峰值reserved超过20GiB则停止。
不承诺未经实测的总耗时，不要同时跑其他GPU实验。CUDA失败不得视为阴性实验结果。
完成后先发summary.json、mechanism-summary.csv；必要时再看两个完整summary及逐病例表。
不需上传大模型或整段日志。

五、本地测试
python diagnostics/teacher-review-diagnostics-v1/test_diagnosis.py -v
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File diagnostics/teacher-review-diagnostics-v1/test_queue.ps1 -TemporaryRoot D:\yijian\tmp
CPU单元测试/队列静态检查不代替远程GPU烟测。
