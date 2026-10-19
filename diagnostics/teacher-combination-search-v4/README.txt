教师内部组合 + 教师池联合组合（v4）
===================================

目的与边界
----------
主要终点是“在对应教师池指导下训练的学生”的官方指标，不是教师独立推理指标。
固定学生：官方 CoTracker3 主干冻结，只训练 compact Mamba 支线；总参数 28,763,025，
可训练参数 3,377,325。40例训练、10例选择、38例最终报告，三个集合病例编号互斥。
60帧、128目标查询、学生4次迭代、lr=0.00005、原损失和原教师抽样机制不变。
每步仍然抽一位教师，不把三位教师的标签平均；教师权重始终不更新。
支线每个实验从同一种子初始化重新训练。pilot不是full的续训。
当前control不是历史原始教师池：在线CoTracker3已经零支持点，离线已经scaled权重。

三个槽位（不是四位同时在场的教师）
--------------------------------
A / v2：CoTracker2.1，cotracker2v1.pth，5x5支持网格，4次迭代。
B / on3：在线CoTracker3，baseline_online.pth，0支持点，4次迭代。
C / offscaled：离线CoTracker3，scaled_offline.pth，100随机支持点，4次迭代。
本轮不同时搜索旧离线权重，也不改Mamba、主干、损失或推理后处理。

第一层：每位教师内部2^4全因子组合
--------------------------------
S：A支持点25->9，B支持点0->9，C支持点100->50。
I：教师迭代4->6，学生仍为4。
N：仅教师输入做整个片段的1%~99%分位数归一化，输出float32 [0,255]。
   确定性采样上限131072像素；平坦片段退回原输入。不是逐帧归一化。
E：原视图+水平翻转双视图，逆变换后对有效、未越界坐标做均值融合。
   先N后生成视图。融合不额外修改主视图V/C；查询帧和在线教师查询前时间不融合。
   保持原有xmean定义，不新增V/C门槛或视图差异门槛。无效位置退回主视图。
   主视图V/C不变仅指同一配置的融合步骤；S/I/N自身可能改变主视图V/C。
支持点和目标查询在所有视图中同步映射；原学生视频、目标查询和随机抽样不被修改。
所有方案维持原支持点随机数消耗，减少离线支持点用原100点的前50点。

每位教师16配置：原配置1、单因素4、双因素6、三因素4、四因素1。
另两位教师保持control。共享对照后共46个学生训练配置，33个为教师内部多项组合。
命名例：on3_sine表示同时启用S/I/N/E，offscaled_ie表示仅启用I/E。

第二层：允许单独不胜出的候选进入教师池组合
------------------------------------------
每个槽位保留两个不同候选：A=非原配置中学生DSC最高；B=至少改两因素且与A不同的
最高DSC配置。并列顺序：HD95升序、CD升序、D98降序、名称升序。
探索入选不要求单独超过control。最终替换主线仍受独立的adoption门槛约束。
三个槽位各有原配置/A/B三种选择，完整3^3=27组：1原配置、6单槽、12双槽、8三槽。
其中7组复用第一层，新增20组，短程阶段共66种不同教师池。
j_si_0_ine表示A启用SI、B保持control、C启用INE。0是槽位原配置而非历史旧教师。
这不是全部12个二元因素的4096种全穷举；局部筛选仍可能遗漏未入围设置之间的交互。

阶段与预算
----------
plan：验证原始源码、权重、数据、固定本地镜像，创建不可变manifest/计划。
smoke：48个教师设置（每槽16）GPU接口检查；control、三种单槽sine及三槽同时sine
       共5组各训练10步，验证
       主干逐值不变、初始支线一致、配对输入/标签、训练后学生GPU推理。
pilot：46组，seed0，每组250步，按1000步调度；学生validation-10评估，冻结局部候选。
joint：27组，seed0，每组250步；复用7组，新增20组；评估后选择full确认名单。
confirm：短程最好的三槽组合的全部8个子集 + 短程总体最优的全部子集 + 最好的单槽。
         去重后8~12组，种子0/1/2各1000步，从相同官方主干/配对支线初始化开始。
         三槽候选即使暂时不优于control也要跑，保留负结果及拆解对照。
         validation-10评估全部配置，冻结selection.json。
test：不训练，只测试confirm名单的全部配置，保留失败组合与消融；另含无支线参照。
all：自动顺序运行smoke -> pilot -> joint -> confirm -> test；任何失败立即停止。

短程16500步；full 24000~36000步；烟测50步，总计40550~52550步。
总训练任务95~107个（含烟测）。这是较大队列，先看烟测速度和磁盘空间；测试模拟通过
不等于远程CUDA烟测通过。每组只保留最近3个训练检查点，仍会占用较大磁盘空间。
同一OutputRoot内身份完全一致时复用结果。不同方案/版本/源码不混用缓存。

最终主线替换门槛（沿用前轮，非统计显著性）
----------------------------------------
只用validation-10三种子：DSC平均增益>0.00001；D98平均下降<=0.002；
HD95平均增加<=0.02；至少2/3种子DSC提升；最差种子DSC下降<=0.002。
不合格时selected_overall=control，但三槽组合和其对照仍出现在最终报告中。
test-38已经多次被查看，不作盲测/显著性声明，不按测试结果重新选种子或配置。
validation-10也经过多轮使用，有限搜索赢家不等于全局最优。

远程运行（先启动Docker Desktop）
--------------------------------
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-combination-search-v4\run_experiment.ps1" -Stage all

默认PriorStudy=C:\zhongliu\ts-v1，用它冻结的源码/训练权重/本地镜像/数据清单。
默认新结果目录C:\zhongliu\tcs-v4，不复用tes-v3/tis-v2/tps-v1中的训练结果。
也可先-Stage smoke，确认后再运行-Stage all；已通过的烟测会验证后跳过。
中断后重复相同命令和路径，不删除旧结果、不强行绕过manifest检查。
运行中不要git pull或启动第二个GPU队列。输出目录带排他锁。
只用本地固定镜像，无build/pull、容器网络关闭，输入只读；不修改系统执行策略。
训练保留GPU余量：分配/保留显存不超过20GiB、driver空闲>=2GiB；稳态中位步时>8秒
则停止保留结果，防止无意启动长期慢队列。每个任务有超时保护。
控制台仅打印阶段、每25步进度、摘要和错误尾部，完整日志保存在各任务文件夹。

结果读取与含义
--------------
Get-Content -LiteralPath "C:\zhongliu\tcs-v4\reports\test\test-38\summary.csv"
Get-Content -LiteralPath "C:\zhongliu\tcs-v4\reports\test\test-38\paired-deltas.csv"
Get-Content -LiteralPath "C:\zhongliu\tcs-v4\reports\test\test-38\interactions.csv"

summary/results/paired-deltas/paired-cases：学生官方指标，N为训练种子数，不是教师数。
interactions.csv：组合增益、各单槽增益之和、两者差、完整子集的交互对比。
  双槽交互=AB-A-B+control；三槽交互=ABC-AB-AC-BC+A+B+C-control。
  DSC/D98的正值、距离指标的负值表示有利方向；都是探索性差分，不给p值。
reports/<stage>/label-runtime.csv：教师标签变化相关计数/V/C/越界/融合/耗时诊断，
  不是教师独立DSC，也不能把置信度或视图一致性当作对应准确率。
budget-estimate.json：根据短烟测外推的训练时间粗略范围，不是上界，不含评估、
  数据加载、保存检查点和审计；具体病例和最终选中配置会影响实际耗时。
selection-diagnostics.csv：每项候选是否通过主线替换门槛、具体失败原因。
local-selection.json / confirmation-selection.json / selection.json：依次冻结候选、
  三种子确认名单、最终报告名单，全部绑定父报告哈希。
pretrained仍采用相同改进推理流程但没有支线，不是未经优化的最初原始基线。

本地检查（不运行真实训练）
------------------------
python diagnostics/teacher-combination-search-v4/test_study.py
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File diagnostics/teacher-combination-search-v4/test_queue.ps1 -TemporaryRoot D:\yijian\tmp
第二条为模拟Docker和数据的编排测试，TemporaryRoot需短且可写；不必在远程训练前执行。
