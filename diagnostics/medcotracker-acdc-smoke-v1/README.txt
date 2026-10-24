MedCoTracker ACDC 外部阶段 GPU 烟测
=================================

前提：medcotracker-acdc-ab-v1 的 native-v2 数据审计已完成。
2026-10-24 用户回传：100患者，80外部训练/20开发；最短patient090为12帧。
8例不足16帧，7例在训练/1例在开发。本包不丢弃这些有效原生cine。

本包只做什么
------------
1. 从完整清单自动确定训练集中最短cine，必须同时覆盖全清单最短长度。
   三类冻结教师分别处理最短cine和正常16帧片段，共6次接口检查。
   数据读取器不补帧；在线教师原版内部滑窗padding保持不变。
   不能把“数据层不补帧”误称为“教师内部完全不补帧”。
2. 使用同一批真实片段、查询点和缓存伪标签，A1/B1/B2/B3各5步。
   输入384x512、128查询、4次迭代；非随机图像、非ED/ES两帧拼视频。
   前三步覆盖最短长度下的3类教师，后两步覆盖16帧。
   固定最短/最长接口覆盖是烟测抽样，不是未来pilot患者均衡采样。
3. 三类教师都不更新参数，保持当前v2/on3_s0/offscaled教师池。
   不重新调教师、不加联动。记录教师输出/输入哈希，所有路线逐步配对。
4. A1只更新compact Mamba；B1只更新fnet；B2只更新updateformer；B3全主干。
   B路线外部阶段不挂会detach主干的支线。
   全部state（含buffer）检查：冻结者逐字节不变、允许更新者确有变化。
5. 保存外部checkpoint，严格重载，权重逐字节一致，复测推理。
   A1保留已训练支线；B路线冻结适配主干并挂相同种子的新支线。
   检查交接后梯度只流向支线；这只是同一ACDC片段的交接图检查，没有优化器更新。

明确不做
--------
不读取TrackRAD病例，不做TrackRAD微调，不读ACDC开发图像，不启动pilot或正式训练。
不输出DSC/HD95，更不能把5步loss或教师置信度称为真实医学跟踪精度。
这只覆盖原A/B计划的“外部阶段GPU烟测+交接检查”；完整5步TrackRAD烟测仍待接入。
后续需要完成TrackRAD交接烟测，才运行seed0的500外部/250TrackRAD pilot。
正式3种子3000/1000须额外确认，本脚本没有all/自动续跑大型队列选项。
输入数据目录只读挂载；只打开外部训练80例的cine验证哈希/采样。

loss与训练口径
--------------
复用仓库sequence_loss：可见点Huber(delta=6)*0.05 + 不可见点L1*0.01，
迭代权重gamma=0.8，样本内有效点帧归一化，V>.9；batch=1。
按既有冻结主线默认，不开启独立confidence/visibility BCE或mask loss。
“hard confidence target”只有显式开启confidence监督才参与loss，本烟测未开启。
B2允许坐标/V/C头训练不等于每一个头都有独立真值监督，记录实际梯度路径。
AdamW weight_decay1e-5，梯度裁剪10，外部LR A1=5e-5/B1=B2=1e-6/B3=5e-7。
5步烟测固定LR，无scheduler；不是长期训练的学习率预算。
数据/查询seed=20261023；支线初始化seed=20261008，与当前主线seed0约定一致。
B路线交接的新支线必须同时与A1预训练前的支线初始化逐字节一致。

运行与重试
----------
先更新远程仓库。Docker Desktop运行且GPU空闲后：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\medcotracker-acdc-smoke-v1\run_smoke.ps1"

复用C:\zhongliu\ts-v1\local-runtime-images.json中的Training镜像哈希。
默认数据C:\zhongliu\acdc-official-v1、审计C:\zhongliu\acdc-ab-native-v2。
权重C:\zhongliu\zhongliu-tuning\training-checkpoints中的cotracker2v1/baseline_online/scaled_offline.pth。
新输出C:\zhongliu\acdc-ab-gpu-smoke-v1，不覆盖准备结果。
无联网、无image build/pull、无自动下阶段；单次容器最长30分钟。
单步超过120秒或预留显存超过22GiB停止；CUDA OOM直接失败，不静默降低分辨率/点数。
不用同时运行其他GPU训练；显存占用和时间不能据此估算完整队列总时长。

中断时保留attempts。重复同一命令会复用校验通过的完整路线；未完成路线在新的
attempt目录从官方初始化重跑5步，不冒充逐步续训、不把两个尝试的日志拼起来。
修改源代码、镜像、权重或清单须新OutputRoot。数据和权重不要进Git。
smoke checkpoint仅供烟测，不作为pilot起点；pilot须按同一seed重新初始化。

完成后只需回传：
Get-Content -LiteralPath "C:\zhongliu\acdc-ab-gpu-smoke-v1\summary.json" -Raw

本地测试
--------
python -m unittest discover -s diagnostics/medcotracker-acdc-smoke-v1 -p test_smoke.py -v
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File diagnostics/medcotracker-acdc-smoke-v1/test_runner.ps1
这些是CPU代码/模拟启动测试，不能替代远程真实权重GPU烟测。
