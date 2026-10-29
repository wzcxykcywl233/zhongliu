MedCoTracker 混合数据、区域弱监督与点区域双分支实验 v1
====================================================
本包是新实验，不修改历史 B1、exclude082、教师或冻结主线结果。
研究问题：医学掩膜提供区域归属/边界约束，能否在教师轨迹之外改善学生？
掩膜不提供同一物理点的跨帧身份，不能直接转成精确点对应真值。

1. 固定配置与四条路线
--------------------
所有组：official scaled_offline.pth 初始化同一个 CoTracker3 offline 主干，
主干参数完全冻结；附加 fresh compact Mamba（沿用现有实现）并训练。
不继承之前 ACDC B1 的 fnet 权重，不替换教师，不解冻 Transformer。

M0 混合数据对照：教师轨迹监督，无额外人工掩膜损失；输出点重建掩膜。
W1 区域弱监督：M0 + 0.05*边界距离损失；输出仍为点重建掩膜。
D1 点区域双分支：W1 + 区域头，增加 0.1*(BCE+soft Dice)。
   同一个 checkpoint 分开评测 points 与 region，禁止挑选较好的帧拼接。
G1 区域主导：D1 去掉教师坐标损失，保留区域损失和弱边界损失；
   仍用教师可见性筛选弱边界约束，仍需内部点轨迹提供区域位置先验。
   这是区域主导的过渡版，不宣称已取消点追踪器；不是历史 R1 对照。
   除6步接口烟测外，只有 pilot D1-region 相对 M0-points 过门槛才训练 G1。

区域头：冻结 fnet 特征降采样到48x64；融合首帧掩膜的特征原型相似度、
当前图像特征、与首帧的特征变化、点轨迹的密度图以及首帧掩膜的平移先验。
小卷积残差头的末层零初始化；预测区域 logits，阈值固定0（sigmoid=.5）。
区域头看不到当前/未来真实掩膜，也不把教师坐标直接当推理输入。
部署保留固定 hierarchical_full_grid0_iterations2_memory_topk_diverse 追踪；
region输出只替换点到掩膜的重建环节。首帧按原流程锁定，后续不喂GT。
训练查询128个首帧轮廓点；部署保持固定原profile点数，不另调后处理。

2. 数据与损失的含义
------------------
ACDC：原 external-train-80 排除 patient082（已观察到的时间跳变）和
patient076（与 external-dev-20 的 patient074 解码后完全重复），共78例。
20例开发集不训练、不用来动态筛选病例；原80/20清单和原始文件不改动。
不是认证其余数据“无质量问题”，未排除其他待复核项目。
每次只用一个切片、一个类别，按查询帧前景>=16像素选择；不依据目标帧质量挑样本。
取ED到ES或ES到ED的完整原生帧间隔，允许反向时间增强，不复制帧补长。
ACDC人工掩膜只在两个端点存在：首端点作查询提示，另一个端点作损失。
中间帧的零占位掩膜标为“未标注”，绝不当成背景，也不插值制造GT。

TrackRAD：只用既定40例训练；10例选方案；38例必须显式报告。
取原生最多60帧，不重复补齐，所有可用标注帧计算相应区域损失；
查询帧不计算区域/边界损失，避免靠复制查询掩膜得到低损失。
训练/测试病例ID和所有输入hash校验；测试影像在test门槛通过前不挂载。

每两步固定 TrackRAD、ACDC 各一步；每域按确定性病例轮换。
同种子所有实验组的图像、轮廓查询、教师槽位、轨迹和可见性逐步配对。
固定教师池：v2旧权重 support5/iter4；online3 support0/iter4；
offline3 scaled权重 support100/iter4。三个教师均冻结，无新增教师训练。
教师置信度只记录，不解释为概率/点对应准确度；弱监督只取visible>.9。
轨迹损失沿用 .05*visible Huber + .01*invisible L1，gamma=.8。
弱监督使用预测轮廓点到真实轮廓的无符号距离/8像素，越界附加同尺度距离罚；
8是固定模型像素尺度而非病例物理间距，避免除以大图像对角线后损失几乎为零。
不可见点、未标注帧、空目标轮廓跳过该项。它不强迫点按轮廓索引一一对应。
区域损失可处理空掩膜。边界损失本身无法确保点分布/身份正确，D1区域头
用于检验覆盖约束；仅凭最终Dice不能证实物理点追踪更准。
多连通结构/心肌环的查询采样沿用最长轮廓，可能遗漏内边界；不把它当完整区域GT。
steps.json记录三类损失、监督帧数、空目标、可见比例、点分布尺度与耗时/峰值显存。

3. 公平对照与限制
----------------
统一AdamW lr=5e-5、weight_decay=1e-5、梯度裁剪10、迭代4。
统一输入384x512；两域训练片段分别p1/p99缩放到0..255；推理保留历史固定
pipeline原始输入处理。训练TrackRAD坐标统一W,H,T -> T,H,W，与部署一致。
这与历史随机点/调度器/数据处理不同，所以M0是本轮必要基准，历史数值只作背景。
本轮可隔离“加入掩膜损失”和“区域头”的影响；没有同预算纯TrackRAD训练组，
因此M0相对历史R1的变化不能单独归因于混合医学数据。
pretrained是同一增强推理pipeline的无支线官方权重参照，不是未经优化的纯基线。
所有正式结果是学生表现，而非教师单独表现；N=3是训练种子数，不是病例数。
不使用测试表现改变门槛；public test-38已反复查看，不能称全新盲测。

4. 队列与门槛
--------------
plan：只校验数据/权重/代码/本地镜像。
smoke：seed0，M0/W1/D1/G1各6步，共24步；覆盖两个数据域*三个教师。
       检查梯度、主干不变、配对标签、保存重载；再用验证集前2例分别检查
       M0-points/D1-region完整部署与官方评测接口。通过前不启动pilot。
pilot：seed0，M0/W1/D1各250步（125+125），验证10例。
full：seed0/1/2，固定M0/W1/D1各2000步（TrackRAD1000+ACDC1000）；
      若pilot D1-region过门槛，再加G1相同预算。每阶段均从相同初始化重新开始。
      共享标签仅3种子*2000=6000条，full优化步18000或24000步。
      不复用pilot模型，也不通过延长某个优胜组训练制造不公平。
all：smoke -> pilot -> full -> validation-10，随后停止。
test：只在完整full验证结果冻结后，报告固定所有full实验组及输出视图。

pilot门槛：D1-region比M0-points的DSC>0，D98下降不超过.002，HD95增加不超过.02。
full资格：同样的三种子均值门槛，至少2/3种子DSC改善，最差种子DSC>=-.002。
按验证DSC排序合格输出；无合格者回退M0-points。W1/D1不中途按test挑优。
G1通过门槛时直接进入三种子full，不另加G1 pilot；门槛关闭时明确记为未开展。
三种子SD不等于置信区间；不做相关帧p值。逐种子、逐病例差值全部输出。

5. 远程运行
------------
默认不构建镜像、不pull镜像、容器无网络。使用ts-v1已固定的Training/Evaluation镜像。
请先git更新；保留acdc-ab-native-v2、原始ACDC、training-checkpoints和ts-v1清单。
默认新结果目录 C:\zhongliu\rs-v1，不覆盖mct-full-v1或mct-x82-v1。

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\region-supervision-v1\run_experiment.ps1" -Stage all

可先只跑 -Stage smoke 或 -Stage pilot；相同命令可续跑。
full验证完成、查阅 reports\full\validation-10\summary.csv 与selection.json后：

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\region-supervision-v1\run_experiment.ps1" -Stage test

完整执行时长以烟测为准：本轮full至少18000步，显著多于过去1000步单组，
不能用以往单个模型的时长估计整队列。每步180秒保护、单任务timeout保护；
异常即停止后续大训练，保留日志。避免与其他GPU训练并行导致误触发性能门槛。

6. 续跑、审计及交付测试
----------------------
optimizer+模型+已完成记录每25步同一原子checkpoint；烟测每步保存。
中断最多重放24步；确定性每步抽样/随机种子和共享伪标签避免重放错位。
GPU某些反向算子可有数值非确定性，不承诺中断前后模型逐位相同；输入与标签必须相同。
每组最终核对所有冻结主干tensor，并实际重载checkpoint做前向。
结果和代码有hash绑定；源代码/权重/数据改变必须使用新OutputRoot。
全目录.lock防并发；无自动删除、chmod/chown或host管理员操作。
评测仍用原官方镜像的默认用户/环境，写入容器tmpfs，按原始字节传回metrics.json；
不修改公式、不用错误Python替代官方环境。已有评测先校验收据再复用。

本地CPU检查（没有真实病例、没有本机GPU训练）：
python diagnostics/region-supervision-v1/test_region.py
powershell.exe -NoProfile -File diagnostics/region-supervision-v1/test_runner.ps1
本地验证包含合成原生稀疏标注、方向/损失/梯度/门槛/续跑收据/调用时序。
正式病例读取、官方镜像依赖、GPU显存与性能必须以远程smoke为准。
