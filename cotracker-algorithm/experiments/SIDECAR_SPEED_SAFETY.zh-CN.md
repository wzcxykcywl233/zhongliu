# 支线训练加速、退化控制与诊断

## 为什么先解决时间

四种支线相对原预训练权重在38例上平均有收益，但长Mamba对MLP的DSC仅高约0.000024。长短Mamba差值较一致、幅度仍小，不能支撑又花数天扩大网络。当前实现逐帧执行状态递推，训练每次有3层、双向扫描、4次迭代。先测执行瓶颈与显存，不靠缩短60帧、减少128查询或省略教师来获得混淆的速度提升。

## 1. 执行加速，不改变模型与监督

状态递推为 `h[t]=a[t]*h[t-1]+b[t]`。并行版本利用仿射变换的结合律做前缀扫描，依赖轮数从T降至约log2(T)，但算术工作增加到O(T log T)。它不是官方mamba-ssm优化CUDA内核，不能保证GPU一定更快。保留原逐帧后端，默认仍为reference；参数名、数量、状态生命周期、双向共享权重和MLP等均不变。

并行前向使用FP32状态；反向按 `lambda[t]=g[t]+a[t+1]*lambda[t+1]` 计算一阶梯度，节省逐帧自动微分图。不支持二阶梯度。避免累计乘积除法，没有以裁剪状态/Δ换取“等价”的数值近似。浮点求和次序改变，必须比较输出及所有参数梯度，不能承诺位级相同。

点维按16/32分块，不缩短时间、切断记忆或跨点共享状态。训练可重计算支线块以降低激活显存；不含随机层，检查重计算前后梯度一致。损失、教师采样、数据序列、初始化、训练步数及学习率不变。不引入AMP、TF32、软标签、辅助教师或新训练损失。

设计参照[官方选择性扫描参考递推](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)；一阶自定义反向与gradcheck遵循[PyTorch扩展说明](https://docs.pytorch.org/docs/stable/notes/extending.html)。

## 2. 先微基准，再真实小规模训练

profile读取旧seed_0已训练的MLP、短Mamba与长Mamba，只在GPU内存中的副本上做微基准，不保存覆盖旧权重。输入为合成的完整rich/hidden表示，60帧×128点，**4次调用放在同一反向图**，测前向、反向、AdamW、总耗时和峰值分配显存。预热1步、测3步，每个后端从相同支线张量起步。GPU上先核对17帧非整块形状的输出和全部参数梯度。

它不包含教师、CNN/Transformer、数据读取、审计日志、保存检查点，不能把其倍数当作整轮训练加速倍数。OOM明确记录；默认并行版本微基准达到2倍才允许下一阶段，不静默回退。未达标时请提供speed-profile.json再定位，而不是直接重训。

16/32点分块各测重计算开/关，共四种执行设置；不预设开启重计算一定更快。它以额外计算换显存，若关闭时能容纳完整四次迭代图且更快，则选关闭版本。微基准显存不包含真实教师等开销，仍须通过随后真实smoke，不能据此保证完整训练不OOM。

smoke再执行四种支线各5步**真实40例训练加载器/教师/完整CoTracker/反向/检查点**，核验配对、冻结权重和GPU推理。独立目录；不能替代长训练效果验证。日志时间包含保存开销，不把5步冒烟当成稳定吞吐基准。正式1000步新训练未自动启动。

## 3. 不重训的退化控制

复用旧12组权重，仅评估MLP和长Mamba、三个种子。仍使用 `hierarchical_full_grid0_iterations2_memory_topk_diverse`，不变查询数、迭代次数、记忆或掩膜重建参数。

| 配置后缀 | 支线修正倍数 | 每次迭代修正上限 | 要回答的问题 |
|---|---:|---:|---|
| full | 1.00 | 无 | 当前完整支线对照 |
| gain050 | 0.50 | 无 | 温和修正是否减少退化 |
| gain025 | 0.25 | 无 | 更接近预训练能力是否稳健 |
| gain050_cap05 | 0.50 | 0.5输入像素 | 限制少数大修正是否有益 |

上限作用于新增支线坐标头贡献，**不是限制整个CoTracker运动量**。坐标头输出处于stride网格坐标，按model.stride换算到模型输入图像像素后限幅，再换回网格；不是原MRI像素或毫米。不是依据测试病例、前缀、标签或坏帧阈值选择支线。

先跑10例验证：冻结预训练对照、8配置×3种子、seed_0逐帧扫描影子对照，共26组。影子对照检验并行后端完整推理的浮点影响：平均DSC差绝对值≤1e-5、最大逐例≤1e-4、最大轨迹差≤0.001输入像素。未通过不进入测试。

每种支线用验证集三种子平均选择同一设置：DSC不低于冻结预训练对照，D98不低于对照减0.001，再按DSC、HD95、CD、D98排序。若没有符合者，停止扩大测试，不声称修正已经成功。候选冻结后显式运行38例，保留两种full对照。公开38例已多次查看，不是全新盲测；正式表现只报其metrics.json。

## 4. 退化诊断

每例保存实际runtime、调用次数、原始/施加坐标修正均值和最大值、限幅比例；另存最终轨迹、V/C和查询缓存。对冻结预训练输出逐例配对：修正幅度→轨迹平均/最大偏移→掩膜改变体素数→正式DSC/HD95/MASD/CD/D98差值。轨迹变化或掩膜变化本身不等于改善。帧数用于后续分析长序列是否真正受益，不能把同一例的帧当独立病例。

诊断覆盖全部病例，不根据已查看测试结果挑选病例获得偏置收益。A_015、C_017、B_016等只能作为事后失败分析例子，不作为运行时规则。固定三种子/全病例报告，避免只展示最有利模型。

另输出分组诊断表：先对每例的三个种子取平均，再汇总all及A/B/C/X的正式指标差、改善/退化例数、修正/轨迹/掩膜变化。不把38例×3种子计为114个独立病例，也不把分组表现用于测试集特定门控。

## 5. 远程入口

旧feature-sidecar-modefix-v3训练与feature-sidecar-inferencefix-v4报告不动。当前新源码不属于旧import-only恢复白名单，**不要重跑旧repair_feature_sidecar_inference.ps1，更不能删除其指纹**。新根为 `protocol-40-10-38\sidecar-speed-safety-loadfix-v2`。旧训练只读挂载，校验完整配对、权重、原训练器以及40/10/38数据文件哈希。复用原镜像环境，不联网安装新依赖。

loadfix-v2修复训练环境复用镜像漏复制resources及根目录Python模块的问题：旧加载器不认识feature_sidecar元数据，会直接以无支线模型加载包含支线的权重。现在同时复制当前加载器及依赖；测速前验证其实际导入路径和仓库SHA256，依然严格加载所有参数，不用strict=False或删掉支线键。旧sidecar-speed-safety-v1目录保留，不删除指纹、不混用已固定的旧镜像；当前默认新根重新复制代码，**不重新训练**。

```powershell
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_sidecar_speed_safety_resumable.ps1" -Stage profile
```

先查看speed-profile.json，达到门槛后依次运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_sidecar_speed_safety_resumable.ps1" -Stage smoke
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_sidecar_speed_safety_resumable.ps1" -Stage validation
```

validation不训练，直接复用旧权重，耗时主要来自26组10例推理。查看selection.json，通过后才运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_sidecar_speed_safety_resumable.ps1" -Stage test
```

输出包括speed-profile.json、validation-10/scan-backend-agreement.csv、各split的sidecar-safety-summary.csv和sidecar-safety-case-diagnostics.csv。源码/协议/镜像或数据变化需另建目录；推理按病例原子提交，相同命令断点续跑。

若微基准和真实smoke证明值得重新训练，可另开新根，显式传本研究冻结的fast-training-runtime.json给run_feature_sidecar_resumable.ps1的SidecarRuntimeConfig。该训练入口只允许gain=1、无坐标限幅，避免把执行加速与训练目标变化混为一个实验。先根据实测再决定，不在本队列启动新的1000步。

本地只有CPU。已检查递推与解析梯度、gradcheck、非2次幂长度、极小衰减、独立点、块重计算梯度、真实模型兼容/零倍数/像素上限及选择与缓存配对。实际GPU速度、峰值显存和正式效果必须由远程运行确认。
