# 动态查询锚点与 Transformer 特征支线：两条独立实验

固定40例训练 / 10例验证 / 38例公开测试。正式性能表只取 **test-38 官方 metrics.json**；验证表不能替代测试表。公开测试集已被多次查看，不是新的盲测。两条方向先独立验证，不交叉组合，不恢复软标签、辅助教师或掩膜损失。

## 1. 动态锚点的思考与设计

之前的跨度、记忆和V/C继承调整仍然会把整个段的末帧坐标作为下一次查询。记忆库拒绝写入低分特征，不等于拒绝低分坐标。这次检验：V或C明显下降时，退回下降前的查询帧和位置，能否减少错误参考点引起的后续漂移。

每个点独立判断，使用**融合前局部分支**的V/C，避免整帧平均和双分支融合掩盖局部下降。绝对阈值0.5；相对阈值为低于之前健康帧EMA基线0.15，EMA系数1/3。连续两帧满足才确认下降，回退到第一次下降前一帧。末尾只有一帧下降时也不晋升可疑末帧；中途单帧下降随后恢复则不回退。不读取未来片段或标签，但当前片段本身是离线推理，不是严格因果在线模型。

例如下一次名义目标为第20帧、某点第7帧开始持续下降，就用第6帧位置/特征重新查询到第20帧；不是把第6帧位置复制成第20帧预测。不同点使用不同查询帧，同一次联合调用仍保留全部1000点。各点特征从其实际查询帧重新采样，继续混合原始查询外观；**不继承V/C初值**。

混合查询帧不能假装属于同一个记忆库时间戳。因此所有匹配组关闭遮挡合段、把历史特征库固定在初始条目，不再新增条目。完整历史方法作为额外reference复跑；回溯的独立效应必须对比匹配的fixed组，不能只对比历史方法。若第一阶段有效，才进一步与持续更新的多锚点记忆结合。

| 配置 | 内容 |
|---|---|
| anchor_reference | 当前历史多锚点方法，保留合段及原来的记忆写入 |
| anchor_fixed | 直接对照：跨度10、末帧查询、关闭合段、固定初始特征库 |
| anchor_diagnostic | 与fixed一样预测，仅统计下降，逐例输出数组须完全一致 |
| anchor_endpoint | 下降时保留上一次参考查询点，不选中间帧 |
| anchor_rollback | 选择持续下降前最后一帧，最大实际跨度30 |
| anchor_rollback_sensitive | 相对下降阈值0.08，其余同rollback |
| anchor_rollback_long | 最大实际跨度60，其余同rollback |

输出游标每次前进10帧，已提交旧帧不重写。过旧查询以原始全局分支当前坐标作为明确标记的恢复查询，该段此点仍采用全局结果；可靠性未恢复时不晋升局部恢复查询。联合片段最多31/61帧，避免无限回溯、失控显存和原地循环。审计记录下降、回退、全局回退、改变的病例数与诊断组输出一致性；零触发不能解释成有效机制。V/C尚未校准，下降只能作为待验证的证据。

## 2. Transformer 特征支线的思考与设计

刚完成的扩容实验中，普通微调、时间加深、时空加深都低于原始预训练权重。因此不再同时改动已有能力，而是冻结原模型，检验新增容量能否辅助原Transformer。

支线位于UpdateFormer最终隐藏表示与**坐标头**之间：

`H_new = stopgrad(H_transformer) + 0.1 * sigmoid(g(H_side)) * W_out(H_side)`

不是过去的六维坐标统计或17维整帧均值。输入为每点每帧的完整Transformer输入（多尺度局部相关性嵌入、位置/运动及V/C输入和时间编码）与最终384维隐藏特征。实际维度取 `input_transform.in_features`。支线3层、宽度256；每点独立处理，不在不同点、迭代或病例之间串状态。

旧CNN、相关性MLP、Transformer、坐标头及V/C头全部冻结，训练后旧张量SHA256必须逐字节一致。新增支线只修正进入坐标头的表示，V/C头直接使用原隐藏表示。修正后的坐标仍会改变下一次迭代的相关性采样，所以不声称最后V/C值永远不变。

只将最后W_out零初始化，sigmoid门保持非零，保证初始输出不变且首步存在梯度。冻结特征路径detach，训练只优化支线。

线性坐标头使用等价形式 `W(H+残差)+b = WH+b+W残差`，保留原计算路径。部分CPU算子仅改变requires_grad标志就可能产生约1e-6的数值差异，因此本研究的pretrained对照也固定为冻结标志、权重逐字节不变；这不是微调。初始化审计分别记录冻结标志转换差异，并严格要求挂接支线相对冻结对照完全一致，不能混用历史未冻结计算标志作为零初始化证明。

### 初始化检查修正（initfix-v2）

远程环境记录到冻结转换差异约 `[5.72e-5, 4.23e-6, 1.13e-6]`，旧代码把其中的V/C与本地设定的1e-6阈值比较，导致优化前误拦截。这不是支线权重学习造成的误差。修正后，该转换差异只记录诊断，不使用随机器变化的阈值；仍拒绝非有限值/形状变化，严格验证旧权重哈希不变，并要求挂接零初始化支线相对**冻结预训练对照**的P/V/C输出完全一致，连1e-7的支线差异也不能通过。

旧v1结果、镜像与日志保留。拉取修复后使用新目录启动，不删除frozen-run.json绕过来源校验：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\repair_feature_sidecar_initialization.ps1" -Stage smoke
```

检查通过后同一入口依次执行 `-Stage validation`、`-Stage test`。正式目录改为 `protocol-40-10-38\feature-sidecar-initfix-v2`，小规模目录为其名称加 `-smoke`。此入口只运行支线研究，不触碰动态锚点结果；不要用旧组合入口重新运行已经冻结的旧目录。

### 首批训练模式修正（modefix-v3，当前入口默认）

远程初始化通过后，第一批数据触发了旧训练器的 `assert model.training`。支线研究刻意使用“主干eval、支线train”，故不能沿用全模型train断言，也不能为通过检查而把冻结主干全部切回train。修正后，首批及每次评估结束恢复训练时统一设置混合模式；每批检查仅支线处于train且可训练，所有旧模块处于eval且参数冻结。原全模型训练实验继续使用原train检查。

入口仍为 `repair_feature_sidecar_initialization.ps1`，默认正式目录更新到 `protocol-40-10-38\feature-sidecar-modefix-v3`，小规模目录为其名称加 `-smoke`；v1/v2目录均保留不动。已增加四种支线真实offline `is_train=True` 首批前向、反向、优化步骤、冻结权重一致性与评估后模式恢复测试。本地CPU检查不等于远程Fabric/NCCL GPU验证，请仍先运行smoke。

### 正式推理导入修正（inferencefix-v4，仅评估恢复）

12组训练完成后，正式验证的第一组MLP报 `No module named ...feature_sidecar`。复用镜像的已安装CoTracker仍是旧版本；训练和旧GPU smoke显式把仓库ext源码放在导入路径最前，而真正的 `inference.py → model.py` 入口没有这样做。因此旧smoke通过不能证明正式入口正确。

现在推理镜像显式设置本地源码PYTHONPATH，model入口也在导入resources之前选择仓库源码。若进程此前已经载入其他来源的CoTracker，则直接拒绝，不能清空模块后混用旧新类。修复不修改主干/支线结构、权重、训练器、推理参数或数值公式。

**已经完成modefix-v3正式训练的电脑，不要重新执行训练入口，也不要删除旧指纹。** 新入口仅运行评估：

```powershell
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\repair_feature_sidecar_inference.ps1" -Stage validation
```

保留原 `feature-sidecar-modefix-v3/train`；原目录只读挂载，不复制、改写或重训12组权重。严格验证源码转换仅包含已审核的导入路径修改和审计输出重定向，训练/模型/数据/配方不得变化；重新计算12组完整配对、冻结张量及最终权重审计，并核验原预训练对照。新结果根为 `feature-sidecar-inferencefix-v4`。原基线验证也重跑，**不复用旧正式入口产生的预测**。

先用真正的model入口对四种已训练支线做GPU推理预检，再评估13组（冻结预训练对照+四支线×三种子）10例验证。评估过程仍按病例断点提交；重启后重复相同命令恢复。完成后显式启动38例测试：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\repair_feature_sidecar_inference.ps1" -Stage test
```

测试要求新目录的验证完成且指标哈希未改变。输出为 `feature-sidecar-inferencefix-v4/test-38/feature-sidecar-summary.csv`、`feature-sidecar-paired-deltas.csv`、`feature-sidecar-paired-cases.csv`。恢复入口不含训练阶段；CPU审计不训练、不调用优化器。本地已测试真实model入口在旧包优先时仍选新源码、拒绝已经载入旧包、严格源码转换及模拟队列的只读/数据/验证顺序保护；真实CUDA预检仍由远程电脑执行。

| 支线 | 总参数 | 新增可训练参数 | 对照目的 |
|---|---:|---:|---|
| pretrained | 25,385,700 | 0 | 当前历史方法直接对照 |
| mlp | 28,630,161 | 3,244,461 | 额外非线性容量 |
| conv | 28,636,305 | 3,250,605 | 有限邻域时序，3层5帧卷积约13帧感受野 |
| mamba_short | 28,763,025 | 3,377,325 | 每10帧重置支线状态 |
| mamba | 28,763,025 | 3,377,325 | 每次调用内完整时序选择性记忆 |

MLP/卷积与Mamba参数预算相差约4%，不声称完全同参数。长短Mamba结构、参数量、初始权重及训练输入一致，只改变10帧分块重置。

Mamba采用输入依赖Δ/B/C、负对角A、深度卷积和门控的双向共享权重选择性扫描。为复用本机环境、不下载新依赖，本版明确标为 **Mamba-style PyTorch参考实现**，不是官方mamba-ssm加速包或Mamba2。实现设计参考[官方Mamba-1模块](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)。本轮回答准确率问题，不能据参考扫描耗时评价官方内核速度。

### 训练与长时序边界

每组种子0/1/2、各1000步，固定60帧片段、128查询、4次迭代、AdamW学习率5e-5。比历史10帧拉长，但还不是在完整长MRI视频上训练。不足60帧的视频沿用加载器反向补齐机制，时间采样也沿用配对加载器。

同一种子组间teacher/case/query/video逐步一致，不同种子可不同。仍按原配方每batch随机一个主教师，辅助教师权重0，不改成共识监督，不加掩膜损失或软标签。沿用坐标蒸馏和硬可见性筛选，**不新增置信度训练损失**；配置的hard标识不意味着这次训练了置信度头。

测试全局分支在一次完整视频调用内保持长记忆，局部分支只处理实际局部片段，不跨局部调用继承状态。长于60帧的推理存在长度外推风险；伪标签偏差也可能使支线没有收益。

报告逐种子及逐例差值：各支线vs pretrained、conv vs MLP、长Mamba vs MLP/conv/短Mamba。诊断记录执行参数量、调用次数、最大调用帧数和隐藏残差幅度，排除模块未执行/未学习。不用测试标签训练或修正位置。

## 3. 远程运行与恢复

先打开Docker Desktop。默认复用本机 `trackrad-cotracker-backbone-growth:latest` 训练环境和 `trackrad-algorithm-cotracker-algorithm:latest` 推理环境，只复制当前代码，不联网安装Pixi或拉镜像，不改执行策略。没有前者可显式传 `-TrainingRuntimeImage trackrad-cotracker-gt-mask-training:latest`；之后固定实际镜像ID，不默默替换。

```powershell
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_anchor_and_sidecar_queues.ps1" -Stage smoke
```

smoke在独立目录做四组5步真实GPU训练、配对审计与训练权重GPU推理重载；不占正式目录，不产正式性能表。通过后依次运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_anchor_and_sidecar_queues.ps1" -Stage validation
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_anchor_and_sidecar_queues.ps1" -Stage test
```

validation先跑动态锚点10例，再做四种支线×三种子的训练与验证。test不训练，在验证完成后分别评估38例。两个队列顺序占用一张GPU。单独跑一条使用 `run_adaptive_anchor_resumable.ps1` 或 `run_feature_sidecar_resumable.ps1`，参数为RepoRoot、ResultsRoot、Stage；支线还支持Stage train。

每25步及每轮末原子保存，保留最近3个训练检查点；恢复含模型、优化器、调度器、教师随机序列、随机状态与epoch/batch位置。重启后重复原命令，完成组跳过，重复步骤输入不一致会拒绝；不要删审计日志绕过检查。推理每例原子提交，未提交病例重算。源码、权重、数据或配方改变后必须新建ResultsRoot。终端实时显示日志，各组保留training.log，总队列保留queue.log。

正式结果文件（均在protocol-40-10-38下）：

- `adaptive-anchor-v1/test-38/adaptive-anchor-test-38-results.csv`
- `adaptive-anchor-v1/test-38/adaptive-anchor-test-38-mechanisms.csv`
- `adaptive-anchor-v1/test-38/anchor-audit.json`
- `feature-sidecar-v1/test-38/feature-sidecar-summary.csv`
- `feature-sidecar-v1/test-38/feature-sidecar-paired-deltas.csv`
- `feature-sidecar-v1/test-38/feature-sidecar-paired-cases.csv`

本地CPU检查仅确认代码、初始化、梯度、冻结、边界与脚本语法，不能替代远程CUDA smoke、训练及官方38例评估。
