# CoTracker 主干加深实验

本轮直接增加CoTracker3轨迹更新网络的独立可学习层。继承已有图像编码器、相关性MLP和原三层UpdateFormer权重，固定推理端配置；参数量来自新注意力/MLP层，不是增加同一组权重的迭代次数。

## 结构与思路

| 训练组 | 时间层数 | 空间交互组数 | 总参数量 | 新增参数量 |
| --- | --- | --- | --- | --- |
| base | 3 | 3 | 25.385700M | 0 |
| time6 | 6 | 3 | 30.704484M | 5.318784M |
| space_time6 | 6 | 6 | 46.665444M | 21.279744M |

time6检验更多时间处理层是否能更充分利用相关性和相邻帧信息；space_time6进一步检验更深的点间/虚拟轨迹交互是否能改善轮廓点的协同运动。额外的空间组包含虚拟轨迹自注意力以及两个方向的交叉注意力，因此参数增加较多。time6与base、space_time6与base为主比较；space_time6与time6也可用于观察在相同时间深度上增加空间交互的作用。

旧代码按时间/空间层数比值均匀安排空间交互。如果直接把时间层数由3改为6，原空间层会从0/1/2移到0/2/4。本轮显式保留原时间/空间三组在0/1/2的执行顺序；time6只在后面接时间层，space_time6在后面接对应时间和空间层。

新层复制原末层的内部权重，但把注意力输出投影和MLP最后一层投影初始化为零。这使新增残差层初始为恒等映射，不共享参数，随后可以独立学习。每个训练启动都用真实CoTracker前向、图像特征和两次更新检查位置/可见性/置信度与原模型完全一致；初始化报告记录差值和参数量。小尺寸CPU检查不是MRI性能评估。

## 配对训练

默认种子0/1/2，每个种子三组各1000步，共9个训练运行、9000个优化步。使用40例训练、10例验证、38例测试。训练片段长度10、384个查询轨迹、每次前向4次更新、AdamW、学习率5e-5、batch size 1、单张GPU。三个结构使用相同教师池，每batch只采样一个主教师；辅助教师权重为0，没有新增掩膜loss或软标签。保留现有训练脚本的vis_conf_head冻结策略，其余主干和新增层参与优化。

每个种子内共享病例顺序、片段采样种子、逐步随机种子及主教师采样种子。每成功完成一步，追加并同步落盘独立JSON记录：教师索引、病例、查询点哈希、视频片段哈希。训练结束必须有100%步骤覆盖且三个结构逐步一致。断点恢复产生的重复记录也必须有相同输入签名。普通终端日志只用于查看进度，不再用于重建步骤审计。

验证和测试均固定使用 `hierarchical_full_grid0_iterations2_memory_topk_diverse`（1000轮廓点、跨度10、更新2次、grid0及四槽记忆），避免同时更改点数/重建。另评估原始未微调权重pretrained：扩容与同条件微调base比较，训练后模型还要与pretrained比较，以免把微调导致的退化隐藏起来。全部三组结构事先固定；本轮不按38例结果改模型或筛选某个种子。

## 运行入口

先等当前旧队列完成，再拉取更新；源码指纹变化会阻止旧队列原地恢复。本轮结果使用独立目录 `protocol-40-10-38\backbone-growth-v1`。

先做GPU冒烟检查（种子0，三组各5步，输出到独立的backbone-growth-v1-smoke）：

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"
git -C $Repo pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$Repo\scripts\run_backbone_growth_resumable.ps1" -Stage smoke
```

冒烟通过后，执行三种子训练、配对审计和10例验证：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_backbone_growth_resumable.ps1" -Stage validation
```

验证完成后单独启动38例测试：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_backbone_growth_resumable.ps1" -Stage test
```

`-Stage train`可只完成训练及审计。想先只跑种子0，可在同一个新ResultsRoot的所有阶段显式传 `-Seeds "0"`，但这是探索性单种子结果。NumSteps、Seeds或其他冻结配置变更时必须指定新的ResultsRoot，不能混用旧断点。1000步是首轮预算，不代表足够训练到收敛；新增容量是否真正有用，需根据验证曲线和结果判断。本轮也不宣称10帧训练证明了完整长序列能力。

## 权重、恢复与日志

### GitHub不可达时复用已有训练环境

若构建停在下载Pixi，而本机已有之前训练成功的 `trackrad-cotracker-gt-mask-training:latest` 镜像，可给所有阶段指定 `-TrainingRuntimeImage`。脚本先检查本地镜像、固定其ID，再只复制当前仓库的训练代码和实验工具；不执行Pixi安装或依赖解析，也不使用旧实验的模型权重或训练配置。导入实际训练入口后记录Python/PyTorch/Lightning版本到runtime-check.json；GPU是否能训练由独立冒烟确认。

更新源码后必须使用新目录，不修改原backbone-growth-v1的冻结记录。以下每个阶段各是一整行命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_backbone_growth_resumable.ps1" -Stage smoke -TrainingRuntimeImage "trackrad-cotracker-gt-mask-training:latest" -ResultsRoot "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\backbone-growth-offline-v1"
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_backbone_growth_resumable.ps1" -Stage validation -TrainingRuntimeImage "trackrad-cotracker-gt-mask-training:latest" -ResultsRoot "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\backbone-growth-offline-v1"
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_backbone_growth_resumable.ps1" -Stage test -TrainingRuntimeImage "trackrad-cotracker-gt-mask-training:latest" -ResultsRoot "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\backbone-growth-offline-v1"
```

该选项只避免训练环境安装，首次验证仍需构建推理/评价镜像。若指定镜像已清理，脚本会明确停止，需先查看docker image ls选择实际存在且有training环境的镜像，不会静默下载替代品。各组必须使用同一固定环境；复用本身不改变三组架构、40/10/38数据协议或训练超参数。

需要已有training-checkpoints下的scaled_offline.pth、baseline_online.pth、baseline_offline.pth、cotracker2v1.pth。首次构建后固定训练、推理和评价镜像ID；冻结源码、权重、数据、种子及训练参数。训练和验证使用同样的推理镜像，测试沿用验证镜像。

每25步以及每个epoch保存完整优化器、学习率调度器、随机状态、教师采样器和数据位置，保留最近3个断点。保存到临时文件并同步，再原子提交；若最新断点因断电不可读取，会尝试更旧的完整断点。所有断点损坏则停止。最终模型也原子保存，包含架构元数据；推理加载时重建对应结构并严格检查全部权重形状。审计还检查新增层的输出投影是否真正离开零初始化。

根目录queue.log记录总进度；train/seed_N/architecture/training.log为实时训练日志；paired-steps.jsonl为完整步骤证据。训练中断后启动Docker并重跑原命令即可续接。病例预测沿用已有原子提交和完成标记；更新仓库、改数据、替换权重或清理冻结镜像后不能继续混用原目录。

## 结果与判定

最终表格只取test-38目录，指标方向：DSC↑、D98↑，HD95↓、MASD↓、CD↓。

- backbone-growth-results.csv：每种子正式指标和未微调参照，保留TimeSec与参数量。
- backbone-growth-summary.csv：三种子均值及样本标准差；pretrained只评估一次，N=1。
- backbone-growth-paired-deltas.csv / paired-summary.csv：扩容减同种子base、训练后模型减pretrained；正DSC/D98及负距离指标为改善。
- backbone-growth-paired-cases.csv：逐病例差值，用于观察B/X等组及D98取舍。
- backbone-growth-audit.json：执行结构、病例完整性、输出文件、权重和正式指标文件校验。

若扩容只超过微调base但仍低于pretrained，应记录为缓解微调退化，不能叫超过现有方案。若三种子配对差值波动很大，则保留不确定性；新增参数本身不是成功标准。38例公开测试已经被反复查看，结果属于该公开集上的探索性评估，不是新的盲测。
