# 记忆特征→轨迹→掩膜：配对诊断

## 目的与固定条件

前一轮 `memory_pointwise_fusion` / `memory_current_retrieval` 多次触发，但最终指标变化很小。本轮不新增优化、不训练、不挑选权重，定位变化发生在哪一层。默认在 **38例公开测试集**上运行；可单独选择10例验证集，两者输出目录隔离，不混合统计。这是已查看过测试结果后的机制诊断，不是新的盲测。

所有组使用相同原始 `scaled_offline.pth`、同一病例、同一初始查询点和随机种子0。控制组为 `hierarchical_full_grid0_iterations2_memory_topk_diverse` 的等价别名 `memory_control`。固定跨度10、迭代2、支持网格0和现有记忆设置，不改变推理超参数。

| 执行顺序 | 配置 | 用途 |
|---|---|---|
| 1 | memory_control | 对照 |
| 2 | memory_control_repeat | 相同对照重复推理，检测运行噪声 |
| 3 | memory_pointwise_fusion | 仅逐点记忆融合，与对照逐病例配对 |
| 4 | memory_current_retrieval | 仅当前外观检索，与对照逐病例配对 |

38例共152次病例推理，不包含重新训练。只读取视频、首帧标注及元数据，不读取后续帧真值标签，也不重新计算DSC等任务评估指标。

## 记录内容

| 阶段 | 记录与比较 | 范围/单位 |
|---|---|---|
| 分段查询 | 分段初始查询位置变化 | 全点，384×512模型坐标系像素 |
| 记忆特征 | memory_track / memory_support 的L2与余弦距离 | 固定采样，不是全量特征统计 |
| 当前查询特征 | current_track / current_support | 同上，识别查询位置变化的影响 |
| 实际输入特征 | input_track / input_support：记忆混合及归一化后的实际特征 | 同上；第一段无记忆时为原始查询特征 |
| 局部轨迹 | 双锚点融合前的轨迹差异 | 全点，均值、P95、最大位移 |
| 全局轨迹 | 对应全局分支切片差异 | 全点，应检查是否为0 |
| 双锚点融合 | 实际局部分支归一化权重及其变化；融合后轨迹差异 | 权重无量纲，位移为模型像素 |
| 最终轨迹 | 拼接、平滑及首帧校正后的轨迹差异 | 全部非首帧、全点；另记超过0.01/0.1/1像素的比例 |
| 整数轮廓 | 与fillPoly一致的向零截断整数坐标 | 坐标变化点数/比例；不把截断误称为四舍五入 |
| 模型分辨率掩膜 | 栅格化后的二值差异 | 变化像素数、帧数、占全图/两掩膜并集的比例 |
| 原始分辨率掩膜 | 恢复原图大小后的二值差异 | 同上，最终输出 |

特征固定取最多64个均匀间隔点的完整通道，记录4层特征金字塔；support特征取49个邻域位置中的首、中、末三个位置。采样ID保存在文件中并做一致性检查。特征L2无像素单位，不可和轨迹位移直接计算“衰减百分比”。

记录全部分段调用，包括遮挡合并过程中最后未采用的尝试。按起止帧及同跨度调用次数匹配；未匹配分段单列，不强行相减。因此分段统计可能重复覆盖帧，**最终轨迹/最终掩膜统计才是完整序列的结果**。分段当前查询可能已不同，特征比较可能包含位置变化的影响，不能仅凭这些观测证明因果。

## 一键运行（远程Windows PowerShell）

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"
git -C $Repo pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$Repo\scripts\run_memory_chain_diagnostics.ps1" -RepoRoot $Repo
```

默认结果目录：`C:\zhongliu\zhongliu-tuning\protocol-40-10-38\memory-chain-diagnostics\test-38`。

如需先在验证集单独运行，增加 `-Split validation-10`；该组结果只用于诊断，不能填入38例测试集表现表。`-Dataset`、`-ResultsRoot`、`-PreviousResults` 可显式指定绝对路径。启动时会检查同一父目录下既有40/10/38三个数据集的病例数与ID互斥性；仅枚举其他split目录，不读取其图像或真值。

脚本不请求关闭杀软、不修改系统执行策略、不生成临时可执行PowerShell脚本、不删除旧实验文件。若组织安全策略阻止运行，请保留拦截记录，不绕过策略。

## 实时日志与断电恢复

```powershell
$Diag = "$Repo\protocol-40-10-38\memory-chain-diagnostics\test-38"
Get-Content -LiteralPath "$Diag\runner.log" -Tail 40 -Wait
```

启动窗口保持运行。断电重启后，先确认Docker Desktop/GPU正常，再运行完全相同的命令。每个病例的每个配置独立提交；`trace.npz`、`trace-meta.json`、`run.json` SHA256全部匹配才跳过。中断的单次推理从病例开头重跑，不是逐帧续算。残留未完成目录保存在`.attempts`供排查，不会当成成功结果。

输出目录独占锁防止重复启动。冻结清单包含镜像ID、输入数据/权重哈希、采样设置及可用旧诊断文件哈希；改变条件必须使用新 `-ResultsRoot`，不会把旧结果混入。如果旧Docker容器仍在运行，先检查 `docker ps -a --filter name=trackrad-memory-chain-diagnostics`，不要并行重复启动或自动删除。

诊断含CPU复制及压缩落盘开销，`seconds_with_observation`不可用来比较正常推理速度。存储全轨迹和采样特征，需要额外磁盘空间，具体大小取决于病例长度；掩膜采用位压缩。保留完整轨迹以便后续复核。

## 输出与阅读顺序

1. `summary.json`：病例数、配对检查、旧输出核对覆盖数、未匹配段数。`pair_checks_passed=true`不代表缺失的旧输出也已核验。
2. `chain-cases.csv`：逐病例的对照重复一致性、旧输出哈希核对结果；空白表示没有旧诊断文件，不是“通过”。
3. `chain-overview.csv`：按配置和阶段汇总，均值按实际观测数加权；`worst_record_p95_not_pooled`是各段/病例P95中的最大值，**不是全体样本P95**。
4. `chain-stages.csv`：逐病例、逐分段、逐金字塔层的详细均值/P95/最大值。
5. `unmatched-segments.csv` 与病例 `comparison.json`：不可直接配对的段/特征缺失信息。
6. `jobs/病例/配置/`：原始观测数组、元数据、日志、运行信息及完成哈希。

对照重复输出或旧输出核对失败时，仍保存报告，但队列以失败退出，先排查再解释优化差异。旧结果默认从 `protocol-40-10-38/memory-refinement/对应split` 读取，只读不修改。

判断示例：

- 记忆特征已经几乎不变：检查候选记忆是否高度相似，而非继续以“触发次数多”推断有效。
- 记忆特征变化而input变化小：当前查询混合/归一化可能削弱变化，需要结合分段查询差异判断。
- input变化而局部轨迹小：跟踪器对该特征扰动不敏感。
- 局部变化、融合后更小：结合实际局部权重判断全局分支的影响。
- 浮点轨迹变、整数轮廓不变：像素栅格化未保留亚像素扰动。
- 整数轮廓变、模型掩膜不变：轮廓点冗余或填充结果相同。
- 模型掩膜变、原尺寸掩膜变化进一步减少：缩放与二值化可能消除部分变化。

这些是诊断线索，不预设哪一个是根因，也不把非零变化自动视为性能提升。
