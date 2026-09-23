# 主线与最优候选：误差定位诊断

## 固定对象

- 主线：`pvc_control`，即 `hierarchical_full_grid0_iterations2_memory_topk_diverse`。
- 候选：`pvc_vc_decay`。必须仅在 `query_state_inheritance` 上不同。
- 默认复用 `query-state-inheritance/test-38` 的两组已完成结果。主成绩严格来自38例 `metrics.json`；验证10例需单独运行、单独目录。
- 不训练、不重新推理、不修改权重或掩膜、不构建镜像、不联网。CPU运行冻结的原镜像，只读挂载预测和数据。

## 五项诊断及决策用途

| 编号 | 问题／预期 | 实验 | 结果文件 | 如何指导改进 |
|---|---|---|---|---|
| D1 | 衰减继承究竟帮了多少病例？ | 五项官方指标逐病例配对；病例bootstrap 10000次；逐一删病例敏感性分析 | official-case-pairs.csv、paired-summary.csv | 收益若依赖单病例，不继续将其宣传为稳定改进；优先查看退化病例 |
| D2 | 后期漂移还是分级边界问题？ | 去掉查询帧，按相对时间四分位、模10偏移、区域、磁场分组，先算病例内均值再跨病例平均 | diagnostic-group-summary.csv、diagnostic-case-groups.csv | 若后期差，则优先考虑记忆有效期；若边界差，则进一步检查更新策略 |
| D3 | 整体位移还是形状误差？ | 每帧算质心距离、面积比；用真值质心整数对齐预测掩膜，观察Dice变化 | diagnostic-frames.csv | 对齐后仍差，说明单纯纠正整体平移不足；对齐改善大则值得研究位置纠偏 |
| D4 | 点到掩膜的表示损失有多大？ | 每例均匀取最多12个非查询帧，真值缩放到384×512；分别采250/500/1000/2000轮廓点，用冻结镜像的原函数填充并缩回；另设仅缩放对照 | representation-summary.csv、representation-frames.csv | 重建损失大则优先检查采样、最大轮廓限制和栅格化；损失小则不能只靠改填充获得大收益 |
| D5 | 实际错误长什么样？ | 每例两张最差主线帧、两张候选相对退化最多帧，去重；并排叠加绿真值、红预测 | jobs/病例/frame-*.png | 人工区分整体偏移、局部形变、边界损失，先看再提新模块 |

## 运行

在远程PowerShell执行（单行，无反引号续行）：

```powershell
git -C C:\zhongliu\zhongliu-tuning pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\diagnose_best_candidates.ps1
```

验证集另跑：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\diagnose_best_candidates.ps1 -Split validation-10
```

可指定 `-Dataset`、`-PreviousResults`、`-OutputRoot`。不需要GPU。保留终端，`runner.log`实时落盘。断电恢复Docker后重跑同一命令：已完成病例核对哈希后跳过，未完成病例重做；没有开机自动运行承诺。失败尝试保留在`.attempts`，不删除正式实验文件。

脚本冻结输入文件、诊断源码、原镜像ID、重建函数和采样数量。任一变化拒绝混用旧结果，请改用新`OutputRoot`。若原冻结镜像已被删除会停止，不会静默换镜像。输出有互斥锁防止双开。

默认目录：`protocol-40-10-38/best-candidate-diagnostics/test-38`。
请回传 `summary.json`、`paired-summary.csv`、`official-case-pairs.csv`、`diagnostic-group-summary.csv`、`representation-summary.csv` 和若干叠加图。

## 解释边界（必须保留）

1. D1复制官方指标，不用新的Dice重新替换原成绩。D2—D4都是诊断量；排除t=0，双空掩膜Dice留空，空预测对非空真值为0。诊断质心距离单位为原图像素，不是官方毫米距离。
2. D3用到了真值，不能部署，不是新算法；质心对齐不是最优刚体配准，不保证改善，也不是理论上界。图像边缘采用零填充，不循环；另记裁剪损失。剩余误差也不能全部认定为形状误差。
3. D4每帧独立重采真值轮廓，没有跨帧点对应，不能称为“真实轨迹上限”。当前采点只取最大轮廓，洞和多连通域可能损失；这一限制正是要诊断的。12帧结果不能直接与全序列官方DSC相减作因果分解。无轮廓或构造失败明确记录覆盖，不按零填充。
4. 模10只是名义分级边界；遮挡合并后不一定对应实际执行边界。时间和场强分组是描述性分析，不确认因果。
5. bootstrap以病例而非帧重采样；若同一患者有多个病例，并不保证患者独立。不报告p值；测试集反复查看，区间仅探索性描述。后续修改在验证集设计冻结后再报告测试结果。
6. 这套实验不声称已经把点跟踪误差与栅格化误差完全分离。若D3/D4仍不能定位，再追加两组真实轨迹、分支融合前后与实际分级边界的观测；不能凭掩膜诊断推断未保存的点轨迹。
# 原镜像不可用时的显式诊断运行环境

默认仍使用已有预测记录中的冻结镜像。若该镜像已被清理，可向 PowerShell 入口传入 `-DiagnosticImage sha256:完整镜像ID`，仅使用本机已有镜像，不构建、不下载、不重跑预测。

脚本分别记录预测来源镜像和诊断运行镜像，并核对镜像内三个重建辅助文件与当前仓库参考文件（仅忽略 BOM 和换行差异）。不一致则停止。`frozen-run.json` 保存依赖版本、辅助文件指纹及替换状态。此校验不代表已证明新环境与不可用的历史环境数值等价；官方指标来自原文件，重建数值只作诊断。

更换运行环境或代码后，已有诊断缓存不混用；如提示指纹变化，传入新的 `-OutputRoot`，保留旧目录。

