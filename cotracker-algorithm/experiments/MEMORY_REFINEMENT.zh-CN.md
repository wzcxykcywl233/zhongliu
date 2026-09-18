# 动态查询记忆：六个独立优化实验

不训练模型。40例训练集仅检查划分；10例验证、38例测试分别评估。
最终表现表使用test-38。测试集已被反复查看，本轮属于探索性比较。

## 固定对照

memory_control与hierarchical_full_grid0_iterations2_memory_topk_diverse完全同配置。
各候选只改变query_memory_refinement一个机制开关，不相互叠加。短名称避免Windows路径过长。
固定原始scaled_offline.pth、1000轮廓点、跨度10、迭代2、网格0、遮挡合并、双分支融合0.5、容量4、
多样性系数0.25、记忆/当前查询特征混合0.5。记忆内部原始锚点份额0.3；无有效历史时回退原始特征。

| 编号 | 配置 | 唯一机制 | 预期与风险 |
|---|---|---|---|
| E0 | memory_control | 当前方案 | 同批次对照 |
| E1 | memory_pointwise_write | 逐点写入 | 减少局部错误污染；可能拒绝真实外观变化 |
| E2 | memory_pointwise_fusion | 逐点质量融合 | 降低不可靠锚点的局部影响；受置信度质量限制 |
| E3 | memory_current_retrieval | 当前外观检索 | 匹配当前外观；当前点漂移可能误导检索 |
| E4 | memory_cycle_write | 循环一致性写入检查 | 拒绝不自洽轨迹；增加计算量 |
| E5 | memory_contour_guard | 弱轮廓修正 | 减少孤立漂移；可能压制局部形变 |
| E6 | memory_recent_slot | 保留最近可靠锚点 | 保留新外观；可能保留相对低质量锚点 |

顺序E0、E1、E2、E3、E6、E4、E5。全部参数预先冻结，不根据测试标签改变配置。

## 参数与边界行为

- E1：每点visibility*confidence>=0.5且与原始特征余弦相似度>=0.5才写入。任一点通过即可接收该帧槽位。
  无效点的特征保存历史有效值，并用有效性掩码禁止该槽位参与该点融合；不重复投票。
  其余槽位仍受容量淘汰约束，无有效历史点回退原始特征。沿用共享锚点质量，不叠加E2。
- E2：准入/淘汰不变。逐点质量q=0.6*可靠性+0.4*clamp((cos+1)/2,0,1)，历史份额0.7逐点归一化。
- E3：准入/淘汰不变。原锚点质量乘exp(cos(current,history)/0.2)后逐点归一化。
  复用本次编码器提取的查询特征，不额外运行编码器；第0金字塔层权重用于所有层。
  可靠性来自已提交上一段在当前查询帧的预测，低于0.5则退回原融合权重。
- E4：在非零查询帧t写入前，从已提交的t位置反向跟踪[max(0,t-10),t]。
  反向使用网格0、迭代2，不带记忆/双分支；与已提交起始位置相距<=2模型输入像素的点允许写入。
  保留原有整组准入，额外逐点屏蔽。原始锚点豁免，重复锚点不重复检查。
  记录额外调用次数及同步GPU后的耗时；循环自洽不保证身份正确。
- E5：在完整跟踪结束后、点转掩膜前，计算相邻帧位移。左右各2个轮廓邻点位移的分量中位数为参考。
  仅可靠性<0.5且偏差>2模型输入像素的点，向参考位移混合25%。非递归修正，不改变查询帧、记忆与后续查询。
- E6：4槽位保留原始1个、最近通过准入1个、其余2个按质量+多样性贪心选择。
  多样性参照包括原始和最近锚点，不重复占位，融合公式不变。

## 执行与恢复

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"
git -C $Repo pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$Repo\scripts\run_memory_refinement_resumable.ps1" -RepoRoot $Repo
```

默认先验证10例、后测试38例，各7组，共336次病例推理。只跑验证加-Stage validation，之后用-Stage test继续。
不自动根据验证结果删除测试候选。结果目录protocol-40-10-38\memory-refinement。

断电后恢复Docker Desktop及GPU，执行同一条命令。已提交病例/指标跳过，中断病例重新推理。
不支持病例内部帧级恢复，不设置开机自动启动。运行时保持终端开启。
源码、权重和数据SHA256冻结；变化时必须使用新ResultsRoot，防止混用缓存。
不复制旧实验结果进入新目录，不并发运行同目录队列。

```powershell
$Root = "$Repo\protocol-40-10-38\memory-refinement"
Get-Content "$Root\queue.log" -Tail 40 -Wait
# 另一个窗口查看病例实时日志：
Get-Content "$Root\test-38\runner.log" -Tail 40 -Wait
```

## 输出

两阶段各有memory-refinement-<split>-results.csv：DSC↑、HD95↓、MASD↓、CD↓、D98↑、TimeSec。
另有-deltas-vs-fixed-control.csv（候选减E0）、-cases.csv（每病例值与差值）、
-diagnostics.csv（每病例计数）和-mechanisms.csv（总计及非零病例数）。

E1看memory_point_rejected/accepted；E2/E3看memory_weight_changed_points；
E3看memory_retrieval_point_reads/fallback_points；E4看memory_cycle_rejected_points/runs/seconds；
E5看memory_contour_corrected_points；E6看memory_recent_slot_prunes。
计数是点事件而非去重轨迹点。0表示未触发；剪枝次数不保证选出的槽位一定不同于E0。
收益仍是待检验假设，需结合病例差值和触发统计解释微小变化。

```powershell
Import-Csv "$Root\test-38\memory-refinement-test-38-results.csv" | ConvertTo-Csv -NoTypeInformation
```
