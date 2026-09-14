# 多锚点动态查询记忆实验

## 目标

本实验只改 CoTracker 分级推理中的查询特征记忆，不替换 CoTracker 主干，不训练
额外网络，也不使用 MedSAM2、教师模型或软标签。直接对照为
`hierarchical_full_grid0_iterations2`，其余推理参数全部冻结。

固定首帧查询锚点用于限制身份漂移。每个已经接受的匹配级边界可产生一个候选
锚点，其可靠性定义为边界点上 `visibility × confidence` 的均值。候选锚点必须
同时满足：可靠性不低于 0.5，且与首帧查询特征的平均余弦相似度不低于 0.5。
发生遮挡级合并时，失败的短级不会写入记忆；只有恢复后的合并级可提交锚点。

## 三个单变量方案

| Profile | 记忆槽 | 保留规则 | 目的 |
|---|---:|---|---|
| `hierarchical_full_grid0_iterations2_memory_latest` | 2 | 首帧永久保留 + 最新可靠锚点 | 验证局部新鲜特征是否足够 |
| `hierarchical_full_grid0_iterations2_memory_topk` | 4 | 首帧永久保留 + 质量最高的 3 个锚点 | 验证多锚点可靠性集成 |
| `hierarchical_full_grid0_iterations2_memory_topk_diverse` | 4 | Top-K 质量分再加 0.25 的特征多样性奖励 | 避免记忆槽被外观近似的锚点重复占用 |

候选质量为：

`0.6 × reliability + 0.4 × ((cosine_similarity + 1) / 2)`。

送入 CoTracker 前，保留锚点按质量融合，首帧锚点至少占历史记忆的 0.3。融合后的
历史记忆与当前查询特征仍按原方案的 0.5/0.5 归一化混合。因此，多锚点方案不会
取消当前局部查询，也不会让首帧身份信息消失。

## 40/10/38 执行规则

这是一组无训练参数的推理实验，40 例训练集只做协议完整性检查，不读取标签。
先在固定 10 例验证集比较三个方案和直接对照；冻结选择后，再看 38 例测试集。
38 例结果不得反向用于调阈值或改槽位。

远程 Windows PowerShell 运行或断电续跑：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "C:\zhongliu\zhongliu-tuning\scripts\run_dynamic_query_memory_resumable.ps1"
```

每个病例独立原子提交，已完成病例会自动跳过。实时日志和汇总文件位于：

`C:\zhongliu\zhongliu-tuning\protocol-40-10-38\dynamic-query-memory`

每个新 profile 的 `diagnostics.json` 必须证明记忆确实发生了写入和融合，并记录
候选拒绝、淘汰、最大槽位数等统计；否则运行器会把该病例判为失败。
