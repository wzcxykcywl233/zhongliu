# 全序列 Mamba 融合门控实验

## 设计目的

前一轮 Mamba 轨迹残差和时间注意力替换实验均失败。主要原因是新增模块仅在
20 帧训练片段上模仿原 CoTracker 教师，没有图像身份信息，也没有直接使用
TrackRAD 的逐帧真实标签。本实验不再修改 CoTracker 内部结构，而让 Mamba
只负责完整病例上的长期分支选择。

固定基础方案为 `hierarchical_full_grid0_iterations2`。它保留 CoTracker 的局部
匹配和全局匹配，并把每一帧整理成一个17维状态标记，包括：时间位置、两分支
平均/最大位置差异、可见性、置信度、原始查询特征相似度、速度、加速度、质心
位移及遮挡点比例。

## 四级对照

| 名称 | 是否部署 | 作用 |
|---|---|---|
| `hierarchical_full_grid0_iterations2` | 是 | 固定融合直接对照 |
| `oracle_branch_selection` | 否 | 使用真实标签逐帧选择 DSC 更高的分支，测量可实现上限 |
| `hierarchical_full_grid0_iterations2_mlp_gate` | 是 | 同样17维输入，但逐帧独立判断，不交流时序信息 |
| `hierarchical_full_grid0_iterations2_mamba_gate` | 是 | 双向 Mamba-style 模块读取完整病例序列后输出动态权重 |

只有 Mamba 同时超过固定对照和 MLP，才能把收益归因于长时序状态建模。如果
Oracle 相比固定对照没有足够空间，则应直接停止门控训练方向。

## 数据隔离

- 40例训练集：允许读取完整 `_labels.mha`，生成分支优劣监督标签；
- 10例验证集：计算 Oracle 上限并比较 MLP/Mamba；
- 38例测试集：不生成训练缓存，不读取标签进行模型选择，只运行最终评估；
- 不使用教师模型训练、不使用辅助教师、不使用软置信度标签。

缓存生成会保存每个病例的完整时序标记和 Oracle 标签。训练每25步原子保存
一次状态并保留最近3个断点。评估继续按病例原子提交。断电后重复相同命令即可
分别从病例缓存、训练步或评估病例继续。

## 运行命令

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"

git -C $Repo pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Repo\scripts\run_long_fusion_mamba_resumable.ps1" `
  -RepoRoot $Repo `
  -TrainingSteps 500 `
  -SaveEverySteps 25
```

实时日志：

```powershell
Get-Content `
  "$Repo\protocol-40-10-38\long-fusion-mamba\queue.log" `
  -Tail 80 -Wait
```

Oracle 上限见 `cache-validation-10/oracle-summary.json`。最终38例表格见
`test-38/long-fusion-test-38-results.csv`，相对固定对照的差值见
`test-38/long-fusion-test-38-deltas-vs-fixed-control.csv`。
