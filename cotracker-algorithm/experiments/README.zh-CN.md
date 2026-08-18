# TrackRAD CoTracker3 单点调优说明

本目录提供一组互相独立的推理期调优。默认 `baseline` 完全保留原提交的
关键参数：1000 个边界点、10x10 支撑网格、4 次迭代、不使用可见性或
置信度筛选、直接将有序边界点填充为掩膜。

每个实验 profile 只改变 `ExperimentConfig` 中一个字段，单元测试会自动
检查这一约束。请先完成单点消融，再组合有效项。

| Profile | 唯一改动 | 目的 |
|---|---|---|
| `points_500` | 边界点 500 | 降低时间和显存占用 |
| `points_1500` | 边界点 1500 | 提高医学目标边界采样密度 |
| `support_grid_0` | 关闭支撑网格 | 检验非目标上下文的作用并加速 |
| `support_grid_15` | 15x15 支撑网格 | 增强全局运动上下文 |
| `iterations_2` | 迭代 2 次 | 低延迟工作点 |
| `iterations_6` | 迭代 6 次 | 精度优先工作点 |
| `visibility_0_5` | 可见性阈值 0.5 | 利用 CoTracker3 可见性头剔除遮挡点 |
| `confidence_0_5` | 置信度阈值 0.5 | 剔除不可靠多边形顶点 |
| `temporal_median_3` | 3 帧轨迹中值滤波 | 抑制 cine-MRI 瞬时抖动 |
| `morph_close_3` | 3x3 闭运算 | 修复小型边界裂隙 |
| `largest_component` | 只保留最大连通域 | 注入单目标解剖先验 |
| `lock_first_mask` | 首帧使用原始标注 | 避免轮廓采样/缩放损失首帧精度 |
| `keyframe_stride_2` | 每隔一帧执行追踪 | K-Track 启发的速度—精度实验 |

可见性和置信度筛选保持原始边界点顺序；若某帧不足 3 个有效点，会回退到
未筛选轮廓，避免产生空掩膜。关键帧实验强制保留首尾帧，并按真实帧索引
线性插值中间帧的坐标、可见性和置信度。

## 在远程电脑上运行

模型权重应位于：

```text
cotracker-algorithm/torch/hub/checkpoints/scaled_offline.pth
```

在 WSL 中执行完整单点矩阵：

```bash
cd /mnt/c/zhongliu/trackrad-cotracker-submission-main
python cotracker-algorithm/experiments/run_ablation.py \
  --dataset-dir ./dataset/trackrad2025_labeled_training_data
```

只运行首轮建议项：

```bash
python cotracker-algorithm/experiments/run_ablation.py \
  --dataset-dir ./dataset/trackrad2025_labeled_training_data \
  --profiles baseline points_500 points_1500 support_grid_0 support_grid_15 \
  iterations_2 iterations_6 visibility_0_5 confidence_0_5 \
  temporal_median_3 morph_close_3 largest_component lock_first_mask \
  keyframe_stride_2
```

结果写入 `ablation-results/summary.json`，每个 profile 另存完整日志。比较：

- DSC：越高越好；
- HD95、MASD、中心距离：越低越好；
- Relative D98：越高越好；
- 算法运行时间：越低越好。

此前四个公开训练病例的基线记录可作为运行一致性参考（不是隐藏测试集
成绩）：DSC 0.886662、HD95 4.620684 mm、MASD 1.814792 mm、中心距离
2.182416 mm、Relative D98 0.872867。

## 与论文的关系

- CoTracker3：直接评估其边界点密度、支撑点、迭代次数以及可见性/置信度头；
- K-Track：先实现低风险的关键帧步长 2 + 线性恢复，确认速度收益后再单独
  增加 Kalman 状态模型；
- TrackRAD：保留官方容器接口和统一指标，新增医学掩膜重建先验；
- CoWTracker、PointSt3R、DiT、DINOv2 融合会改变特征或模型结构，需要
  独立训练协议，不混入本轮推理期单点消融。

CoTracker3 的真实 cine-MRI 伪标签微调建议作为第二阶段开展：先用本轮结果
确定推理和掩膜重建设置，再固定这些设置比较预训练权重与医学微调权重。

## 单点消融后的组合验证

50 病例公开数据消融表明，`support_grid_0` 是唯一同时改善五项精度指标并
缩短时间的单点设置；`iterations_2` 提供了均衡的速度收益，
`keyframe_stride_2` 提供了最强的延迟降低。第二阶段只组合这三个已获得
单点证据的设置：

| Profile | 组合设置 | 验证目标 |
|---|---|---|
| `grid0_iterations2` | 关闭支撑网格 + 2 次迭代 | 精度与速度均衡 |
| `grid0_stride2` | 关闭支撑网格 + 关键帧步长 2 | 高速追踪与目标上下文优化 |
| `grid0_iterations2_stride2` | 关闭支撑网格 + 2 次迭代 + 步长 2 | 极致速度候选 |

只运行第二阶段组合验证：

```bash
python cotracker-algorithm/experiments/run_ablation.py \
  --dataset-dir ./dataset/trackrad2025_labeled_training_data \
  --profiles grid0_iterations2 grid0_stride2 grid0_iterations2_stride2
```
