# 掩膜级外观校验实验

## 设计目标

本实验保留 CoTracker 主干和已由 10 例验证集选出的
`hierarchical_full_grid0_iterations2_memory_topk`，只在点轨迹生成掩膜后增加
一个无训练参数的外观校验步骤。它借鉴记忆分割模型“以首帧目标外观约束后续
掩膜”的思想，但不加载 MedSAM2，也不改变 CoTracker 点匹配结果。

首帧真实掩膜覆盖区域内的冻结 CoTracker 特征取平均并归一化，得到目标外观
原型。对每个后续预测掩膜，保持形状和面积不变，在特征网格上尝试有限的整体
平移；候选掩膜内的平均特征与首帧原型计算余弦相似度。候选目标函数为：

`similarity - 0.01 × normalized_displacement`。

只有目标函数相对原掩膜提高至少 0.01 才接受移动。超出图像边界、导致面积被
裁剪的候选直接拒绝；查询帧不修正。

## 对照与候选

| Profile | 最大搜索半径 | 每帧候选 | 目的 |
|---|---:|---:|---|
| `hierarchical_full_grid0_iterations2_memory_topk` | 0 | 0 | 固定直接对照 |
| `hierarchical_full_grid0_iterations2_memory_topk_mask_appearance_r4` | 4 像素 | 8 | 保守纠正小幅整体偏移 |
| `hierarchical_full_grid0_iterations2_memory_topk_mask_appearance_r8` | 8 像素 | 24 | 检验更大搜索范围是否有益 |

CoTracker 特征步长为 4 像素，因此候选位移按 4 像素离散。两个实验仅搜索半径
不同，其余参数完全一致。

## 运行与恢复

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "C:\zhongliu\zhongliu-tuning\scripts\run_mask_appearance_validation_resumable.ps1"
```

脚本先检查 40/10/38 数据目录；该方法没有训练参数，所以 40 例只做协议完整性
检查。随后依次评估 10 例验证集和 38 例测试集。每个病例完整生成输出、元数据
和诊断文件后才原子提交；断电后重新执行同一命令即可跳过已完成病例。

结果位于：

`C:\zhongliu\zhongliu-tuning\protocol-40-10-38\mask-appearance-validation`

诊断记录会给出实际校验帧数、候选数、修正帧数、累计位移和相似度收益，避免
出现“配置存在但机制没有实际执行”的无效实验。
