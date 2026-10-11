# Mamba 支线显存调整：稀疏状态 + 解析一阶反向

## 诊断依据

相同病例、主教师索引、查询哈希的前两步，MLP 整步约 2.79 / 1.24 秒，Mamba 约 206.91 / 595.96 秒；Mamba 完整反向约 181.20 / 554.62 秒，占整步 87.6% / 93.1%。Mamba 的逻辑分配峰值约 27.67 GiB，4090 D 报告物理总量约 23.99 GiB，驱动剩余显存为零。记录强烈支持中间状态/反向图造成的显存压力，但不单凭这些计数认定某个 Windows 换页机制。这里只定位训练时间，不能直接解释准确率退化。

当前为 PyTorch 逐帧扫描，并不是官方优化 Mamba CUDA 内核。增加参数只有约 3.38M，但三层、双向、四次训练迭代保留的大量状态远大于权重本身。调整不增加/减少参数，不将 60 帧改为短窗口，也不减少 128 点或四次迭代。

## 改什么

新增显式 `compact` 后端，原 `reference` 和 `parallel` 保留，默认仍是 reference。原模型及旧权重不会自动换后端。

保持原递推：

`h[t] = exp(delta[t] * A) * h[t-1] + delta[t] * B[t] * u[t]`

`y[t] = sum(h[t] * C[t]) + D * u[t]`

前向沿原顺序计算，不构建每帧的自动求导展开图；仅保存输入投影、delta、B/C/A/D，以及每 8 帧开始前的状态。反向时重算当前最多 8 帧的状态，再逆序计算链式梯度。跨段的前向状态及反向伴随梯度均连续传递，不是截断反向，也不是重置记忆；不以衰减除法反推出旧状态，避免小衰减下不稳定。

令 `lambda[t]` 为状态总梯度，计算当前 C、B、u、delta、A、D 的梯度后，将 `lambda[t] * exp(delta[t] * A)` 传回前一帧。只保留稀疏检查点，不保存 `[points,time,channels,state]` 的完整 alpha/beta 张量。块输入投影、卷积、FFN、门控、最终输出层和参数名称都不变，也不重计算整个支线块。

使用自定义一阶反向，**不支持二阶梯度**；FP32 保持原训练精度，FP64 用于 gradcheck。不是新的融合 CUDA 内核，仍有逐帧 PyTorch 调用，所以目前只能预期降低内存压力，不能承诺加速倍数。前向算术顺序保留；解析反向的梯度归约顺序不同，要求数值接近，不承诺训练后的模型位级相同。实现遵循 [PyTorch 自定义自动求导说明](https://docs.pytorch.org/docs/2.14/notes/extending.html)，远程仍沿用已安装的 Torch 2.13 环境，不升级或安装依赖。

## 验证和安全门槛

1. 本地 CPU：六种输入梯度、1/7/8/9/17/60 帧、跨稀疏检查点状态、FP64 gradcheck、非连续输入、极小衰减、真实扫描参数/输入梯度、短/长 Mamba 与 MLP 四次迭代同一图梯度及一次 AdamW 更新；真实小 CoTracker P/V/C 与冻结张量检查。验证保存的张量只有六类输入 + 稀疏锚点，不再存整段逐帧展开图。
2. 远程 `proof`：MLP/短 Mamba/长 Mamba 三个新进程，各读取旧 seed_0 已训练权重只读副本。17 帧 × 5 点与 reference 核对输出和所有参数梯度；随后测 60 帧 × 128 点、同一反向图四次调用，预热一次、测两次。每进程 180 秒预算。reference 只做安全的小形状正确性对照，不再重跑其全尺寸高显存训练图。
3. GPU 小模型 P/V/C：同一组微基准结束后的权重，3 帧 × 2 点、2 次迭代；位置最大差不超过 0.001 输入像素，V/C 最大差不超过 1e-5。不是完整 MRI 或官方分割表现。
4. 三组 proof 全部完成，输出/梯度及 P/V/C 证明通过；合成工况 peak allocated/reserved 均不超过 16 GiB，驱动总量至少 20 GiB、结束时空闲至少 2 GiB，才允许下一阶段。失败/超时/非有限值/旧权重哈希变化均拒绝进入真实训练，绝不静默回退。
5. 显式 `training`：只 seed_0 的 MLP/长 Mamba 各 3 步真实计时，冻结主干、原教师/病例/查询/视频序列配对、硬标签、原千步学习率调度；gain=1、无限幅、无软标签/辅助教师/掩膜损失、无 AMP/TF32、无整块重计算。MLP 的 compact 标签不涉及任何扫描操作，仅保持两组执行配置元数据相同。每种仍为 900 秒预算，**不增加超时时间**。

真实每个完成步骤还检查峰值 allocated/reserved 不超过 20 GiB、结束时驱动空闲至少 2 GiB；若不满足，保存部分计时后中止。预算只在步骤结束时检查，不能保证峰值超额前抢先终止，也不能替代真实 GPU 测量。合成通过不保证真实流程同样通过。相同种子/输入不等于更新张量位级一致，数值归约次序差异要记录。

没有新千步训练、没有自动验证/测试全队列。退化控制仍是独立复用旧权重的诊断，不将这 3 步诊断权重用于正式38例结果。

## 远程运行

新根 `protocol-40-10-38\sidecar-compact-scan-v4`；保留旧 `sidecar-runtime-diagnostics-v3` 的超时报表。复用现有镜像已安装环境，只复制新源码；不联网安装依赖。原训练文件 SHA256 保持不变；代码/数据/镜像/配置指纹仍严格检查，更新后不要在旧根重跑或删指纹。

先单独更新：

```powershell
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
```

再做 GPU 正确性及显存验证：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\repair_sidecar_scan_memory.ps1" -Stage proof
```

看到 `COMPACT GPU PROOF PASSED` 后才做 3 步真实计时：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\repair_sidecar_scan_memory.ps1" -Stage training
```

完成后回传 `compact-readiness.json`、`training-phase-summary.csv`、两组 `training/seed_0/*/phase-profile.json`。若失败，先回传对应日志和部分报告，不加预算，不重启千步训练。相同配置完成的部分可跳过；部分真实更新不能混用，保留后换新根。

本机没有 GPU，实际显存、耗时和真实配对证明仍由远程执行确认。
