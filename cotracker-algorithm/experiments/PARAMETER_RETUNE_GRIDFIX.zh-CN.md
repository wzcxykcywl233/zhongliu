# 支持网格与动态记忆兼容修复

错误：`point reliability shape must match query points`。轨迹输出裁剪为N个轮廓点，但原查询特征记忆保留N+grid²个点，导致记忆写入失败。

修复同时覆盖写入和读取：

- 记忆库仅存储前N个轮廓点的track/support多尺度特征。
- 下一分段只对轮廓点应用记忆混合；辅助网格点保留当前分段提取的特征。网格点不是历史跟踪身份。
- 非零网格的可调用记忆检索只收到前N个点。
- grid=0仍采用原完整特征混合路径；默认行为不新增裁剪或拼接。
- 真实网络CPU测试覆盖grid=0/3/5/7、多分段读写以及辅助点特征保持不变。不是只移除维度检查。

## 已完成首轮大部分配置的修复入口

先确保旧队列已经停止，再拉取更新：

```powershell
git -C C:\zhongliu\zhongliu-tuning pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\repair_parameter_retune_grid.ps1
```

旧目录 `parameter-retune` 不修改。修复后的默认结果目录为同级 `parameter-retune-gridfix-v2`。之前失败的 `parameter-retune-gridfix` 也原样保留，不删除或改写任何 frozen-run / frozen-images 指纹。由于运行脚本已更新，不能在旧结果目录里混用新旧源码。

同一轮运行首次构建后会记录算法与评价镜像的不可变 ID；后续恢复、复核及测试直接使用这些 ID，不再重复构建同名镜像。如果记录的镜像已被清理，队列会停止，绝不自动改用当前标签。

流程：

1. 新环境重跑rt_control、rt_repeat、rt_decay三组10例验证对照。
2. 核对旧来源中的两个受修复文件确属已知修复前版本；其他推理/评价代码、依赖配置、权重、数据和实验清单保持一致。
3. 三组对照逐病例预测数组哈希及五项指标必须与旧结果一致。
4. 验证57组grid0旧结果的配置、病例覆盖、输出文件哈希和完成标记后复制到新目录。每组保留reuse-origin.json；新阶段保留reuse-provenance.json，明确记录旧/新镜像、来源和复核情况。
5. 补跑rt_grid_1/2/3，随后继续原验证组合选择和入围复核。

这是显式的跨版本审计复用，不宣称复用结果来自新镜像；三个对照一致也不能数学上保证所有配置数值等价。因此入围配置仍在新环境重跑验证，输出必须匹配才能进测试。复用时间跨运行批次，仅作探索性比较。

任一检查失败都停止，不静默绕过。旧运行和新运行都有文件锁。复制使用独立暂存目录，完成后提交；断电后启动Docker，再执行相同修复命令续跑。没有设置开机自启或修改执行策略。

验证完成后：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File C:\zhongliu\zhongliu-tuning\scripts\repair_parameter_retune_grid.ps1 -Stage test
```

最终报告只使用 `parameter-retune-gridfix-v2\final\test-38`。全新实验可直接使用常规入口并指定全新ResultsRoot，不传ReuseResultsRoot。
