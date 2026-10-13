# 支线退化机制诊断 v3（Git交付）

诊断代码通过Git交付，独立放在仓库根目录的 `diagnostics/sidecar-mechanism-v3`。原实验源码指纹只枚举 `cotracker-algorithm`、`evaluation`、`scripts`；新增此目录不会改这些数值源码。运行时不改冻结镜像、旧训练权重或旧结果，没有训练、优化器、反向传播、镜像构建/拉取或38例重跑。所有输入只读挂载，容器断网。

v3修复了Windows PowerShell 5.1读取子进程退出码为空时的误报：启动后立即保留进程句柄，不再用Refresh丢弃缓存状态，等待输出排空后读取数字退出码。若仍读不到，明确报告“未知状态”，绝不当成0或1继续。默认改用新的v3输出目录，保留v2已有文件；诊断定义不变。

## 要回答的问题

1. **整体偏移还是局部形变？** 对4例已有完整视频的配对轨迹计算逐帧平均位移和零均值剩余形变，分别回放原轨迹、完整支线、仅整体位移、仅局部形变。原轨迹及完整支线必须先与旧 output.mha 逐像素一致，否则停止。
2. **是哪轮迭代产生方向性偏差？** 观察每轮实际支线坐标残差经过冻结 flow head 后的方向、大小及整体/局部分量，并记录1至4轮的轨迹和掩膜几何。只观察，不修改模型输出。
3. **教师目标是否推动同方向修正？** 在完全相同视频、查询点下比较三种原训练教师、冻结基线、MLP、Mamba。记录教师相对基线的偏移及与两个支线最终轨迹差的方向余弦；另记录各轮实际残差与教师剩余位移的余弦（减去残差重建修正前位置，存在浮点舍入误差）。教师可见度>0.9才计入方向统计。教师不是点级真值，此结果不能单独证明训练退化原因。
4. **训练与部署预处理是否一致？** 在实际旧训练镜像中调用数据读取器，用非正方形哨兵确定空间轴顺序，验证读取器SHA；在评估镜像核对关键数值源码。随后比较原始部署输入、仅归一化部署输入、训练读取器轴序+归一化输入。局部代码显示可能存在轴序及强度差异，远程核查前不将其视为已证实原因，也不直接修复。

## 规模和边界

- 只用验证集：B_006（退化集中）、A_028（改善）、A_012（退化）、A_010（小目标）。它们根据已有诊断选取，不是随机独立样本。
- 缓存拆分：4例×3种子×MLP/Mamba，共24对，完整已有视频。
- GPU：先10帧/32点烟测，确认观察前后P/V/C逐位一致，再4例×3输入路径。每个任务60帧、128边界点、4次迭代、种子0，逐个加载冻结基线、两条支线、三个教师。
- 采用已有1000步最终权重，不使用100步速度诊断权重。权重、数据、协议、缓存、脚本、镜像ID通过指纹冻结；输入变化须换输出目录。
- 使用原参考扫描后端，不比较加速后端。默认每个容器最多600秒、全队列4500秒；超时仅停止本次唯一命名的容器并保留文件，不会启动更大任务。
- GPU每个模型前向后检查至少2GiB空闲、峰值不超过20GiB。这是停止门槛，不是对异步OOM的提前保证。
- GPU用首个连续60帧及首帧掩膜边界查询，不复刻历史随机训练批次，也不执行完整分层/动态记忆流程；教师附加网格和随机查询沿用记录的训练设置。跨输入路径不能认为原始空间中的128点身份严格相同。
- `training_exact`仅指读取器的轴序与强度处理，不代表完整训练增强/随机采样的精确重放。
- 掩膜诊断Dice、像素质心、教师方向都不是官方评估指标。不开D98新统计、不做p值或盲测声明、不用测试集选阈值。
- 整体/局部反事实需要两模型轨迹，不是可直接部署的新方法；两部分的Dice收益不必相加。

## 运行

在远程仓库正常 `git pull --ff-only origin main`，打开Docker Desktop。不要复制旧v2包覆盖Git文件，也不要把新代码写进冻结研究目录。

先做CPU缓存拆分及镜像预处理核对：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\sidecar-mechanism-v3\run_mechanism_probes.ps1" -Stage cached
```

再做有时间上限的GPU诊断（复用同一输出目录）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\sidecar-mechanism-v3\run_mechanism_probes.ps1" -Stage probe
```

也可使用 `-Stage all` 一次执行两层。默认输出 `C:\zhongliu\sidecar-mechanism-v3-results`。默认镜像来自旧训练和退化研究的 `local-runtime-images.json`；缺失即停止，不自动替换/下载。可显式指定已安装的不可变SHA镜像，但关键源码核对仍须通过，不接受tag。

失败后保留输出，原样运行可跳过已哈希核验的任务。若更改代码、镜像、输入或权重，使用新的 `-OutputRoot`。旁边的 `.launch-*.stdout.log/.stderr.log` 是各任务日志；不需要粘贴几十MB轨迹。

## 发回哪些文件

1. `summary.json`、`training-preprocessing.json`、`inference-preprocessing.json`。
2. `cached\cached-split-case-summary.csv`（先合并同病例的3个种子）及 `cached\cached-split-summary.csv`。
3. `iteration-corrections-all.csv`、`iteration-geometry-all.csv`、`teacher-directions-all.csv`、`teacher-iteration-directions-all.csv`。
4. 若某任务超时或失败，发该任务 `.launch-*.stderr.log` 和对应 `probes\病例-路径\complete.json`（若存在）。

验证包括10项Python测试、Windows PowerShell 5.1模拟启动流程及真实子进程退出0/7回归测试。它们覆盖CPU真实CoTracker迭代观察、合成缓存回放、非方形读取器、观察钩子不改输出、冻结/防覆盖及空退出码拒绝；远程GPU烟测尚未执行。报告中必须保留这个区别。
