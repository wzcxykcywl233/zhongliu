# 支线耗时与退化：独立诊断，不再等待加速门槛

## 为什么换诊断方式

上一轮合成微基准没有选出满足门槛的长 Mamba 加速配置；短 Mamba 的逐帧版本却比长版本慢得多，重计算还出现了数百秒耗时。旧测试在同一进程连续切换配置，只记录 allocated 显存，不足以认定原因就是显存溢出或 Windows 共享显存，也不能把支线微基准当整轮训练速度。现在把三个问题拆开。

## 1. 真实训练阶段计时（优先）

入口 `scripts/run_sidecar_runtime_diagnostics.ps1 -Stage training`，默认仅 seed_0 的 MLP 和长 Mamba 各 **3 次真实更新**。每种支线一个新容器/新进程，单种总预算 900 秒。初始化使用原预训练权重，仍为 40 例加载器、三种主教师池、每 batch 一个教师、60 帧、128 查询、4 次训练迭代、硬标签、无辅助教师/掩膜损失，不使用 AMP/TF32。

原训练文件不改，先核对它与旧训练协议的 SHA256，再在内存中加入计时范围和提前停止。`num_steps` 仍为 1000，保留原 OneCycleLR 调度；不是把学习率计划缩成 3 步。诊断权重另存，不能作为正式实验模型。只观测开头三步，不能判断训练质量或稳态吞吐；数据与教师选择同种子配对并严格审计，冻结主干张量必须完全不变。

计时输出包括数据读取、搬运、教师前向、学生前向、支线前向、损失/其余 forward_batch、反向、优化器、同步、日志和审计、最终保存。每个范围有 inclusive/exclusive 两列；嵌套范围不要相加 inclusive 时间。学生 exclusive 包括主干及其他学生操作，不是纯 Transformer 时间；`backward_total` 包含梯度穿过冻结操作，不是纯 Mamba 反向。周期检查点仍每 25 步，3 步内不会触发；最终模型保存单独记录。

阶段同步会干扰吞吐，作用是定位费用，不代表生产训练速度。训练失败尽量保存 partial `phase-profile.json`；预算到期强制终止可能只留下之前写入的报告和日志。部分诊断不能与重新开始混合，保留后换新 ResultsRoot。

## 2. 独立进程合成微基准

入口 `-Stage profile`。MLP 原实现一组；短/长 Mamba 各原实现、parallel 16/32 点分块 × 重计算开/关，共 **11 组**。每组启动新容器，读取旧 seed_0 权重只读副本，先分别检查该配置的 17 帧 × 5 点输出及全部参数梯度，再以相同随机输入做 60 帧 × 128 点、同一图 4 次调用的前向/反向/AdamW。预热一次，测两次；每配置整个进程预算 180 秒，避免再次一组卡十几分钟。

每次记录分配、保留显存、峰值和驱动剩余显存。它们不是 Windows 共享显存数据；新进程也不能排除其他应用/GPU 驱动负载。预算包含加载、正确性检查和预热，超时不等于单步耗时大于 180 秒。exit 124 标 timeout；137 标 killed，可能是超时升级杀进程或其他 SIGKILL/OOM，不能据此确认原因。其他错误直接报错，不静默改后端。已记录配置可续跑，超时不伪装为成功，不自动选择加速后端。

## 3. 独立退化控制，不重训

入口 `-Stage validation`；不要求 profile 成功、2 倍加速或 smoke 证明。复用原先已训练的 12 组权重中的 MLP/长 Mamba 三种子，固定 **reference 后端、无分块、无重计算**，只改支线修正强度/上限：

| 后缀 | gain | 每次新增坐标修正上限 |
|---|---:|---:|
| full | 1 | 无 |
| gain050 | 0.5 | 无 |
| gain025 | 0.25 | 无 |
| gain050_cap05 | 0.5 | 0.5 模型输入像素 |

上限只作用新增支线的坐标贡献，不限制整个点的运动，不是原 MRI 像素/mm，也不直接限 V/C。其他参数仍为 `hierarchical_full_grid0_iterations2_memory_topk_diverse`。冻结预训练一组 + 8 配置 × 3 种子，共 25 组 10 例验证。因为没有换计算后端，不另跑 parallel 影子对照。程序会拒绝以这个模式跳过 parallel 的一致性检查。

诊断全病例的“新增修正幅度→轨迹变化→掩膜变化→正式指标变化”，按病例合并三个种子后再汇总 all/A/B/C/X，不能把 114 次重复当独立患者。全 split 的逐例指标、限幅比例、轨迹偏移和改变体素数保留，不只挑有利案例。未采用测试病例专用门控。

仍按验证三种子均值选择相同设置：DSC 不低于冻结预训练，D98 差值不低于 -0.001，再按 DSC/HD95/CD/D98 排名。未达标就停止，不扩大到测试。验证完成后显式 `-Stage test` 才评估 38 例，保留 full 两种支线对照，不能根据测试结果再选配置。正式表现只用 test-38 的 metrics.json；这是反复查看过的公开测试集，不作盲测/独立患者统计声明。

## 运行和回传

新目录为 `protocol-40-10-38\sidecar-runtime-diagnostics-v3`；退化评估在其 `degradation` 子目录。旧训练/旧测速/旧报告不动；镜像复用已装依赖，只复制新代码，无网络安装。源代码/数据/镜像/协议变动仍要求新根，不删除原指纹。不使用旧 import-only 恢复脚本。各阶段不要同时占用 GPU。

每条命令单独粘贴执行：

```powershell
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_sidecar_runtime_diagnostics.ps1" -Stage training
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_sidecar_runtime_diagnostics.ps1" -Stage profile
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_sidecar_runtime_diagnostics.ps1" -Stage validation
```

验证选出合格候选后再执行同一入口的 `-Stage test`，不是自动全跑。若没有合格候选，请回传 validation 的诊断表，暂不执行 test。

优先回传 `training-phase-summary.csv` 与两组 `training/seed_0/*/phase-profile.json`；随后回传 `isolated-summary.json`。每组微基准还有 `progress.json`，保留已结束的预热/试验耗时及下一试验前的显存，超时也可回传；写入在试验计时范围之外。真实阶段每个完成的 step 会更新部分报告。退化阶段回传 `degradation/selection.json`、`degradation/validation-10/sidecar-safety-summary.csv` 和 `sidecar-safety-cohort-diagnostics.csv`，必要时附逐例诊断。

本机无 GPU/Docker；本地只验证语法、真实训练器 AST 锚点、无随机/梯度改变、原调度保持、CPU 模型/加载/梯度、模拟队列的预算与数据/选择保护。远程实际耗时与效果仍需运行。
