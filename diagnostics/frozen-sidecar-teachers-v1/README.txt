冻结主干主线：固定 Mamba 学生，只比较教师（v1）

长期方向（用户于 2026-10-16 明确）
冻结官方 CoTracker 主干，保留既有支线学生，探索能改善学生表现的教师。
不要默认转回全主干微调，不要把教师自身分割得分当作学生一定提升的证据。
本次只改变离线教师槽，另外两个在线教师仍固定；不是把全部训练步都交给新教师。

为什么这样做
全主干微调中，新 iter6 教师减轻了退化，但没有超过直接使用官方权重。
这不回答冻结主干支线是否会受益。因此回到已经验证过的冻结 Mamba 方案。
固定支线结构，不再同时扩展 MLP/Conv 或改变学生点数，以专注教师单因素链：
old：baseline_offline.pth，100 个随机辅助查询，4 次迭代。
scaled：仅替换为 scaled_offline.pth，仍 100 个辅助查询、4 次迭代。
grid0：在 scaled 上去掉离线辅助查询，仍 4 次迭代。
iter6：在 grid0 上改为 6 次迭代。
注意 grid0 是历史配置名称，这里去掉的是离线教师随机辅助查询，不是学生轮廓点。

固定学生和数据口径
冻结原 scaled_offline 主干的全部参数，只训练原 Mamba 支线 3,377,325 参数。
总参数 28,763,025；这是仓库中的 Mamba-style 实现，不是新增官方 mamba-ssm 模块。
每段 60 帧，128 个任意时间训练查询点，学生迭代 4 次，lr=5e-5，硬标签；无辅助教师损失、无 mask 损失。
seed=20261008/20261009/20261010，独立 fresh 初始化，每组 1000 步，共 4x3=12 组。
复用旧优化器、调度器、数据增强和可恢复状态，不承接 ts-full-v1 的全主干训练权重。
推理统一为 hierarchical_full_grid0_iterations2_memory_topk_diverse：1000 个轮廓点、2 次迭代。
本次不切换到 250 点 rt_points_0；教师收益优先和本轮相同学生的 old 对照比较。
训练 40 例，验证 10 例，报告 38 例。所有四组预先声明并完整报告，不按测试集挑选。
N=3 表示三个训练种子，不能写成只有三个测试病例；pretrained 只有一个固定参照。
38 例已多次查看，不能声称独立盲测。

运行与审计
使用 ts-v1 已验证的三个不可变镜像 ID；不下载、不安装、不重新构建。
原 cotracker-algorithm/scripts/evaluation 文件不改，保持旧实验来源指纹。
新包在内存中检查并调整原训练器的 sidecar 兼容条件；未改原损失、优化器。
Mamba 统一 compact 扫描、gain=1、无位移截断、无 block checkpoint。
旧教师也在 compact 下重跑：原 reference 扫描的历史结果只作背景，不充当本轮配对对照。
逐项核对主干 tensor hash、支线初始权重、病例、视频、查询点、教师抽样顺序。
原来抽到在线教师的步，原始 P/V/C 标签必须完全一致；断点重复记录也必须一致。
检查模型加载为正确支线、分支实际执行、输出文件哈希、正式指标来源。
每组每 25 步存断点，保留最近 3 个；崩溃后重复同一命令从完整断点续跑。
10 步 smoke 仍使用 1000 步学习率日程，单独目录；正式训练不接着 smoke 权重训练。
原随机种子的前 5 步没有抽到离线教师，因此检查跑到第 10 步，覆盖实际换教师的步骤。
先通过 smoke 的训练和 GPU 推理检查，才允许 all。
连续 5 步中位数超过 8 秒（前 2 步预热除外），或显存峰值超过 20GiB/余量少于 2GiB，会停止。
每个 smoke 进程 10 分钟上限，每个完整训练进程 4 小时上限；这些是异常保护，不是耗时预测。
CPU 审计、GPU 训练和推理的时间不同，不能拿训练步时间代替端到端耗时。
默认输出用短路径 C:\zhongliu\fst-runtimefix-v2，避免之前的 Windows 长路径问题。

2026-10-16 运行镜像校验修复
旧检查误把仓库顶层 tests/*.py 当作镜像必须包含的文件；两个 reuse Dockerfile 都没有复制这个目录。
运行检查现严格对应镜像 COPY 范围：根目录 *.py 及 ext/co-tracker、resources、experiments 下的 Python 文件。
这些范围内的文件缺失或哈希不同依旧停止；实际加载的支线模块仍须位于 /opt/app。
整个仓库（包括测试文件）的来源校验仍保留，冻结主干/配对输入/标签/指标校验没有降低。
原 C:\zhongliu\fst-v1 在训练前失败，不修改其中 manifest、不删除任何内容。
修复包的指纹会变化，所以默认切换到 fst-runtimefix-v2；先重新 smoke，再在同一个新目录运行 all。
不需要重建镜像、下载环境或重新进行旧实验；四组教师与固定学生设置不变。

远程运行（先 git pull，保持 Docker Desktop 开启）
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\frozen-sidecar-teachers-v1\run_experiment.ps1" -Stage smoke
smoke 完成后：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\frozen-sidecar-teachers-v1\run_experiment.ps1" -Stage all
也支持 plan/train/validation/test 分阶段。
默认 PriorStudy=C:\zhongliu\ts-v1，可用 -PriorStudy 指向原 smoke/pilot 的根目录。
代码或数据指纹变化时使用正确旧 checkout 或新结果目录，禁止删除指纹绕过检查。

最小回传
Get-Content -LiteralPath "C:\zhongliu\fst-runtimefix-v2\full\test-38\frozen-sidecar-teachers-summary.csv"
Get-Content -LiteralPath "C:\zhongliu\fst-runtimefix-v2\full\test-38\frozen-sidecar-teachers-paired-deltas.csv"
Get-Content -LiteralPath "C:\zhongliu\fst-runtimefix-v2\full\label-runtime-diagnostics.csv"
还会生成逐病例差值、三种子差值均值/SD、每项改善种子数及来源审计。

先回答的三个问题
1. scaled/去辅助查询/增迭代，各自在同一冻结学生上是否带来稳定收益？
2. 新教师学生是否超过本轮 old 学生以及同配置官方权重参照？
3. DSC/距离改善是否以 D98 下降为代价？同时报告正负结果和训练成本。
若要扩展 diverse/span5 等教师，另建实验并先验证任意时间查询接口；不能把首帧轮廓接口直接塞入训练。
