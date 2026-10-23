MedCoTracker / ACDC / A-B 第一轮准备包
===================================

当前边界
--------
本包已实现：官方下载、逐文件重试与核验、患者级划分、4D 时间轴/ED-ES
标注核对、真实 cine 读取器，以及 A1/B1/B2/B3 参数和梯度检查。
本包尚不启动医学预训练、TrackRAD 微调或正式性能评估。
run_prepare.ps1 的 all 指“全部准备步骤”，不是完整训练队列。
先拿到完整 data-manifest.json，再接教师伪标签/GPU 烟测与训练队列。
不修改原实验包、不构建镜像、不拉取镜像、不访问 TrackRAD 或官方 ACDC test-50。

为什么选 ACDC
------------
目标是把 CoTracker 特化成 MedCoTracker，不移植 MedSAM2 网络。
ACDC 提供真实临床短轴心脏 cine MRI。它包含真实时间变化，但心脏运动、
门控采集、组织与 MRI-linac 肿瘤追踪有差异；不能预先保证迁移有效。
下载的是完整 *_4d.nii.gz，不是只有 ED/ES 两帧的分割图片。
只读同一 z 切片上的时间序列，绝不把 z 轴当作时间轴。

官方来源与许可
--------------
介绍：https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
下载：https://humanheart-project.creatis.insa-lyon.fr/database/
许可：https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/item/66e290e0961576b1bad4ee3c/download
官方许可为 CC BY-NC-SA 4.0，并限定非商业科学研究。
运行下载须由操作者确认 -AcceptResearchTerms；脚本保存许可原文和引用要求。
用户已于本轮交流确认该用途；脚本保留显式开关，便于日后其他使用者确认。
数据、训练权重与患者文件都不要提交到 Git。
必要引用：O. Bernard et al., IEEE TMI 37(11), 2514-2525, 2018,
doi:10.1109/TMI.2018.2837502。

数据规则
--------
1. 只下载官方 training 的 patient001..100，6 个必要文件/患者，加许可与引用。
2. 按 Info.cfg 的 Group 分层，不从患者编号猜疾病类别。
   每类 16 train + 4 dev，总计 80/20；固定 hash 排序种子 20261023。
   同一患者所有空间切片、所有时间帧、ED/ES 都留在同一集合。
3. NIfTI 标量图像维度必须为 X,Y,Z,T，SimpleITK 数组必须为 T,Z,Y,X。
   同时用 ED/ES 原图与 4D 对应时间帧比对，验证时间索引不是猜出来的。
4. 16 个原生连续帧，stride=1，不循环、不倒序补帧，不凑成 60 帧。
   每个 epoch 每位患者一个随机切片/随机时间窗，避免切片多的患者占优。
5. 固定切片整个 cine 的 1/99 百分位归一化，输出 float32 [0,255]，
   灰度复制为三个通道，双线性缩放到 384x512；不使用标注选切片或做心脏裁剪。
6. ED/ES 掩膜只用于质控/后续独立诊断。未标注帧不能当成背景标注，
   分割轮廓也不是跨帧同一物理点的真值。第一轮不加 mask loss。
7. 官方文件列表未公开返回 SHA256/SHA512 时，明确不声称官方校验和验证；
   核对官方文件 ID/大小，保存下载后本地 SHA256，再通过 NIfTI/ED-ES 一致性检查。
8. 重试复用核验通过的文件；未完成的 .partial 文件重下，完整文件不覆盖。
   单文件下载重试4次，超时/中断保留现场。修改了文件或数据范围则使用新目录。
9. 历史准备文件可重复核验；本地 1 患者样本仅供审计，不能生成正式训练划分。
10. SimpleITK 若报告原始 NIfTI sform 警告会保留到日志；不覆盖原文件。
    spacing 的第4项不自动解释为可靠秒数/帧率；本轮按原生帧索引采样。

A/B 的严格区分
--------------
R1：无 ACDC 预训练；官方主干 + 新支线，直接走相同 TrackRAD 阶段。
A1：ACDC 只训练 Mamba 支线（3,377,325 参数）；主干保持官方权重，
    将已预训练支线带入 TrackRAD，继续只训练支线。
B1：ACDC 只更新 fnet 图像编码器（2,632,288 参数）。
B2：ACDC 只更新 updateformer 及其坐标/V/C 头（21,732,484 参数），
    corr_mlp 不更新。B2 不是“小型输出头适配”。
B3：ACDC 更新全部 25,385,700 个主干参数，小学习率。
B1/B2/B3 的外部阶段都不挂支线，因为当前支线会 detach 主干特征。
之后冻结各自适配过的主干，挂同样种子初始化的新 Mamba，TrackRAD 只训支线。
所有路线教师固定；不训练教师，不把医学预训练与教师调优混在一轮。
具体预算、固定教师池与选优规则在 plan.json；其训练阶段属于待接入计划。

比较与止损
----------
外部预训练 pilot 500 步，TrackRAD pilot 250 步，先 seed0。
正式确认规划为外部3000/TrackRAD1000、3个种子，需确认后再开，不能本脚本自动开。
只用 TrackRAD validation-10 选择；正式 test-38 仅在方案冻结后报告。
同一 R1 对照下 DSC 必须提高，D98 下滑不得超过0.002，HD95上升不得超过0.02；
正式确认要求至少2/3种子DSC改善，任一种子DSC下降不超过0.002。
无合格候选就保留 R1，不因“预训练完成”强制升级。
ACDC dev 的伪标签损失变低不是医学跟踪准确率变高；还要看真实迁移效果。
历史 .882228 只作背景，本轮需要同期 R1。额外 ACDC 步数不是等算力比较；
如要归因“医学数据胜过多训”，另设同等额外更新预算的 TrackRAD-only 对照。
test-38 反复看过，不能称为全新盲测。

远程运行（代码已传到仓库后）
----------------------------
先启动 Docker Desktop。默认复用 C:\zhongliu\ts-v1\local-runtime-images.json
中的 Training 镜像哈希，不需要再次下载依赖。各命令独立一行粘贴。

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\medcotracker-acdc-ab-v1\run_prepare.ps1" -Stage all -AcceptResearchTerms

默认数据目录：C:\zhongliu\acdc-official-v1
默认审计目录：C:\zhongliu\acdc-ab-v1
all = download（唯一允许联网的步骤）→prepare（断网CPU）→check-model（断网CPU）。
无需 GPU，不会与另一训练竞争 GPU，也不会在完成后自行启动训练。
若镜像哈希已不存在，可显式提供已核实的本地训练镜像 -RuntimeImage sha256:...。
下载失败重复原命令；只做离线检查可 -Stage prepare 或 -Stage check-model。

完成后查看
----------
Get-Content -LiteralPath "C:\zhongliu\acdc-ab-v1\data-manifest.json" -Raw
Get-Content -LiteralPath "C:\zhongliu\acdc-ab-v1\ab-parameter-check.json" -Raw

无需粘贴完整大 JSON。把最后日志，以及 data-manifest 的 training_ready、
split_counts 和4条参数检查结果发回即可。没有正式GPU烟测通过前，不运行大训练。

测试
----
python -m unittest discover -s diagnostics/medcotracker-acdc-ab-v1 -p test_preparation.py -v
python diagnostics/medcotracker-acdc-ab-v1/ab_modes.py --app cotracker-algorithm --output <新审计目录>
依赖：Python3.10+、NumPy、PyTorch、SimpleITK；下载阶段仅用 Python 标准库。
ab_modes 是随机权重CPU梯度/冻结检查，不能宣传成官方权重GPU训练烟测。
