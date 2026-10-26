ACDC / MedCoTracker 原视频、人工分割与教师伪轨迹质量审计 v1
=======================================================

用途
----
针对“标注不全、不动、跳跃”的反馈做分层核验，不预设 ACDC 有问题。
本包不是新的预训练/筛选实验，不修改原数据、80/20 划分、旧训练结果或权重。
不删除任何病例、切片、帧或轨迹；不自动生成可用于训练的排除名单。
已有 B1 使用教师伪轨迹训练，而不是 ED/ES 人工分割损失；两类标签必须分开解释。

执行（Windows PowerShell 5.1；先启动 Docker Desktop Linux）
-----------------------------------------------------
默认路径沿用已完成的实验：
DataRoot   C:\zhongliu\acdc-official-v1
AuditRoot  C:\zhongliu\acdc-ab-native-v2
StudyRoot  C:\zhongliu\mct-full-v1
PriorStudy C:\zhongliu\ts-v1 （只读取其中 local-runtime-images.json 的 Training 镜像 ID）
OutputRoot C:\zhongliu\acdc-qc-v1 （全新目录；不要指向原数据/结果）

完整检查，先自动进行依赖检查与合成烟测，通过后才审计：
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\acdc-quality-audit-v1\run_audit.ps1" -Stage all

只做烟测：同一命令把 -Stage all 改为 -Stage smoke。
只检查图像与人工标注：-Stage raw，不要求 StudyRoot 存在。
只检查已有教师标签：-Stage labels；可复用同一 OutputRoot 的 raw 检查。
默认每种子最多 12 个回放；-ReplaysPerSeed 0 可只统计而不生成轨迹回放。
注意不能在同一 OutputRoot 改回放预算、镜像、源码或审计清单；改变配置请另设新目录。
默认容器上限 3 小时，可用 -TimeoutSeconds 调整；超时保留结果，不扩大训练队列。
可用 -RuntimeImage sha256:... 指定另一已存在的训练镜像，但需新 OutputRoot。
镜像需已有 numpy/Pillow/SimpleITK/torch；本脚本不会安装、拉取或构建。

范围与资源
----------
CPU 4 核、内存上限 8 GiB，不申请 GPU，不运行教师或学生推理。
容器禁用网络，根文件系统只读；原数据/清单/外部标签均只读挂载。
只有 OutputRoot 与容器临时目录可写；容器 UID 0 仅用于新输出目录写权限，不改宿主权限。
不挂载 TrackRAD 数据、模型权重、test-38 输出或指标。StudyRoot 仅挂载逐种子的外部
labels、manifest、external/B1/steps.json、verified.json，未挂载整个结果目录。
原始病例检查完成后带结果哈希可复用；教师统计中断后会重新扫描缓存，不会重训。
日志完整保存，终端只显示进度和错误尾部。

检查内容
--------
1. 原数据来源/基本完整性：复核 native-v2 清单的文件哈希、100 名患者及固定 80/20；
   逐病例报告缺失或变动，不把本地哈希当成官方完整性证明。
2. NIfTI：独立读取 qform/sform/pixdim，与 SimpleITK 读图及 ED/ES 几何对应核验。
   qform/sform 差异和 sform 缩放警告是 review，不自动重写头信息或宣布文件损坏。
3. 时序：所有原生帧、所有切片的精确重复、静态有纹理序列、相邻帧归一化强度差、
   突变候选。序列首尾不作为相邻帧；不将短于 16 帧的原生序列判坏、不补帧。
4. ED/ES：原生帧索引、尺寸/间距/方向/原点、标签 ID、类别/切片面积与中心，
   ED/ES 完全相同掩膜、内部空切片、极端相邻切片变化的复核提示。
   没有中间帧掩膜是正常现象；两期掩膜 Dice 只是描述，不用于衡量标注质量。
5. 预处理：每切片 p01/p99、截断比例、输入 512x384 与原像素宽高比，
   原切片 min/max 显示与模型 p01/p99、512x384 宽高比并排比较。
   两侧是显示用缩略预览，不用于哈希核验，也不是物理等比例测量。
   GIF 同时展示首/中/末切片及最多两个被标记切片；ED/ES 静态图覆盖所有切片。
6. 重复：解码后的 float32 整个 cine 精确相同（相同形状）跨患者比较。
   不宣称覆盖近重复、重新采样副本或患者身份校验。
7. 伪标签：扫描 full-v1 3种子各3000份实际消费标签，共9000份；
   标签SHA256、收据、训练steps记录、来源患者/切片/原生帧范围必须一致。
   只用 torch.load(weights_only=True, map_location='cpu')，不加载不受限pickle。
   统计可见度、置信度、出界、静止、位移/二阶差分、查询时刻还原误差。
   按教师槽、种子及患者/切片汇总，保留逐片段信息。
8. 抽样回放：每教师选两个高跳跃/查询误差片段及首/中间片段，去重后最多12/种子。
   使用原始数组和记录的采样重建输入，必须通过原训练 video_sha256 才绘制轨迹。
   回放不是随机流行率估计；不能从抽样的异常比例推断全部标签。
   对静止轨迹处的图像强度变化再记录候选：强度变化也可能是噪声，不等于物体运动。

输出
----
index.html                 本地离线总览，浏览器打开；无外网资源。
summary.json               raw/标签覆盖、错误与回放数量；complete 不等于质量合格。
patients.csv               病例级错误/复核清单，不是剔除清单。
duplicate-cines.json        精确重复候选。
raw/patientXXX/             每病例 JSON、逐帧差/逐切片统计、ED/ES 全切片叠加图和动画。
label-metrics.csv           全部9000个外部训练片段的标签诊断及无效证明。
teacher-summary.csv         3种子×3教师槽统计，仅是伪标签性质，不是准确率。
sampling-slices.csv         实际采样切片次数及静止/跳跃比例。
label-replays.csv           输入回放哈希结果、原生帧号与疑似静止候选。
labels/seed_N/*.gif         通过精确输入复核的轨迹叠加动画。
run-config.json             源码/镜像/清单/提示阈值冻结记录。
*.log                      依赖、合成烟测、检查日志；ITK 警告保留在日志。

结果传输请复制完整 OutputRoot 目录，index.html 使用相对链接。
先发 summary.json、patients.csv、teacher-summary.csv、label-replays.csv 即可初步分析。
不要把受许可约束的原始图像/衍生预览上传到公开 Git 仓库。

解释边界
--------
自动阈值仅用于排序复核：不是医学质量阈值，不以模型低置信度删除难例。
没有密集点真值，无法自动认证真实对应点准确率或确认所有语义标注错误。
背景静止、弱运动病理、心脏基底/心尖变化、二维切片出平面运动均可能触发提示。
预览GIF时间步160ms只用于观看，不代表真实时间；循环回到第一帧不是检测到的跳帧。
不以 TrackRAD test-38 表现决定删哪些患者。后续筛选须人工确认并另建版本化清单，
优先按切片/片段/轨迹处理，保留原80/20归属、病种构成与未筛选对照。

来源说明
--------
官方 ACDC 为心脏 cine MRI，人工参考主要为 ED/ES：
https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
https://humanheart-project.creatis.insa-lyon.fr/databases.html
本包不下载新数据，继承已确认的非商业研究用途，不改变既有许可义务。

本地测试
--------
python -m unittest discover -s diagnostics/acdc-quality-audit-v1 -p test_qa.py -v
powershell.exe -NoProfile -File diagnostics/acdc-quality-audit-v1/test_runner.ps1
本地若缺SimpleITK，真实NIfTI合成往返用例会标记skip；远程依赖预检要求它存在，
因此远程烟测必须执行该用例。合成测试通过也不等于真实100例检查已完成。
