混合区域实验：短序列教师窗口修复 v1
===================================
故障证据：原rs-v1标签缓存至00060.json，step60是TrackRAD C_009/slot2。
确定性抽样公式给step61 draw=20267190、slot1（CoTracker3在线教师）。
失败日志的patient092标注端点为frame01/frame06；本轮取两端点间原生6帧。
原在线教师窗口16、步长8，num_windows=(T-16+7)//8+1在T<=8时为0。
即使模型内部已补足窗口，循环仍未执行：轨迹保持0，可见度sigmoid(0)=.5。
原可见度>.9检查正确地拦截了无效标签，不应降低阈值或删除病例来绕过。

修复范围
--------
只在本实验冻结V2/online3教师输入2..8原生帧时，令num_windows至少为1。
保持模型原有内部窗口补齐和输出裁剪，不在数据集或损失中复制帧/掩膜。
不更换权重、support、迭代次数；不改学生、优化器、损失、可见度阈值。
离线教师、>=9帧教师调用走原路径；不修改APP或旧实验任何文件。
执行窗口后仍须通过原finite/visible>.9检查；不承诺所有短片段一定有好标签。
遇到新标签错误会保存label-failures/seed_*_*.json，含原生帧、教师槽位和诊断。
短序列打印TEACHER WINDOW REPAIRED，必须windows=1且输出帧数仍是原生长度。

兼容与审计
----------
仍使用C:\zhongliu\rs-v1，不删除、不改写manifest或既有检查点。
新region-short-teacher-fix-v1.json绑定新增修复文件及原shape修复文件的hash。
旧source核验不放宽；旧shape修复保持原样，D1 smoke结果可复用。
已成功的旧标签可复用。无修复证明的旧短序列在线教师标签/训练记录拒绝采用。
新标签与训练收据绑定修复；新的resume.pth包含修复hash，防止混用版本。
原子检查点每25步提交：本次M0从最近已提交位置续跑，未提交步骤会重放，
已缓存标签复用；不把标签缓存到step60误解为优化器也提交到了step60。
GPU重放的模型数值不保证逐位复现；输入和教师标签配对校验仍保留。

远程运行
--------
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw "更新失败，请停止" }
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\region-short-teacher-fix-v1\run_repair.ps1" -Stage all

之后续跑/test使用这个入口，它同时包含已验证的区域维度修复。
all仍停在validation-10；38例评测仍必须显式-Stage test，不增加训练预算。
已完成烟测会复用；M0重放到step61时进行真实短教师前向，若仍不合格即停止。
本地测试覆盖真实forward控制流程、模拟窗口、断点/标签来源及启动脚本；
另用完整随机初始化V2/online3网络在CPU核对：修复的6帧输出与原模型已有
16帧窗口计算的前6帧数值一致（坐标逐位一致，可见度/置信度容许float32末位误差）。
这些不是远程官方权重的准确性验证，也不代表区域训练方案已有正向收益。
