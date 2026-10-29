区域输出维度修复 v1（绑定原8239037，不改原包）
================================================
已观察到：M0/W1/D1/G1各完成6步训练，M0两例推理通过；D1-region首次推理失败：
torch.Size([1,239,1,270,270]) != (1,239,270,270)。pilot/full尚未开始。

根因：原convert_point_trajectory_to_mask_sequence真实返回B,T,H,W。
model.py附近有一条过时的五维注释；新增region分支误返回B,T,1,H,W。
reshape_video会保留非空间维度，所以缩放并不能消掉额外的通道轴。
修复仅将各块T,1,H,W拼接后移除channel轴，再添加batch轴。
不使用无参数squeeze，保证T=1时batch和time两个轴都保留。
不修改训练、损失、权重、阈值、像素值、插值方式或评测公式。

恢复方式
--------
原region-supervision-v1目录全部保持原样，原manifest.json不改写。
新入口校验原runner及infer.py的精确版本，只在内存中改一个推理返回语句。
新增region-output-shape-fix-v1.json记录修复源码hash；region输出收据绑定它。
所有原代码/输入/模型收据仍严格校验，不放宽source校验或删除checkpoint。
已完成四组smoke训练会显示CACHE VERIFIED，不重训；M0输出与指标复用。
D1-region重新推理，烟测评测及审计通过后，all才继续pilot/full和验证10例。
这不是重新创建结果目录，不要删除C:\zhongliu\rs-v1，也不要改原infer.py。

git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw "更新失败，请停止" }
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\region-output-shape-fix-v1\run_repair.ps1" -Stage all

之后同一队列的续跑及test也使用这个run_repair.ps1。-Stage test仍须显式执行。
完整队列继续采用原门槛；本修复不代表区域方案有效，只修正接口兼容性。

测试
----
test_fix.py覆盖T=1/2/5/239、尾块、多通道拒绝、原生非方形尺寸缩放；
并执行真实model.py的点/区域后处理链（CPU合成帧、模拟追踪特征），
检查可选首帧锁定分支、W,H,T返回、region修复前错误可复现、point路径不变。
test_runner.ps1校验新入口只替换推理dispatch，未修改训练队列或原文件。
本地测试不等同远程GPU真实病例通过；远程重新跑D1 smoke后才能确认。
