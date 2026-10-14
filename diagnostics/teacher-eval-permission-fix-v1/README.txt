教师评价器权限与依赖环境修复（运行流程v2，目录名沿用v1）

适用错误
1. 评价器能读预测，但写/output/metrics.json时PermissionError。
2. 改为目录owner用户后，出现ModuleNotFoundError: psutil。
   用户库可能安装在原镜像用户的专属路径，或新UID无权访问该HOME。
   不再猜测/home/user/.local，也不安装/替换新包或放宽HOME权限。

原实验保持不变
diagnostics/teacher-single-factor-v1中的文件、manifest.json、smoke收据及
已完成jobs均不修改。新流程仍挂载原包、调用原worker、冻结原镜像/代码/
权重/数据/配置和筛选规则。只修复评价结果写入与传输，不重新定义指标。

分离“计算用户”和“结果写入用户”
1. 用评价镜像的原默认用户做CPU依赖审计，实际导入numpy/scipy/pandas/psutil，
   记录原HOME、sys.path、解释器、UID/GID、库版本/路径/文件哈希及正式评分源哈希。
   不传--user，不覆盖HOME、PYTHONUSERBASE或PYTHONPATH。
2. 在原推理镜像中核对清单、原包和所有已完成预测，生成新的独立preflight收据。
3. 正式评价仍使用原评价镜像默认用户和原依赖环境，四个评分Python文件哈希
   必须保持一致。输入/真值/代码/收据/清单只读；容器根文件系统只读；
   只在该原用户拥有的私有/output tmpfs内写指标，另有受限/tmp tmpfs。
   去掉全部Linux capabilities并设置no-new-privileges，不使用privileged。
4. 指标原始UTF-8文本通过带标签的ASCII JSON数据消息输出到独立启动日志。
   这是结果数据，不是编码脚本或PowerShell命令；无Invoke-Expression、Bypass。
5. 原推理镜像中的stdlib-only写入进程采用实际evaluation目录owner的UID:GID，
   校验清单、修复收据、结果SHA256、10/38个病例、指标有限性和原预测哈希，
   然后逐字节原样写回metrics.json；不反序列化重算或重排浮点数字。
6. 如有未验证/截断metrics，先快照并保留式改名为metrics.unverified-<ID>.json。
   已验证指标绝不覆盖。原worker最终再验证并提交官方指标完成标记。
7. 若原镜像用户也导入不了依赖，会在计算前停止并保留审计日志。
   写入进程不导入numpy/scipy/psutil，不会再因HOME/UID切换隐藏用户库。

远程运行（原命令/原OutputRoot不变）
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-eval-permission-fix-v1\run_repair.ps1" -Stage validation -Group weights

预期先出现old的10个CACHE VERIFIED；随后START evaluator-runtime，
EVALUATOR ORIGINAL RUNTIME PASSED、OFFICIAL METRICS PUBLISHED BYTE-IDENTICAL，
再提交old指标并继续scaled。不用删除结果、换目录、重跑smoke或构建镜像。
所有分组及select/test均继续使用这个入口；父对照缓存/验证选择规则不变。
终端进度tail与最终tail可能重复显示同一旧日志，不代表重复推理。

证据位置（默认C:\zhongliu\teacher-single-factor-v1-results）
permission-repair/transport/<ID>/runtime.json：原用户/解释器/依赖审计
permission-repair/transport/<ID>/metrics-response.json：原始指标文本及SHA256
permission-repair/<split>/<profile>/<ID>/preflight.json：预测与原包校验
<split>/<profile>/evaluation/permission-repair-receipt.json：计算/写入双方身份及字节一致证明
所有旧修复收据、失败日志和未验证指标快照保留；原manifest不改写。

本地测试
python -m unittest discover -s diagnostics/teacher-eval-permission-fix-v1 -p "test_*.py" -v
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File diagnostics/teacher-eval-permission-fix-v1/test_launcher.ps1
本地不使用Docker/GPU；测试模拟不同计算/写入UID，验证原环境保持和逐字节传输，
并拒绝被篡改/病例不全/依赖变更的结果。远程实际依赖由原用户审计确认。
