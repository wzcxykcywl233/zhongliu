教师评价器输出权限修复 v1

适用错误：教师推理已经COMMITTED，评价器能读预测，但写/output/metrics.json时
PermissionError。常见原因是推理镜像和评价镜像中同名user的数字UID/GID不同。
新入口会记录实际目录owner、生产进程身份与权限，按实际目录owner运行评价器，
不假定固定UID，不修改Windows ACL，不执行chmod/chown，更不设置777权限。

原实验不变
diagnostics/teacher-single-factor-v1内所有文件、manifest.json、smoke收据及
已完成的jobs都不修改。修复入口位于独立目录，继续挂载原包并运行原worker。
原代码/镜像/权重/数据/配置/指标筛选规则/缓存哈希守卫保持启用。
不构建/拉取镜像，不重新训练，不重新推理已验证病例；未完成的scaled仍会继续。
不要手工改manifest.json来绕过脚本哈希检查，也不要换新目录丢掉已有预测。

修复如何工作
1. 原worker完成原输入守卫、逐病例缓存校验；完整预测通过后才准备评价。
2. 单独CPU预检再次核对原包、原清单、10/38个病例及全部输出哈希；读取
   evaluation目录owner的UID/GID，记录独立preflight收据。
3. 只让评价容器使用该UID:GID。输入/真值/修复代码/收据/原清单均只读；
   仅该profile的evaluation目录可写，容器根文件系统只读，临时/tmp为tmpfs。
   去掉全部Linux capabilities并设置no-new-privileges；不使用privileged或宿主管理员。
   若owner为0，也只是受限容器身份，不代表以Windows管理员运行。
4. 更换UID后保留镜像原/home/user/.local里的依赖搜索路径，以免pip --user
   安装的numpy/scipy等因HOME改变而找不到。不会安装新依赖。
5. 核对镜像内四个正式评分Python文件与原研究的源哈希；先实际写入/删除
   一个唯一临时探针，写入仍失败时在计算指标前停止，输出身份/owner证据。
6. 使用同一Python解释器、同一评分代码、原预测时间戳计算正式指标；原worker
   验证病例数量和metrics后提交完成标记。没有重新定义DSC/D98或改变模型耗时。
7. 若有未验证/截断metrics.json，先逐字节另存快照；由目录owner在评价前
   保留式改名为metrics.unverified-<ID>.json，防止旧文件自身权限继续阻塞写入。
   已验证的指标绝不覆盖，旧失败日志也保留。

远程运行：先拉取，再使用新入口，保持原OutputRoot不变。
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\teacher-eval-permission-fix-v1\run_repair.ps1" -Stage validation -Group weights

预期先出现old的10个CACHE VERIFIED，而不是COMMITTED重新推理；之后是
EVALUATOR PERMISSION PREFLIGHT、EVALUATOR WRITE IDENTITY、WRITE PROBE PASSED，
再提交old指标并继续scaled。终端重复打印早先COMMITTED可能只是进度tail与最终
日志tail重复显示，不代表重复推理；按每个case的verified.json/哈希确认缓存。

后续教师分组也使用这个run_repair.ps1，其他参数/分组/选择/test流程与原入口相同。
PackageRoot默认仓库/diagnostics/teacher-single-factor-v1，绝不指向修复包。
Group及Stage不会自动扩大；仍需先验证、显式select，再显式test。

证据位置（默认原结果目录C:\zhongliu\teacher-single-factor-v1-results）
permission-repair/<split>/<profile>/<8位ID>/preflight.json
permission-repair/<split>/<profile>/latest.json（收据位置与SHA256）
<split>/<profile>/evaluation/permission-repair-receipt.json
如有旧未验证指标，preflight同目录保存previous-unverified-metrics.json。

本地测试
python -m unittest discover -s diagnostics/teacher-eval-permission-fix-v1 -p "test_*.py" -v
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File diagnostics/teacher-eval-permission-fix-v1/test_launcher.ps1
测试不使用本地Docker/GPU；远程容器的实际写权限由运行时写入探针确认。
