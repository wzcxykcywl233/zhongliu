# 选角有效性诊断：不重跑38例、不构建镜像

本任务只分析已有旋转回查并运行已知角度合成测试，不改预测、训练或调整阈值。原38例官方结果保持不变。

## A. 已有真实结果

读取test-38的38对diagnostics.json及预测文件，核验控制组与诊断组输出一致及预测文件SHA256。原日志没有逐点分数，因此不能恢复逐点胜率、分位数、角度误差。

但回退0度点的patch_aligned与patch_zero相同。某帧有N个有效点、A个接受非零角的点，则非零角点的增益总和可恢复为N*(帧均值aligned-帧均值zero)。跨帧求和再除以总A，得到非零角点帧加权平均增益。记录float32均值带来的保守舍入容差2e-6*N/A；极小增益不应超出精度解释。

输出all_supported和original_main_eligible（有效点>=4）两种范围，分病例统计。非零角是算法筛选后的子集，平均增益不是因果证明，更不等价于位置更准确。没有重新读标签或重新推理MRI视频。

## B. 已知角度的合成测试

使用原实验镜像中的原始scaled_offline.pth及其fnet。固定4个种子，每种子3类纹理：平滑随机纹理、重复条纹、常量平面（不可辨识负对照）。从128像素源图直接采样64像素查询/目标块，目标施加已知旋转；不加平移、遮挡或形变，先隔离旋转问题。

真值角度[-30,-20,-10,-5,0,5,10,20,30]，分为搜索网格内、范围内非网格、范围外三类。三个固定探针偏移(2,1)、(8,4)、(12,-6)，模拟历史偏移位置，不是假装由真实迭代产生。共4*3*9*3=324项相关试验。

搜索仍是[-20,-10,0,10,20]，原gain>=0.01及gap>=0.02规则不改。同时保存原始argmax和回退后角度。输出MAE、始终选0度的参照MAE、非零接受率、边界命中率、五角得分及中心重新编码增益。

判断：原始argmax已不准时，不能只归咎阈值；原始argmax准确但被大量回退时再考虑规则；范围外样本落边界本属预期，不与范围内混算。平坦纹理无可辨识旋转角，其MAE只作程序输出，重点看是否出现错误的自信接受。不要把“全部回退所以没有假阳性”当作选角准确。

合成是已知图像变换，不是真实MRI成像或物理旋转。查询旋转与目标生成共享插值方式，可能乐观；不据此直接扩大搜索范围或开启纠偏。重复纹理/同图多角度/多偏移相关，不进行独立样本显著性检验。

## 运行与恢复

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"
git -C $Repo pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$Repo\scripts\diagnose_rotation_selection.ps1" -RepoRoot $Repo
```

读取`protocol-40-10-38\rotation-backcheck\test-38`，写入独立`protocol-40-10-38\rotation-selection-diagnostics`。使用frozen-images.json记载的镜像ID，不使用可能变化的latest标签，不重新build，不联网下载、不改旧结果。旧镜像已被删除则明确停止，不自动换镜像。

当前仓库的诊断源码只读挂载到镜像，不替换旧镜像模型。源日志/诊断脚本/辅助脚本/镜像/权重指纹冻结。输出目录互斥锁防止重复运行；按种子+纹理原子保存12个断点，断电后同命令恢复。CPU负责读日志，合成编码用GPU；先结束其他GPU实验。diagnostic.log实时打印完成单元。

完成后发回：

- selection-diagnostic-summary.json
- synthetic-angle-summary.csv
- real-case-accepted-gains.csv（需要核对病例差异时）

完整明细：real-frame-accepted-gains.csv、synthetic-angle-trials.csv。无需重跑原旋转队列。
