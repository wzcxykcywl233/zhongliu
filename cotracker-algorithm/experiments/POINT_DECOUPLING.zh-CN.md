# 跟踪点数与轮廓重建点数拆分实验

前一轮250点相对1000点，在38例中DSC改善32例、MASD改善36例，平均DSC增加0.001085，平均MASD下降0.022089。与此同时D98平均下降0.002270，主要发生在B和X组。既有数据说明值得继续探索，但还不能解释增益究竟来自轨迹还是轮廓。

当前算法从查询掩膜中采样有序轮廓点，用CoTracker跟踪，再将这些点直接交给整数坐标多边形填充。减少点数同时改变查询点集合、空间交互及记忆统计，也改变最终多边形。独立采样250点和1000点还有一个细节：原代码使用包含两个端点的linspace，二者不是严格包含关系。闭合轮廓的首尾重复点沿用既有规则，不在本轮改动。

## 五组固定配置

所有组沿用 `hierarchical_full_grid0_iterations2_memory_topk_diverse` 的跨度10、迭代2、grid0、4槽记忆、特征混合0.5、双锚点融合0.5、遮挡合并及多样性检索，采用与前轮相同的种子20260923和原始权重。

| 名称 | 查询与跟踪 | 重建 | 作用 |
| --- | --- | --- | --- |
| pd_dense | 原生采样并跟踪1000点 | 全部1000点 | 固定对照，对应rt_control |
| pd_dense_repeat | 同配置1000点复跑 | 全部1000点 | 验证查询、浮点轨迹与输出掩膜逐元素一致 |
| pd_dense_thin | 直接读取pd_dense提交的轨迹缓存 | 固定索引选250点 | 固定轨迹，只改变重建顶点密度 |
| pd_sparse_matched | 从同一组1000点中按相同索引选250点，再跟踪 | 全部250点 | 查询坐标与pd_dense_thin严格相同 |
| pd_sparse_native | 原生250点采样并跟踪 | 全部250点 | 对应此前有增益的rt_points_0 |

索引规则为在0到999上等距取250个位置并四舍五入；固定保留0和999。只依赖第一帧查询掩膜，不依赖后续真实标签、病例组别或结果好坏。

## 四个主要配对与预期

1. **pd_dense_thin − pd_dense：重建点数影响。** 若只将同一条稠密轨迹用于250点多边形即可恢复相近的DSC/MASD收益，说明重建密度有贡献。更少的顶点可能减少局部锯齿或异常折返，但也可能丢失边界细节；结果不能预先确定。
2. **pd_sparse_matched − pd_dense_thin：跟踪查询密度影响。** 两组用相同查询位置、相同250点重建，差别是这些点在1000点环境还是250点环境中被跟踪。若这一步改善明显，增益与跟踪管线有关。这个对比不能进一步单独归因于Transformer注意力，因为查询数量也影响记忆质量聚合和遮挡合并统计。
3. **pd_sparse_native − pd_sparse_matched：查询采样网格影响。** 两组点数相同，检查原生250点采样位置是否恰好更合适。
4. **pd_sparse_native − pd_dense：历史总效果复现。** 默认还与前轮对应病例的输出数组哈希核对。

三段DSC等指标差值相加应等于总差值。它们是沿此固定路径得到的配对效应，不是对任意点数和方法都成立的独立可加机制。D98保留所有病例，并单列A/B/C/X；不根据测试集组别选择点数。既有38例已经多次被查看，本轮属于探索性机制实验。

## 远程运行

先拉取，再运行10例验证：

```powershell
$Repo = "C:\zhongliu\zhongliu-tuning"
git -C $Repo pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$Repo\scripts\run_point_decoupling_resumable.ps1" -Stage validation
```

验证审计通过后，在38例上运行固定的同五组：

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\scripts\run_point_decoupling_resumable.ps1" -Stage test
```

使用40/10/38数据协议；本轮没有训练任务，40例只用于数据一致性检查。默认新的结果目录为 `protocol-40-10-38\point-decoupling-v1`，旧重调目录保持原样。默认核对旧目录 `parameter-retune-gridfix-v2\final` 的1000点/250点输出；其他机器没有旧结果时可显式传 `-PriorResultsRoot ""`，审计会如实记为0次历史核对。

## 实时日志与断点

根目录queue.log记录执行过程，各split有runner.log、病例case.log与point-audit.log。首次构建后记录不可变算法/评价镜像ID，后续恢复和测试使用同一ID。代码、权重、数据和实验清单指纹也会冻结。

每例的掩膜、轨迹缓存、诊断、prediction.json及完成标记一起提交；断电后启动Docker并重跑原命令，完成病例跳过，未提交病例重算。文件锁阻止同一目录重复启动。pd_dense_thin读取已提交的pd_dense缓存，不加载教师或跟踪模型。它的TimeSec只包含缓存读取和重建成本，不能与完整推理时间直接比较。其余四组的本轮TimeSec包含轨迹缓存写入开销，与旧轮未存轨迹的时间也不直接等价。

完成后审计：所有病例集合、配置、输出文件哈希、缓存内部哈希、稠密复跑一致性、精确的查询子集、重建源轨迹及历史输出。任何不一致均停止并保留结果，不能通过删除冻结指纹绕过。运行期间不要更新仓库或清理冻结镜像。

## 结果文件

正式性能取 `test-38\point-decoupling-test-38-results.csv`，包含全部五项指标及TimeSec；不用10例验证值填入测试表。

- `point-decoupling-test-38-matched-deltas.csv`：四个配对的五项正式指标差值。
- `point-decoupling-test-38-paired-cases.csv`：逐病例差值及输出是否完全一致。
- `point-decoupling-test-38-paired-summary.csv`：各配对在总体及A/B/C/X中的改善/退化/持平病例数，重点看B、X的D98。
- `point-decoupling-test-38-trajectory-diagnostics.csv`：匹配查询下稠密与稀疏轨迹的位移，以及原生250点与嵌套250点查询位置差异，单位为模型输入像素。
- `point-decoupling-audit.json`：完整性、固定轨迹与历史输出复现结果。

上述轨迹诊断不替代正式指标；没有保证任何候选一定改善。下一步根据配对证据再决定修改采样/重建还是细调跟踪查询密度。
