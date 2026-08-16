# DCT v3.8.2：MGPTR与癌种自适应辅助损失

> 状态：候选训练目标。代码、对照和监控已完成；真实癌种结果出来前，不能写成已经验证有效的方法。

## 方法组成

DCT v3.8.2保留固定权重模式，并新增`adaptive_full`模式。主生存目标
`NLL`始终保持系数1，作为不可关闭的预测锚点；以下五个辅助目标参与自适应分配：

1. `IPCW rank`：删失感知的患者风险排序；
2. `TID/direction`：风险干预方向一致性；
3. `TDM/dose`：风险响应的剂量单调性；
4. `TCR/reconfiguration`：干预必须引起运输计划重构；
5. `MGPTR`：每一种事实运输几何都必须独立携带预后信息。

总体目标为：

\[
\mathcal L
=
\mathcal L_{\mathrm{NLL}}
+\sum_{i\in\mathcal A_t}\lambda_i^{(t)}\mathcal L_i
+\gamma D_{\mathrm{KL}}(p_t\Vert p_0),
\]

其中活动目标集合为\(\mathcal A_t\)。活动目标的基础权重预算保持不变：

\[
B_t=\sum_{i\in\mathcal A_t}\lambda_i^{(0)},
\qquad
\lambda_i^{(t)}
=B_t\left[\rho p_i^{(0)}+(1-\rho)
\operatorname{softmax}(a/T)_i\right].
\]

默认基础权重为：

| 目标 | IPCW | TID | TDM | TCR | MGPTR |
|---|---:|---:|---:|---:|---:|
| 初始权重 | 0.10 | 0.05 | 0.03 | 0.02 | 0.05 |

默认`prior_fraction=0.25`，因此每个目标至少保留原始分配的一部分；活动目标总预算固定，模型不能通过把所有权重同时压到0来降低损失。默认KL强度为0.01，限制权重过快偏离初始分配。

自适应权重只使用训练折梯度学习。它能为BLCA和BRCA形成不同权重，但不等于直接优化验证C-index；最终仍然必须通过固定权重对照和独立验证折判断效果。

## Minimal sibling recipe

`dct_v382_minimal_transport`（新，类名 `DCTV382MonotoneDoseResponse`）只保留 `IPCW rank` 与 `direction`，把
MGPTR / adaptive / dose / reconfiguration 在 `__init__` 中强制归零。它是
回答"单调剂量响应"这一核心科学问题的**最小必要组件集**，与本 fixed-full
对照可在 BLCA 5 折上量化 MGPTR/adaptive/dose/TCR 的边际贡献。

```bash
python scripts/run_dct_v382_minimal_cross_cancer.py plan \
  --cancers blca,kirc,ucec --folds 0,1,2,3,4
```

最小配方的 C-index 与 fixed-full 的差值即这些被剥离项的总体边际贡献。

## MGPTR

事实路径在每个生存阶段产生cosine、euclidean、dot三种Sinkhorn coupling。融合头可能掩盖其中某一种没有预后信息的问题。MGPTR要求每种几何单独预测患者生存分布：

\[
\mathcal L_{\mathrm{MGPTR}}
=\frac{1}{3}\sum_g\mathcal L_{\mathrm{NLL}}(z^g;y,c)
+\beta\frac{1}{3}\sum_g
D_{\mathrm{obs}}\left(\sigma(z^g),
\operatorname{sg}(\sigma(z^F))\right).
\]

`D_obs`只计算到患者的观测时间箱，删失后的未知尾部不参与蒸馏。固定模式下MGPTR不增加参数和Sinkhorn求解；`adaptive_full`只额外增加5个权重分配标量。

## BLCA/BRCA筛选

查看默认计划（BLCA、BRCA、fold0、20 epoch、robust协议、adaptive_full）：

```bash
python scripts/run_dct_v382_mgptr.py plan
```

第一轮建议同时运行三项对照：

```bash
python scripts/run_dct_v382_mgptr.py run \
  --cancers blca,brca \
  --folds 0 \
  --protocols robust \
  --variants base,fixed_full,adaptive_full \
  --max-epochs 20
```

- `base`：NLL + 固定0.10 IPCW；
- `fixed_full`：NLL + 固定IPCW/TID/TDM/TCR/MGPTR；
- `adaptive_full`：NLL固定，五项辅助损失动态分配固定预算。

查看C-index以及最佳epoch对应的权重：

```bash
python scripts/monitor_v382_adaptive.py
```

结果目录：

```text
results/dct_v3.8.2_20ep/robust/<variant>/<cancer>/
```

每个`epoch_curve_fold*.csv`都会记录五项
`train_v382_adaptive_weight_*`，可以检查不同癌种是否学出了稳定且不同的分配。

## 晋级标准

1. `adaptive_full`必须与同协议`fixed_full`和`base`比较；
2. fold0只用于筛选，晋级后运行完整5折；
3. 学到的权重不能长期贴近约束边界；
4. 至少一个癌种提升，并且另一个癌种没有系统性显著下降；
5. 报告最佳epoch权重之外，还要报告权重随epoch变化，防止只挑好看的瞬时值；
6. 若BLCA与BRCA学出的权重几乎相同，则“癌种自适应”假设没有得到支持。
