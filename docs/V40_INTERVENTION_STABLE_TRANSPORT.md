# IST-Surv v4.0：Intervention-Stable Additive Transport

> 状态：独立实验候选。它不是 DCT v3.9，也不继承 DCT 的
> IPCW、ETAR、anchor、listwise 或 Slot Attention 主线。是否具有论文创新性，
> 仍需正式文献核验、五折结果和解释忠实性实验共同支持。

## 当前瘦身筛选版（2026-07-30）

- factual 与全部遮挡视图被展平成一个批次，只执行一次向量化
  Sinkhorn；稳定代价只再求解一次，因此每次 forward 从多次顺序求解
  收敛为两个 Sinkhorn 批次；
- 默认只保留 WSI 删除与 pathway 删除两个干预视图，联合删除留作解释
  实验而不是训练默认项；
- 默认关闭与 attribution stability 重复的 risk stability 权重，核心辅助
  目标只保留 plan stability 与 exact attribution stability；
- 筛选配置统一为 20 epochs、30 次 Sinkhorn 迭代。精确 hazard-logit
  分解公式和 deletion re-optimization 均保持不变。

## 1. 科学问题

普通 WSI–omics 融合可能把训练集中的共现关系当成预后证据。IST-Surv
提出的问题是：

> 哪些 patch–pathway transport relations 在系统性遮挡干预下仍然稳定，
> 这些稳定关系能否形成准确、可完整分解、可通过删除实验验证的生存预测？

方法灵感来自 CVPR 2026
[Multi-Modal Image Fusion via Intervention-Stable Feature Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Multi-Modal_Image_Fusion_via_Intervention-Stable_Feature_Learning_CVPR_2026_paper.html)。
该工作研究多模态图像融合；IST-Surv 的候选增量是删失生存任务中的
patch–pathway transport、稳定性回写 cost、重新求解 Sinkhorn 和内生可加和
风险归因。特征遮挡不是临床 `do` 干预，论文只能使用
`intervention-stable` 或 `perturbation-validated`，不能直接宣称生物学因果。

## 2. 独立网络

输入为 UNI2-h patch token \(x_{i,n}\) 和 pathway token \(g_{i,p}\)，不经过
Slot Attention：

\[
C^0_{i,n,p}=1-\cos(x_{i,n},g_{i,p}),\qquad
P^0_i=\operatorname{Sinkhorn}(C^0_i,a_i,b_i).
\]

三类默认干预分别遮挡 WSI patches、pathways，以及同时遮挡两侧。每个干预都
只在可见 support 上重新求解：

\[
P_i^{(v)}=\operatorname{Sinkhorn}
(C^0_i,a_i^{(v)},b_i^{(v)}).
\]

对每条边计算多视图均值、方差和可靠性：

\[
R_{i,n,p}
=\exp\left(
-\beta\frac{\operatorname{Var}_v(P^{(v)}_{i,n,p})}
{\operatorname{Mean}_v(P^{(v)}_{i,n,p})^2+\epsilon}
\right).
\]

稳定质量分数写回 cost：

\[
\widetilde C_{i,n,p}
=C^0_{i,n,p}
-\alpha\log(S_{i,n,p}+\epsilon),\qquad
\widetilde P_i=\operatorname{Sinkhorn}(\widetilde C_i,a_i,b_i).
\]

## 3. 强可解释性是预测方程

每个离散生存阶段 \(s\) 的边风险值由 WSI、pathway 和低成本 pair term
共同产生：

\[
v_{i,s,n,p}
=\operatorname{tanh}\left(
w_s^\top x_{i,n}
+u_s^\top g_{i,p}
+\gamma_s^\top\psi(x_{i,n},g_{i,p})
\right).
\]

最终 hazard logit 没有后置 Transformer 或黑盒 attention：

\[
A_{i,s,n,p}=\widetilde P_{i,n,p}v_{i,s,n,p},
\qquad
\ell_{i,s}=b_s+\sum_{n,p}A_{i,s,n,p}.
\]

因此以下等式逐样本、逐阶段成立：

\[
\sum_{n,p}A_{i,s,n,p}=\ell_{i,s}-b_s.
\]

模型直接导出：

- signed patch–pathway contribution；
- signed WSI patch contribution；
- signed pathway contribution；
- transport stability、variance 和 reliability；
- factual/intervention/stable cost 与 coupling；
- hazard、survival、risk、完整性误差和边际误差。

验证集采样索引按数据集的确定性规则重建。存在空间坐标时输出 signed
scatter；同时存在坐标和原始 WSI 时才输出 overlay。没有坐标时只输出 patch
排名，并记录 `spatial_coordinates_available=false`。

## 4. 训练目标

训练器继续提供离散生存 NLL，模型只增加三个与结构命题对应的目标：

\[
\mathcal L
=\mathcal L_{\mathrm{NLL}}
+\lambda_P\mathcal L_{\mathrm{plan}}
+\lambda_A\mathcal L_{\mathrm{attribution}}
+\lambda_R\mathcal L_{\mathrm{risk}}.
\]

- `plan`：公共可见 support 上 factual/intervention coupling 的
  Jensen–Shannon divergence；
- `attribution`：归一化 signed edge contribution 的稳定性；
- `risk`：轻度遮挡视图的阶段风险不能任意漂移。

第一轮不叠加 IPCW ranking、ETAR、listwise 或 anchor loss。

## 5. 解释忠实性验收

模型内置 `deletion_sweep()`，每次删除都提高相应 edge cost 并重新求解
Sinkhorn，而不是把 contribution 直接置零。正式五折至少报告：

1. hazard-logit completeness error；
2. stable coupling marginal error；
3. top 5%/10%/20% edge 删除相对等数量随机删除的风险变化；
4. 删除正风险边后对应 stage logit 降低的比例；
5. 删除负风险边后对应 stage logit 提高的比例；
6. transport plan shift；
7. patch/pathway top-k 在重复采样和轻度遮挡下的稳定性；
8. 患者级 bootstrap 置信区间。

只有 top attribution 持续优于随机对照，且符号方向在多数患者成立，才能把
“强可解释”写进论文结论。

## 6. BLCA/BRCA 五折

本机只运行单元测试和合成前后向。真实训练在服务器执行。

```bash
# 只查看将要运行的 2 癌种 × 5 folds × 2 protocols
python scripts/run_v40_intervention_stable_transport.py plan

# 服务器 smoke；真实输出与旧结果完全隔离
python scripts/run_v40_intervention_stable_transport.py smoke

# 服务器正式运行
python scripts/run_v40_intervention_stable_transport.py run
```

默认分别保存：

```text
results/ist_surv_v4.0/highscore/full/{blca,brca}
results/ist_surv_v4.0/clean/full/{blca,brca}
```

`highscore` 使用历史全局分箱，仅用于和 v3.3 高分协议对照；`clean` 使用
train-fold-only 分箱。两者不得合并统计，也不得用 highscore 替代严格协议结论。

最近的候选版本可由一个只读总计划统一列出：

```bash
python scripts/run_recent_transport_5fold.py plan
```

它覆盖 v3.6-TCL、v3.7-UNI2H、v3.8 transport consistency 和 v4.0
IST-Surv，均只选 BLCA、BRCA folds 0–4，并继续使用各自独立结果目录。

## 2026-08-11：反馈语义修复闸门

完成的 v4.0 结果必须继续用 `raw_mass + legacy_product` 复现。审计发现旧
`-log(relative_mass * reliability)` 会把低运输质量本身变成大额 cost 惩罚；
在随机 cost 数值探针中，mass 项平均约为 reliability 项的 347 倍，因此旧 B
档主要表现为 plan sharpening，而不是纯粹的稳定性回写。同时，支持遮蔽会改变
均匀边际，直接比较 raw plan mass 会把确定性的边际缩放计入“不稳定”。

修复模式使用：

- `ist_stability_normalization=independence_lift`：每个视图先用其独立耦合
  `a*b` 归一化 transport mass，再比较关系 lift；
- `ist_feedback_mode=importance_weighted_instability`：回写量改为
  `factual_importance * -log(reliability)`，只惩罚重要且不稳定的边。

该模式不是新的默认版本，也不得直接扩跑六癌种。预注册入口为：

```bash
python scripts/run_ist_v40_repair_gate.py plan
python scripts/run_ist_v40_repair_gate.py doctor
python scripts/run_ist_v40_repair_gate.py smoke
python scripts/run_ist_v40_repair_gate.py run
```

默认只跑 BLCA folds 1/2/4。仅当三折均值至少达到 factual A 的
`0.7072 + 0.005 = 0.7122`，且至少 2/3 折改善，才补 BLCA folds 0/3，随后把
SKCM/HNSC/LUSC/KIRC/UCEC 作为锁定后的跨癌种验证；否则停止 IST 论文线。

训练完成后执行以下命令自动生成逐折对照表并给出 `PASS`/`STOP` 裁决：

```bash
python scripts/summarize_ist_v40_repair_gate.py
```

裁决使用与现有 A 档一致的逐折 best validation C-index，同时附带 best3 和
last5 供稳定性审计；不得在看到结果后更换门槛或选择口径。

## 7. 病例解释导出

```bash
python scripts/export_v40_explanations.py \
  --config configs/intervention_stable_survival_transport_blca.yaml \
  --checkpoint /path/to/fold0_checkpoint.pt \
  --fold 0 \
  --coordinate-root /optional/coords \
  --slide-root /optional/svs
```

每名患者输出：

```text
summary.json
stage_patch_pathway.csv
stage_patch_attribution.csv
stage_pathway_attribution.csv
transport_matrices.npz
*_stage*_signed_heatmap.png   # 仅在坐标存在时
```
