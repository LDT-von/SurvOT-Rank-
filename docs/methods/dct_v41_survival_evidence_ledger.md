# DCT v4.1：缺失感知生存证据账本槽

## 一句话定义

DCT v4.1 用一个从零实现的 **Survival-Evidence Ledger（SEL，生存证据账本）** 替换 v3.3 的 Slot Attention 与跨患者 prototype：每个 WSI patch 或 pathway token 把非负“证据精度质量”守恒地写入若干患者条件化账本槽；当某个模态缺失时，另一模态只恢复低秩、可跨模态预测的共享证据，同时显式保留不可恢复的模态私有残差及其不确定性；该不确定性直接降低后续 Sinkhorn OT 边际质量。

实现入口：

- `survot_rank/research/methods/dct_v41_survival_evidence_ledger/model.py`
- 类 `SurvivalEvidenceLedger`
- 类 `CrossLedgerCompletion`
- 类 `DCTV41SurvivalEvidenceLedger`

## 2026-07-30 核心机制升级：共享证据与私有不确定性

对目标模态账本 \(S^t\)，v4.1 不再预测一个看似完整的伪账本，而是作
以下精确分解：

\[
S^t = S_{\mathrm{shared}}^t + S_{\mathrm{private}}^t.
\]

其中 \(S_{\mathrm{shared}}^t\) 经过低秩瓶颈，可以由源模态账本预测；
\(S_{\mathrm{private}}^t\) 是目标模态独有、不能从源模态可靠恢复的残差。
缺失模态推理只把预测的共享证据送入运输，不伪造私有残差。模型同时
预测私有不确定性 \(u_j^{t\leftarrow s}\)，完成置信度为

\[
q_j^{t\leftarrow s}
=\min\left(
q_j^s\,
\rho_j^{t\leftarrow s}\,
\exp(-u_j^{t\leftarrow s}),
q_{\max}
\right),
\]

其中 \(\rho\) 是共享证据可恢复度。训练增加私有不确定性校准项，使
\(u\) 逼近真实配对模态中私有残差的槽级能量。论文主张因此是
“只运输可恢复证据并保留不可恢复不确定性”，而不是一般的特征生成。

## 为什么不继续改 Slot Attention

现有研究已经覆盖了多条直观路线：

- BO-QSA 已研究可学习 query 初始化；
- AdaSlot 已研究动态 slot 数量；
- MetaSlot 已研究固定 slot/codebook 与去重；
- SurvQ、APL 已把查询或 prototype 思想用于多模态生存学习；
- M2Surv、Distilled Prompt、条件潜变量 VAE、MUST 等已研究不完整多模态生存预测。

因此，“换成 learned query”“动态决定 K”“加 prototype”“做一个普通重建 loss”都不能作为 v4.1 的独立创新。v4.1 的边界是：**不再把槽看成待迭代优化的对象 query，而把槽看成可审计的患者级证据账本；缺失补全的置信度不是附加输出，而是 OT 实际使用的质量约束。**

以上论文仅用于划定创新边界，v4.1 没有复制其实现：

- [BO-QSA](https://openreview.net/forum?id=_-FN9mJsgg)
- [AdaSlot, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Adaptive_Slot_Attention_Object_Discovery_with_Dynamic_Slot_Number_CVPR_2024_paper.html)
- [SurvQ](https://openreview.net/forum?id=4oA5xPOTmy)
- [Distilled Prompt, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Distilled_Prompt_Learning_for_Incomplete_Multimodal_Survival_Prediction_CVPR_2025_paper.pdf)
- [MUST, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_MUST_Modality-Specific_Representation-Aware_Transformer_for_Diffusion-Enhanced_Survival_Prediction_with_Missing_CVPR_2026_paper.pdf)

## 模块一：生存证据账本槽

令模态 \(m\) 的 token 为 \(x_i^m\)，其标准化表示、value 与非负精度质量分别为

\[
k_i^m = \operatorname{norm}(W_k\operatorname{LN}(x_i^m)),\qquad
v_i^m = W_v\operatorname{LN}(x_i^m),
\]

\[
p_i^m = \operatorname{softplus}(W_p\operatorname{LN}(x_i^m))+\epsilon.
\]

第 \(j\) 个账本地址由确定性 harmonic code \(h_j\) 与患者上下文场共同给出：

\[
a_j^m = \operatorname{norm}\left(h_j+
F_m\left(\frac{1}{N_m}\sum_i v_i^m\right)_j\right).
\]

token 对账本槽的责任为

\[
r_{ij}^m =
\operatorname{softmax}_{j}\left((k_i^m)^\top a_j^m/\tau\right).
\]

账本质量与槽表示为

\[
M_j^m=\sum_i p_i^m r_{ij}^m,\qquad
s_j^m=\operatorname{LN}\left(
\frac{\sum_i p_i^m r_{ij}^m v_i^m}{M_j^m+\epsilon}+a_j^m
\right).
\]

因为 \(\sum_j r_{ij}^m=1\)，所以

\[
\sum_j M_j^m=\sum_i p_i^m.
\]

这条证据质量守恒关系由结构严格保证，不需要额外 penalty。代码测试会逐患者验证等式。

## 专项检查与维修：Evidence Ledger 审计

v4.1 不把“防塌缩”再做成一个未经验证的附加 loss；每个 forward 会生成只读审计量，并在训练日志中以 `v41_*` 写出。它们不参与反向传播，专门用于定位异常：

- `v41_{wsi,omic}_ledger_mass_error`：写入精度质量与读出 slot 质量的绝对差。正常值应仅为浮点舍入误差；若明显升高，应先检查 token 输入、precision 分支和责任归一化，不应继续解释 OT 结果。
- `v41_{wsi,omic}_ledger_assignment_error`：每个 token 对所有账本地址的责任和偏离 1 的最大值。它直接检查守恒前提。
- `v41_{wsi,omic}_active_slot_fraction`：质量不低于患者期望 slot 质量 1% 的账本槽比例。持续偏低表示 ledger 塌缩风险，优先检查 `v41_ledger_temperature`、token 数量和训练数据，而不是盲目增加 loss。
- `v41_{wsi,omic}_assignment_entropy`：token 到 ledger 地址的归一化责任熵。应结合 active-slot 比例读取：低熵且低 active fraction 是硬塌缩信号，高熵且低性能则更可能是证据缺乏区分度。

在 eval 的 `explain_last_batch()` 中，同一批病人的 `*_ledger_written_mass`、`*_ledger_read_mass`、误差、active fraction 与 entropy 会一并返回，可将异常追到具体病人和模态。先确认两条守恒误差，再查看槽位活跃度，最后才判断补全置信度和 OT 边际是否可信。

## 模块二：不确定性槽级缺失补全

对缺失的目标模态 \(t\)，使用现存源模态 \(s\) 的对应账本槽预测

\[
(\mu_j^{t\leftarrow s},\log\sigma_j^{2,t\leftarrow s})
=G_{s\to t}(s_j^s).
\]

同时预测补全置信度

\[
q_j^{t\leftarrow s}
=\min\left(q_j^s\cdot \sigma(g_{s\to t}(s_j^s)), q_{\max}\right).
\]

这里不是在原始 patch 或 pathway 空间伪造大量 token，而只补全下游 OT 真正需要的 \(K\) 个账本槽。若两个模态都缺失，则使用可学习 null ledger，并把置信度固定到低 floor；该路径保证数值可运行，但不应被解释为有充分患者证据。

支持四种状态：

| WSI | omics | v4.1 行为 |
|---|---|---|
| 有 | 有 | 两侧真实账本 |
| 有 | 缺 | WSI 槽补全 omics 槽 |
| 缺 | 有 | omics 槽补全 WSI 槽 |
| 缺 | 缺 | 双 null ledger，仅作安全 fallback |

训练时对完整病例随机遮蔽一个模态，永远不会同时人为遮蔽两侧。推理时支持全局 `--wsi_missing`、`--omic_missing`，模型接口还支持逐患者 `wsi_available`、`omics_available`。

## 模块三：置信度调温的 DCT

v3.3 产生的 factual OT 初始边际为 \(r_j,c_k\)。v4.1 不把补全槽与真实槽等价处理，而是令

\[
\tilde r_j =
\frac{r_j\max(q_j^{w},q_{\min})}
{\sum_l r_l\max(q_l^{w},q_{\min})},
\qquad
\tilde c_k =
\frac{c_k\max(q_k^{o},q_{\min})}
{\sum_l c_l\max(q_l^{o},q_{\min})}.
\]

随后 v3.3 的阶段化三几何 cost、Sinkhorn factual transport、训练折 low/high-risk anchor、cost intervention、counterfactual re-transport、score-first survival head 与 IPCW ranking 保持不变。也就是说，v4.1 只替换槽前端，并让缺失不确定性进入真实 OT 计算，而不是复制一份 v3.3 forward。

为此，v3.3 基类新增了 `_encode_transport_slots` 扩展点；默认实现仍执行原始 Slot Attention 与 semantic prototype，保证 v3.3 行为不变。v4.1 覆盖该扩展点，并从自身模型中删除继承得到的四个旧模块。

## 自研 SELC 损失

v4.1 的专属目标为 **Survival-Evidence Ledger Consistency（SELC）**：

\[
\mathcal L_{\mathrm{SELC}}
=\lambda_{\mathrm{cmp}}\mathcal L_{\mathrm{cmp}}
+\lambda_{\mathrm{led}}\mathcal L_{\mathrm{led}}
+\lambda_{\mathrm{sur}}\mathcal L_{\mathrm{sur}}.
\]

### 1. 异方差双向槽补全

\[
\mathcal L_{\mathrm{cmp}}
=\sum_{s\to t}\frac{
\sum_j q_j^t\left[
\lVert\mu_j^{t\leftarrow s}-\operatorname{sg}(s_j^t)\rVert_2^2
\odot e^{-\log\sigma_j^2}+\log\sigma_j^2
\right]}
{\sum_j q_j^t}.
\]

代码按逐维 Gaussian NLL 的等价次序实现：平方误差先乘 \(e^{-\log\sigma^2}\)，再加 \(\log\sigma^2\)。teacher target stop-gradient，防止两侧通过共同塌缩获得虚假低损失。

### 2. 账本证据分布一致性

\[
\mathcal L_{\mathrm{led}}
=\operatorname{JS}(\hat q^{w\leftarrow o}\Vert q^w)
+\operatorname{JS}(\hat q^{o\leftarrow w}\Vert q^o).
\]

它约束补全器恢复“证据落在哪些账本槽”的相对分布，而不只是恢复平均 feature。

### 3. 缺失前后生存分布一致性

将完整模态 factual 预测作为 stop-gradient teacher，把 hazard 转成“各时间段事件质量＋最终生存尾质量”的归一化分布 \(P_{\mathrm{full}}\)，对随机单模态遮蔽预测 \(P_{\mathrm{miss}}\) 使用

\[
\mathcal L_{\mathrm{sur}}
=\operatorname{KL}\left(
P_{\mathrm{full}}\Vert P_{\mathrm{miss}}
\right).
\]

它只作用于本批次被人为遮蔽的完整患者，使槽补全直接服务于生存输出，而不是只追求 feature reconstruction。

最终训练总目标仍包含 v3.3 score-first 主线：

\[
\mathcal L =
\mathcal L_{\mathrm{NLL-surv}}
+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{IPCW-rank}}
+\mathcal L_{\mathrm{SELC}}.
\]

默认权重：

- \(\lambda_{\mathrm{rank}}=0.10\)
- \(\lambda_{\mathrm{cmp}}=0.05\)
- \(\lambda_{\mathrm{led}}=0.02\)
- \(\lambda_{\mathrm{sur}}=0.05\)

这些权重是待验证起点，不是已由实验确认的最优值。

## v4.1 继承与不继承

从 v3.3 继承：

- score-first factual survival supervision；
- 阶段化三几何 cost；
- evidence-conditioned factual Sinkhorn OT；
- 仅由训练折建立的 low/high-risk cost anchor；
- cost-space counterfactual intervention 与重新求解 transport；
- 离散时间 survival head；
- censoring-aware IPCW ranking。

明确不继承：

- v3.3 WSI Slot Attention；
- v3.3 omics Slot Attention；
- v3.3 shared WSI prototype；
- v3.3 shared omics prototype；
- ETAR 与 v3.8 loss。

## 实验边界

- 癌种：BLCA、BRCA、STAD、HNSC。用户输入的 `snsc` 在当前 TCGA 数据与仓库命名中按 `hnsc` 处理。
- 折：仅 fold 0、2、4。
- WSI 特征：UNI，1024 维。
- 配置：
  - `configs/dct_v41_survival_evidence_ledger_blca.yaml`
  - `configs/dct_v41_survival_evidence_ledger_brca.yaml`
  - `configs/dct_v41_survival_evidence_ledger_stad.yaml`
  - `configs/dct_v41_survival_evidence_ledger_hnsc.yaml`

先检查数据：

```bash
python scripts/run_dct_v41_survival_evidence_ledger.py doctor
```

查看全部 12 个任务：

```bash
python scripts/run_dct_v41_survival_evidence_ledger.py plan
```

每个任务跑两个 batch 的 smoke：

```bash
python scripts/run_dct_v41_survival_evidence_ledger.py smoke
```

正式运行：

```bash
python scripts/run_dct_v41_survival_evidence_ledger.py run
```

脚本会拒绝其他癌种和 fold 1/3，避免误跑到用户未指定的实验。

## 严格的创新声明与当前风险

当前可以主张的是一个待实证的原创机制组合：

1. 把槽从 object-query 迭代更新改成患者条件化、非负证据质量守恒的 prognostic ledger；
2. 在 ledger 空间完成带异方差不确定性的跨模态补全，并把置信度直接写入 OT 边际；
3. 用 SELC 同时约束槽内容、证据分布和离散生存分布。

当前不能主张“优于 SlotSPE 或现有缺失模态方法”，因为尚未完成 12 个正式任务与同协议对照。论文至少需要：

- v3.3 vs v4.1 完整模态对照；
- 随机缺失 WSI、缺失 omics、混合缺失率曲线；
- 去掉 precision mass、去掉异方差、去掉 confidence-tempered marginal；
- SELC 三项逐项消融；
- 真实缺失与人工缺失的分开报告；
- 参数量、训练时间与 Sinkhorn 稳定性报告。
