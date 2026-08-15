# DCT v3.8：Transport-Intervention Consistency

> 状态：候选结构目标，尚未经过真实癌种训练，不得写成已经验证的原创损失。

## 为什么需要 v3.8

DCT v3.3 的 factual 预测已经必须经过：

\[
C^F\rightarrow P^F=\operatorname{Sinkhorn}(C^F,a(e),b(e))
\rightarrow r^F .
\]

因此它在计算结构上是 OT 模型，即使额外的 OT energy loss 为零。但是
v3.3 最有辨识度的 high/low anchor cost intervention 与 re-Sinkhorn 只在
评估阶段运行，没有进入训练目标。v3.8 保留 v3.3 的
`NLL + 0.1 * IPCW ranking`，只增加一个与该机制绑定的结构目标。

v3.8 相对 v3.3 **增加 0 个可训练参数**。它的占比不能用参数比例衡量：
新增目标的每个有效值都依赖干预后的 cost、重新求解的 coupling 和最终
risk decoder。关闭三个 v3.8 权重时，训练目标退化回 v3.3；但实际实验
仍使用 v3.7 的 UNI2-h 输入和 highscore/robust 协议，因此结果表中的
`base` 必须称为 **v3.7-matched base**，不能简称为“v3.3”。

## 三个可独立消融的损失

### 1. Transport Intervention Direction（TID）

\[
\mathcal L_{\mathrm{TID}}
=\frac12\left[
\phi(m_d-(r^H-r^F))+
\phi(m_d-(r^F-r^L))
\right],
\]

其中 \(\phi\) 为平滑 softplus margin。它要求 high-anchor transport 提高
预测风险，low-anchor transport 降低预测风险。

### 2. Transport Dose Monotonicity（TDM）

对 \(0<\alpha_m<\alpha_f\le1\)：

\[
r^H(0)<r^H(\alpha_m)<r^H(\alpha_f),\qquad
r^L(0)>r^L(\alpha_m)>r^L(\alpha_f).
\]

四个相邻增量分别进入平滑 margin loss。默认
\(\alpha_m=0.5,\alpha_f=1.0\)；服务器筛选每两个 epoch 计算一次额外的
midpoint 分支，以限制开销。

### 3. Transport Coupling Reconfiguration（TCR）

\[
d(P^{CF},P^F)=\frac12\lVert P^{CF}-P^F\rVert_1,
\]

\[
\mathcal L_{\mathrm{TCR}}
=\frac12\left[
\phi(m_p-d(P^H,P^F))+
\phi(m_p-d(P^L,P^F))
\right].
\]

它不允许 cost intervention 只改变数值记录而 coupling 基本不动。只在
high/low anchor 都已由训练折观察到的阶段计算。

最终辅助目标为：

\[
\mathcal L_{\mathrm{v3.8}}
=0.05\mathcal L_{\mathrm{TID}}
+0.03\mathcal L_{\mathrm{TDM}}
+0.02\mathcal L_{\mathrm{TCR}}.
\]

完整训练目标仍包含 trainer 提供的生存 NLL 和 v3.3 IPCW ranking。

## 必须保留的科学边界

- TID/TDM 把方向与单调性写进了训练目标，所以测试集上的同方向结果只能
  证明约束泛化，不能单独证明解释天然忠实。
- `counterfactual` 仅指 model-based transport intervention，不是治疗干预，
  也不满足个体因果效应识别条件。
- [MOTCat](https://arxiv.org/abs/2306.08330) 已覆盖 WSI–genomics OT 生存融合；
  [DISCOUNT](https://proceedings.mlr.press/v258/you25a.html) 已覆盖
  distributional counterfactual explanation with OT。因此不能把 OT、
  counterfactual 或 direction loss 单独宣布为原创。
- 当前定向检索尚未确认已有工作完整使用“训练折删失风险集 anchor →
  cost-space dose intervention → re-Sinkhorn → survival-risk consistency”
  全链，但正式论文前仍需系统文献核验。

独立验证必须包含随机 anchor、不重新 Sinkhorn、固定 factual coupling、
患者级 bootstrap，以及 patch/pathway deletion 对照。

## 受控筛选

默认实验对照使用 v3.7 UNI2-h highscore 骨架；其生存目标继承 v3.3：

- full-cohort `global_qcut`
- `fit_bins_on_train=false`
- gaussian slot initialization
- BLCA/BRCA fold0/fold2

该协议与历史高分和部分既有代码一致，但不是严格 train-fold-only 分箱。
`clean` 只作为明确的审计协议，不与 high-score 结果混报。

```bash
# 只检查输入文件
python scripts/run_dct_v38_transport_consistency.py doctor

# 只打印默认 BLCA/BRCA fold0/fold2 计划，不训练
python scripts/run_dct_v38_transport_consistency.py plan

# 服务器两 epoch smoke；两 epoch 是为了越过 anchor warmup
python scripts/run_dct_v38_transport_consistency.py smoke

# 默认只跑 full
python scripts/run_dct_v38_transport_consistency.py run

# 第一阶段：基线和三个单损失，判断每项独立贡献
python scripts/run_dct_v38_transport_consistency.py run \
  --variants base,direction,dose,reconfiguration

# 第二阶段：三个两两组合，判断协同或冲突
python scripts/run_dct_v38_transport_consistency.py run \
  --variants direction_dose,direction_reconfiguration,dose_reconfiguration

# 第三阶段：只有前两阶段支持时才运行三项全开
python scripts/run_dct_v38_transport_consistency.py run --variants full

# 也可以一次生成完整 2^3 因子消融
python scripts/run_dct_v38_transport_consistency.py run --variants all

# 最快的跨癌种初筛：STAD + BLCA，fold0，20 epoch，共 2×1×8=16 次
python scripts/run_dct_v38_transport_consistency.py run \
  --protocols robust \
  --variants all \
  --cancers stad,blca \
  --folds 0 \
  --max-epochs 20

# 严格分箱审计
python scripts/run_dct_v38_transport_consistency.py run --protocols clean
```

独立结果目录：

```text
results/dct_v3.8_transport_consistency/<protocol>/<variant>/<cancer>
results/dct_v3.8_transport_consistency_smoke/<protocol>/<variant>/<cancer>
```

脚本发现对应 fold 的最终结果时默认跳过，不覆盖 v3.3、v3.5、v3.6、
v3.7、ETAR 或 Recovery 结果。

## 晋级规则

fold0/fold2 只能筛选。完整 \(2^3\) 因子消融包括：

| 类别 | 变体 |
|---|---|
| 基线 | `base` |
| 单项 | `direction`、`dose`、`reconfiguration` |
| 两项 | `direction_dose`、`direction_reconfiguration`、`dose_reconfiguration` |
| 三项 | `full` |

不能只用 `base` 对比 `full` 就判定三个损失都无效。应先看单项主效应，
再看两项交互；例如单项有效而 `full` 下降，说明更可能是损失冲突或权重
不合适，而不是三个机制全部无效。`full` 至少需要满足：

1. 无 NaN，`v38_finite=1`，anchor stage coverage 可审计；
2. 相对同协议 base，四折总体 Last5 提高，或 Best–Last gap 明显缩小；
3. TID/TDM/TCR 至少有一项单独消融能解释 full 的收益；
4. 随机 anchor 和 fixed-coupling 对照不能复现同样响应；
5. 通过后才运行 BLCA/BRCA 完整五折，再考虑 LUAD/LUSC。

不满足时保留 v3.3/v3.7，不继续增加结构损失。

## Minimal sibling recipe

`dct_v382_minimal_transport`（详见 `docs/DCT_V382_MGPTR.md` 的 minimal 节，类名 `DCTV382MonotoneDoseResponse`）
只保留 `IPCW rank + direction`，把 dose / reconfiguration / MGPTR /
adaptive 全部置零。它使用与本文件相同的 direction 数学形式，但跳过其他
两项独立消融已显示无显著增益的项，从而把"单调剂量响应"作为可单独归因
的科学 claim 来论证。
