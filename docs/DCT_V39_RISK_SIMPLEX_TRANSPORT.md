# DCT v3.9：Risk-Simplex Transport（风险单形运输）

> 状态：已实现、已单测（13 项）、已本地标定，**尚无任何真实癌种结果**。
> 下面所有"提分"都是机制层面的论证，不是实测结论。
> 代码：`survot_rank/research/methods/dct_v39_risk_simplex_transport/model.py`
> 配置：`configs/dct_v39_risk_simplex_transport_blca.yaml`
> 注册名：`dct_v39_risk_simplex_transport`（别名 `dct_v39` / `rst`）

---

## 1. 先说 v3.3 为什么不好：一个实测到的表征塌缩

这不是推测，是本地实测。`_semantic_slots` 用 8 个 shared prototype 对 8 个
local slot 做 softmax 加权平均，实测结果：

| 阶段 | slot 两两余弦 |
|---|---|
| slot attention 输出（gaussian init） | +0.5519 |
| slot attention 输出（learned init） | +0.9362 |
| **过 `_semantic_slots` 后（T=0.30，配置实际值）** | **+0.9987** |
| 过 `_semantic_slots` 后（learned init） | **+1.0000** |

8 个"全局坐标"在数值上是同一个向量。后果直接体现在运输计划上：

| 配置 | OT plan 相对均匀分布的总变差 |
|---|---|
| v3.3 现状 | **0.0577** |
| 删掉 `_semantic_slots` | 0.1798 |
| 删掉 + 跨 slot 中心化 | **0.6135** |

塌缩与 prototype 初始化尺度无关（std 从 0.02 试到 1.0 全部塌缩），也无法用
正交初始化或 Sinkhorn 双随机化修复 —— 根因是**任何跨 8 个 token 的 softmax
加权平均都趋向同一个均值向量**：高维 cosine 分数集中在 ±1/sqrt(D) 内，除以
温度后 softmax 接近均匀，输出因此都是同一批 local slot 的近似平均。

**这一条解释了 DCT 全部的历史现象**：

- score-first（关掉全部 OT 相关损失）反而最好 —— OT 分支本来就不携带患者
  特异信息，那些损失只是噪声；
- v3.6 listwise、v3.8 三损失、v4.1 ledger、ETAR 加什么都不涨 —— 机制都建在
  一个塌缩的表征上；
- v3.2 诊断里 `anchor_coverage` 恒为 1.0、组成熵接近最大值；
- best 与 last5 的 gap 常年 −0.10 以上 —— 预测靠 30M 参数记训练集，而不是
  靠结构。

复现脚本见第 7 节。

## 2. v3.9 的 idea：把反事实从"事后查询"变成"预测的定义"

v3.3--v3.8 的做法是回归一个自由的风险分数，再在评估期做 anchor 干预。
v3.9 改变预测对象本身：

> 患者自己的跨模态耦合 P，在低危队列的代价几何 C_L 下更省，还是在高危队列
> 的代价几何 C_H 下更省？

$$\lambda_s=\sigma\!\left(\frac{\langle P_s,C_{L,s}\rangle-\langle P_s,C_{H,s}\rangle}{\tau}\right),\qquad
\ell_s=(1-\lambda_s)h_{L,s}+\lambda_s h_{H,s}$$

最终 logits 是各阶段的 gate 加权和。反事实不再是附加分析，它就是坐标轴：
`λ=0` 表示"如果这名患者的运输结构变成低危队列的样子"，`λ=1` 是高危队列。

一句话卖点：**我们不回归风险分数，我们定位患者在低危—高危运输几何连续谱上
的坐标。** 主图是一条 λ 轴，两端是锚定运输计划热图，患者散布其间，配风险分层
KM 曲线。

### 三条结构性保证（恒等式，不是靠损失"鼓励"）

1. **预测有界**：默认无残差旁路时 logits 严格落在 `{h_L, h_H}` 的凸包内。
2. **方向与剂量单调**：`h_H = h_L + softplus(gap)` 保证逐时间箱 hazard 偏序，
   于是 risk 关于 λ 严格单调递增。**v3.8 用 TID/TDM 两个 margin 损失去换的
   方向一致性和剂量单调性，在这里是恒等式** —— 同时消除了 v3.8 那个自我削弱
   的逻辑漏洞（把方向写进训练目标后，测试集同方向就不能再当作解释忠实的证据）。
3. **运输几何只由 slot 差异决定**：跨 slot 中心化后 `sum_i v_i = 0`，两两内积
   之和严格等于 `-sum_i ||v_i||^2 < 0`。（各 slot 范数相等时两两余弦均值恰为
   `-1/(K-1)`；论文里的命题要写内积形式，余弦只是特例。）

三条都有对应单测数值验证。

### 为什么可能提分

- **自由度被压死**：主信号路径是 slot → cost → plan → λ → hazard，预测的
  自由度只有 4 个 λ 加上 `2 × stages × n_classes` 个锚定 hazard 参数。
  event Transformer 不再直接输出风险，只决定阶段权重。这直接打击历史上
  −0.13 的 best/last5 gap。
- **目标与评估对齐**：λ 是标量且单调对应 risk，IPCW pairwise ranking 作用在
  它上面与 C-index 完全同向。
- **弱化分箱敏感性**：λ 是连续坐标，不依赖 4 分箱的边界位置。

## 3. 相对 v3.3 删掉了什么

| 删除项 | 依据 |
|---|---|
| `_semantic_slots` + shared prototypes | 实测塌缩到余弦 0.9987、plan 偏离均匀仅 0.0577。索引稳定性改由 learned slot query 承担 |
| ETAR（`_etar_loss` 等 5 个参数） | v3.6 实测 BLCA 略好、LUAD/LUSC 更差，且是无结构保证的启发式 |
| geometry reliability（RTEM） | 从未启用过的分支 |
| evidence cost 注入 | 默认权重 0，从未启用 |
| `_project_coupling` 的 1000 次图内迭代 | log-domain Sinkhorn 本身已收敛到给定边缘，该投影只用于修补 `nan_to_num` 的破坏；默认降到 3 次且判据放进 `no_grad` |

模块是 `del` 掉的，不是权重置零 —— "关掉权重"和"移除模块"在论文口径上不是
一回事。单测断言 state_dict 里不再有对应键。

## 4. 实现中发现并修掉的一个真 bug

v3.9 把队列锚点拉进了训练前向路径（v3.3 只在评估分支读它），而
`_update_risk_anchors` 用 `lerp_`/`copy_` 原地更新那个 buffer。不做
`detach().clone()` 时，**第二个 batch 的 backward 会直接抛
RuntimeError**（张量版本被原地修改）。已修复并加了回归测试。

## 5. 温度标定：一个必须避开的失效模式

锚点代价差的实际尺度实测约 **0.01**（归一化 cost 下，std 0.011~0.016）。
温度取值错配会让方法直接学不动：

| τ | λ 范围 | λ 标准差 | 判断 |
|---|---|---|---|
| 0.25 | [0.466, 0.589] | 0.035 | 退化，几乎不区分患者 |
| 0.05 | [0.335, 0.859] | 0.154 | 健康 |
| 0.01 | [0.032, 1.000] | 0.352 | 接近饱和 |
| 0.002 | [0.000, 1.000] | 0.408 | 完全饱和，梯度消失 |

因此默认 `dct_v39_tau_init=0.02`，并默认开启 `dct_v39_tau_autoscale`：在队列
锚点就绪后的首个 batch，用代价差的标准差标定 τ，使初始 logit 落在约 ±1 的
高梯度区间。τ 本身仍可学习。

标定后的梯度健康检查（λ 通路确实回传到运输通路，与 v3.3 的塌缩形成对照）：

| 参数 | 梯度均值 |
|---|---|
| `stage_pair_cost.3.weight` | 6.47e-04 |
| `evidence_gate.3.weight` | 2.12e-04 |
| `slot_attention_wsi.slot_queries` | 6.46e-03 |
| `anchor_hazard_low` | 2.19e-02 |
| `v39_log_tau` | 5.62e-03 |

**训练期必须监控 `v39_lambda_std`**：若持续低于 0.02，说明 τ 过大或锚点未
分化，应减小 `dct_v39_tau_init`。这是为了避免重犯 ArcSurv 全零初始化那类
"跑很久才发现退化"的错误。

## 6. 创新边界（诚实标注）

**不能当创新点的**：learned slot query（BO-QSA/SurvQ 已有）、跨 slot 中心化
（标准去相关操作）、OT 用于 WSI+omics 生存（MOTCat）、prototype/pathway
对齐（MMP/ProtoPathway）、distributional counterfactual + OT（DISCOUNT）。
前两项在本方法中是**正确性修复**，不是贡献。

**可以主张的**：以"训练折删失风险集锚定的运输几何"为坐标系，把生存预测
参数化为该坐标上的凸组合，从而使方向一致性与剂量单调性成为结构恒等式而非
损失项。名称建议避开 `counterfactual`，用
**Censoring-Aware Risk-Simplex Transport**。

`λ` 是基于模型的运输敏感性坐标，不是可识别的因果治疗效应，论文不能写成
治疗建议。

## 7. 复现与运行

```bash
# 塌缩诊断（本文第 1、5 节的全部数字）
python tools/diagnose_transport_collapse.py
python tools/diagnose_transport_collapse.py --section pooling   # 只看池化方式
python tools/diagnose_transport_collapse.py --section methods   # 只看端到端

# 单测（13 项结构保证与回归）
python -m pytest tests/test_dct_v39_risk_simplex_transport.py -q

# 训练
python -m survot_rank.cli train --config configs/dct_v39_risk_simplex_transport_blca.yaml
```

## 8. 必须做的实验（按顺序，不要跳）

1. **BLCA fold0 + fold2，20 epoch 的三档对照**，这一步只回答"塌缩修复是否
   真的有用"：
   - `A` v3.3 原样（塌缩基线）
   - `B` v3.9 关闭中心化（`dct_v39_center_slots=false`，只删 prototype）
   - `C` v3.9 默认（删 prototype + 中心化 + λ 坐标）

   若 C 不优于 A，本方法的前提就不成立，**停止，不要扩癌种**。
2. **消融 λ 坐标 vs 自由回归**：`dct_v39_residual_scale` 取 0 / 0.3 / 1.0，
   回答"结构约束 vs 模型容量"。这张表本身是论文的一个卖点。
3. **坐标端点忠实性**：`audit_coordinate_endpoints()` 报告
   `λ(low) < λ(factual) < λ(high)` 的患者比例。这不是训练目标，因此是对坐标
   定义的独立检验。
4. **锚点冻结消融**：`dct_v39_anchor_freeze_epoch` 取 0 / 10，回答"λ 追逐
   移动目标"是否是问题。
5. 通过前 4 步后再上 3 seeds × 5 折 × 多癌种，并按仓库既有协议报告
   last5 而不是 per-fold best。

## 9. 已知风险

- **锚点是移动目标**：λ 的参考几何由 EMA 更新，训练早期会漂移。
  `dct_v39_anchor_freeze_epoch` 是准备好的缓解手段，但未实测。
- **容量可能不足**：把预测压到 4 个 λ 是很强的约束，可能欠拟合。
  `dct_v39_residual_scale` 是可控的放松旋钮，默认关闭以保持结构严格。
- **第一个 batch 的 λ 恒为 0.5**：锚点尚未就绪，属预期行为，但意味着首个
  batch 只有 gate 与锚定 hazard 参数获得梯度。
- **中心化会移除共模信息**：若某癌种的预后信号确实主要在共模分量里，
  中心化会有害。`dct_v39_center_slots=false` 是对照组，这个假设必须用第 8 节
  第 1 步的 B 档去检验，不能假定。
