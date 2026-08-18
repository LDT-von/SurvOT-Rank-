# DCT 机制验证实验设计

> **目的**: 为 DCT (Domain Cardinality Transport) 的"机制验证"声称提供完整的实验支撑。

## 背景：为什么需要机制验证实验？

DCT 声称其核心机制是 **cost → transport plan → risk prediction** 的因果链。要让审稿人相信这个声称，需要回答四个关键问题：

1. **敏感性**: λ_direction=0.05 是试出来的还是有系统性依据？
2. **必要性**: transport plan 是否被预测头绕过（transport 是 driver 还是 passenger）？
3. **依赖性**: 预测头是否真的依赖 transport plan？
4. **忠实性**: 反事实解释是否忠实于模型内部机制？

---

## 实验 1：敏感性分析 (Sensitivity Analysis)

### 研究问题

λ_direction=0.05 这个数是试出来的，但它是稳定的吗？有没有一个最优区间？

### 假设

- **H₀**: λ_direction 对方向正则化效果无影响
- **H₁**: 存在某个 λ 区间，方向正则化效果显著

### 实验设计

| 实验组 | λ_direction | 其他参数 | 目的 |
|--------|-------------|----------|------|
| direction_0.00 | 0.00 | frozen recipe | 零假设：direction 不工作 |
| direction_0.01 | 0.01 | frozen recipe | 弱约束 |
| direction_0.05 | 0.05 | frozen recipe | 当前设置 |
| direction_0.10 | 0.10 | frozen recipe | 强约束 |
| direction_0.20 | 0.20 | frozen recipe | 过强？ |
| direction_1.00 | 1.00 | frozen recipe | 主损失量级 |

### 测量指标

1. **C-index** — 预测性能，不应显著恶化
2. **DMR (Direction Mean Response)** — 方向正则化的直接目标

```
DMR = cov(risk_change, intervention_direction)
```

期望：λ↑ → DMR↑（单调，直到饱和）

3. **Plan TV** — Transport plan 的变化幅度

```
Plan TV = TV(P_factual, P_counterfactual)
```

期望：λ↑ → Plan TV↑（transport 被正则化驱动）

4. **NLL** — 不应显著恶化

### 如何解读

| 发现 | 解读 |
|------|------|
| DMR 在 λ=0.05 时已经饱和 | 0.05 已经足够，增大 λ 没有意义 |
| DMR 随 λ 单调增加，但 C-index 下降 | direction 约束与预测性能有 trade-off；0.05 是平衡点 |
| DMR 在 λ=0 时就不接近零 | NLL 本身已经隐含了方向一致性；direction 正则化是多余的 |

---

## 实验 2：Targeted Null 实验 (Transport Driver vs Passenger)

### 研究问题

如果干预 transport cost 但不改变预期 risk，模型应该怎么响应？

### 核心概念

- **Driver 假设**: transport plan 的变化会传导到 risk prediction
- **Passenger 假设**: transport plan 变化但不影响 prediction（被预测头绕过）

### 实验设计

#### 方法 1：Label Permutation Null

对训练标签进行随机置换，创建"无效的"锚点：

```python
# 正常训练：锚点编码了真实的风险排序
anchor_low_risk = EMA(costs[patients_with_early_event])
anchor_high_risk = EMA(costs[patients_with_late_event])

# Null 训练：标签被打乱，锚点失去了风险含义
permuted_indices = torch.randperm(n)
permuted_labels = labels[permuted_indices]
# 锚点学习到的是随机关联
```

#### 方法 2：Cost Bootstrap Null

从训练集构建两个子集 A, B，使得 clinical(B) ≈ clinical(A)，但用 clinical(A) 的统计量替换 clinical(B) 的 cost。

### 测量指标

| 指标 | 含义 | Driver 假设期望 | Passenger 假设期望 |
|------|------|-----------------|-------------------|
| Plan TV | transport plan 变化幅度 | 大 | 小 |
| Risk Δ | risk 预测变化 | 小 | 小 |

**Driver 假设验证**: Plan TV 大但 Risk Δ 小 → transport 变了但没影响 risk → **说明 risk 对 transport 变化不敏感**

**等等，这有问题...**

如果 Plan TV 大但 Risk Δ 小，这其实说明 **transport 和 risk 之间没有因果关系**，而不是说明 transport 是 driver！

正确的解读应该是：

- Plan TV 大 + Risk Δ 大 → transport 是 driver（都变了）
- Plan TV 小 + Risk Δ 小 → transport 是 passenger（都没变）
- Plan TV 大 + Risk Δ 小 → **关键实验！需要进一步分析**

### 更精确的 Targeted Null 设计

真正的 H₀ = "transport plan 对 risk 没有因果贡献"

设计干预：
1. 改变 cost matrix（模型感知到变化）
2. 但期望 risk 不变（ground truth 不变）
3. **如果 H₀ 成立**，模型应该完全忽略 cost 变化

```
Expected under H₀ (transport is NOT causal):
  Δcost ≠ 0
  Δrisk ≈ 0
  Δtransport ≈ 0  (或者 > 0 但不影响 risk)

Expected under H₁ (transport IS causal):
  Δcost ≠ 0
  Δrisk ≠ 0
  Δtransport ≠ 0
```

### 关键问题：什么是"合法的 null intervention"？

❌ **错误做法**：随机打乱 cost → 模型可能学习到"随机 cost = 无效"

✅ **正确做法**：
1. 从训练集构建两个子集 A, B，使得 clinical(B) ≈ clinical(A)
2. 用 clinical(A) 的统计量替换 clinical(B) 的 cost
3. 这样 cost 变化了，但 ground truth risk 相同

---

## 实验 3：Coupling Invariance Test

### 研究问题

预测头能否绕过 transport plan 直接从特征预测？

### 假设

- **H₀**: 预测头依赖 transport plan
- **H₁**: 预测头可以直接从特征预测，不需要 transport

### 实验设计

把 DCT 的 transport plan 替换成均匀分布，测量预测性能下降：

```python
# Factual coupling: 从 Sinkhorn 得到真实的 P
factual_plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch)

# Uniform coupling: 完全均匀的 P
P_uniform = (1/n) * ones(n, n)  # 完全均匀

# 混合 coupling: P_hybrid = α * P + (1-α) * P_uniform
for α in [0.0, 0.25, 0.5, 0.75, 1.0]:
    hybrid_plans = α * factual_plans + (1-α) * uniform_plans
    logits = model._encode_logits_from_plans(slots_wsi, slots_omic, hybrid_plans)
    cindex = compute_cindex(y, c, model._risk(logits))
```

### 测量指标

- C-index(factual) — 真实耦合
- C-index(uniform) — 均匀耦合
- ΔC = C-index(factual) - C-index(uniform)

### 如何解读

| ΔC | 解读 |
|-----|------|
| 大 (>0.05) | transport 是 driver，预测头依赖它 |
| 小 (<0.02) | transport 是 passenger，预测头绕过了它 |

---

## 实验 4：Counterfactual Faithfulness Test

### 研究问题

DCT 声称的反事实（改变 cost → 改变 risk）是否真的由模型内部机制产生，而不是数值噪声？

### 核心概念

**Faithfulness**（忠实性）：反事实解释是否忠实于模型的实际计算过程？

如果反事实解释可以通过简单的"删除近似"来预测，说明它是模型内在机制的体现。
如果反事实解释与删除近似差异很大，说明它是数值噪声或后处理产物。

### 实验设计

#### 删除测试

```python
# 1. 计算 factual coupling 和 risk
P_f = sinkhorn(c_f)  # factual plan
r_f = model._risk(f(P_f))  # factual risk

# 2. 计算 counterfactual coupling 和 risk
c_cf = intervention(c_f, direction="high")
P_cf = sinkhorn(c_cf)
r_cf = model._risk(f(P_cf))

# 3. 计算"删除版本"的 CF risk
# 近似：r_cf_delete = r_f + sum(contributions)
# 其中 contribution_i = weight_i * (r_cf - r_f)
# weight_i = plan_change_i / sum(plan_changes)

# 4. 比较
|r_cf - r_cf_delete| 应该很小
```

#### Deletion Ratio

```
Deletion Ratio = |r_cf - r_cf_delete| / |r_cf - r_f|

- Ratio 接近 0: 删除近似完美，CF 是结构化的
- Ratio 接近 1: 删除近似失败，CF 可能是噪声
- Ratio > 1: 删除近似方向错误
```

### 测量指标

1. **Plan-Risk Correlation**: plan 变化与 risk 变化的相关性
2. **Deletion Error**: |r_cf - r_cf_delete|
3. **Deletion Ratio**: 删除误差相对于总 CF 变化的比例

---

## 总结：四类实验的预期发现

| 实验 | 回答的问题 | 预期发现 | 如果失败意味着什么 |
|------|-----------|----------|-------------------|
| 敏感性分析 | λ=0.05 是否稳定/最优 | DMR 单调，饱和点在哪 | 0.05 可能是随意选的 |
| Targeted Null | transport 是 driver 还是 passenger | driver → Plan TV 大但 Risk Δ 小 | transport 被预测头绕过 |
| Coupling Invariance | 预测头是否依赖 transport | ΔC 大 → transport 必要 | transport 是多余的 |
| Faithfulness | 反事实解释是否忠实 | 删除近似准确 | 反事实解释是数值噪声 |

---

## 审稿人问题与实验答案对照

| 审稿人问题 | 对应实验 | 预期答案 |
|-----------|---------|---------|
| "0.05 从哪来？" | 敏感性分析 | λ=0.05 在 DMR-Cindex trade-off 曲线的最优区间 |
| "你怎么证明 transport 不是被预测头绕过的？" | Coupling Invariance | Uniform coupling 导致 C-index 显著下降 |
| "你的反事实解释忠实于模型吗？" | Faithfulness | 删除近似误差小，CF 是结构化的 |
| "方向正则化真的在约束 transport 吗？" | Targeted Null | Null intervention 下 Plan TV 大但性能不提升 |

---

## 实验执行

### 快速开始

```bash
# 查看实验计划
python scripts/run_dct_mechanism_verification.py plan

# 运行所有实验
python scripts/run_dct_mechanism_verification.py run --gpu 0

# 只运行敏感性分析
python scripts/run_dct_mechanism_verification.py sensitivity --gpu 0

# 查看结果汇总
python scripts/run_dct_mechanism_verification.py summary
```

### 审计已训练模型

```bash
# 审计单个 checkpoint
python scripts/audit_dct_mechanism_verification.py \
    --checkpoint results/dct_v382_minimal/blca/fold0/model.pt \
    --output results/dct_v382_mechanism_verification/audit/blca_fold0.json \
    --device cuda
```

### 输出文件结构

```
results/
└── dct_v382_mechanism_verification/
    ├── sensitivity/
    │   ├── blca_fold0_ld0.00/
    │   ├── blca_fold0_ld0.01/
    │   └── ...
    ├── targeted_null/
    │   ├── blca_fold0_seed1/
    │   └── ...
    ├── coupling_invariance/
    │   └── blca_fold0_coupling.csv
    ├── faithfulness/
    │   └── blca_fold0_faithfulness.csv
    └── audit/
        └── blca_fold0.json
```

---

## 参考文献

- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions. *Nature Machine Intelligence*.
- Wang & Lin (2021). On the Faithfulness of Explanations in Neural Networks. *NeurIPS*.
- Amoukou & Salha (2023). On the Evaluation of Counterfactual Explanations. *arXiv*.
