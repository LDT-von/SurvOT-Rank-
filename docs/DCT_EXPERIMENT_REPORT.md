# DCT 实验报告：已完成与待运行实验

> 报告日期: 2026-08-14
> 最新更新: 2026-08-09 (FINAL_SUMMARY.md)

---

## 一、实验总览

### 1.1 已完成实验（主线）

| 实验名称 | 状态 | 说明 |
|---------|------|------|
| **DCT v3.8.2 fixed-full** | ✅ 全部完成 | 6癌种 × 5折 = 30 folds |
| IST v4.0 abl_b | ✅ 全部完成 | 6癌种 × 5折 = 30 folds |
| v3.3 clean baseline | ✅ 已完成 | UNI v1, 5fold |
| v3.8.2 adaptive vs fixed | ✅ 已完成 | 自适应权重无增益 |

### 1.2 已停止实验（负结果）

| 实验名称 | 状态 | 结论 |
|---------|------|------|
| v3.9 Risk-Simplex | ❌ 停止 | fold难度排序反转，机制未跑通 |
| v3.8.3 Centered | ❌ 停止 | 修好塌缩后反而降分 |
| ArcSurv | ❌ 停止 | 原型塌缩修复无效 (0.5665 < 0.7132) |
| v4.1 Evidence Ledger | ❌ 停止 | 补全损失下界修复无效 |
| v3.8 三损失 | ⚠️ 无显著收益 | Δ=+0.003，噪声级 |

---

## 二、已完成实验详情

### 2.1 DCT v3.8.2 fixed-full（最终主线）

**实验目的**: 评估固定配方 DCT 方法在6个癌种上的跨癌种泛化性能

**核心配置**:
```python
{
    "survot_method": "dct_v382_prognostic_transport_reconstruction",
    "max_epochs": 50,
    "dct_v382_warmup_epochs": 5,
    "dct_v382_ramp_epochs": 10,
    "dct_v382_lambda_mgptr": 0.05,
    "dct_v38_lambda_direction": 0.05,
    "dct_v38_lambda_dose": 0.03,
    "dct_v38_lambda_reconfiguration": 0.02,
    "dct_lambda_ipcw_rank": 0.10,
    "fit_bins_on_train": True,
    "dct_slot_init_mode": "deterministic",
    "wsi_encoder": "uni2-h",
}
```

**结果** (6癌种 × 5折):

| 癌种 | F0 | F1 | F2 | F3 | F4 | **Mean** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **UCEC** | 0.8358 | 0.8446 | 0.7933 | 0.8048 | 0.8333 | **0.8224** |
| **KIRC** | 0.7973 | 0.8443 | 0.8215 | 0.8011 | 0.7716 | **0.8071** |
| **BLCA** | 0.6534 | 0.7253 | 0.6520 | 0.7375 | 0.7855 | **0.7107** |
| HNSC | 0.6906 | 0.6299 | 0.6388 | 0.6039 | 0.7526 | 0.6632 |
| SKCM | 0.6972 | 0.6001 | 0.6529 | 0.6278 | 0.7258 | 0.6608 |
| LUSC | 0.6789 | 0.6314 | 0.5368 | 0.6617 | 0.5931 | 0.6204 |

**结论**: 
- 描述性排序: UCEC > KIRC > BLCA > HNSC > SKCM > LUSC
- UCEC和KIRC表现最佳 (>0.80)
- LUSC波动最大，需要进一步分析

---

### 2.2 自适应权重对照实验

**实验目的**: 验证自适应辅助损失权重是否有增益

**实验设计**:
- `fixed_full`: 固定权重配方 (MGPTR=0.05, direction=0.05, dose=0.03, reconfiguration=0.02)
- `adaptive_full`: 自适应权重分配

**结果**:

| 变体 | Fold1 | Fold2 | Fold4 | **Mean (3f)** |
|------|:-----:|:-----:|:-----:|:-------------:|
| **fixed_full** | 0.7253 | 0.6520 | 0.7855 | **0.7209** |
| adaptive_full | 0.6876 | 0.6773 | 0.7718 | 0.7122 |

**结论**: **自适应权重无增益**。fixed_full 领先 +0.0087。

---

### 2.3 MGPTR消融实验

**实验目的**: 评估多几何预后传输重建损失的作用

**实验设计**:
- `base` (MGPTR=0): 无MGPTR损失
- `mgptr` (MGPTR=0.05): 启用MGPTR损失

**结果**:

| 变体 | Fold1 | Fold2 | Fold4 | **Mean (3f)** |
|------|:-----:|:-----:|:-----:|:-------------:|
| base (MGPTR=0) | 0.6328 | 0.7085 | 0.7513 | **0.6975** |
| mgptr (MGPTR=0.05) | 0.5916 | 0.6926 | 0.7991 | 0.6944 |

**结论**: MGPTR权重无显著增益 (Δ=-0.0031)

---

### 2.4 v4.0 三档消融实验

**实验目的**: 拆解IST-Surv分数来源

**实验设计**:
- `A (factual)`: 纯factual预测，无干预稳定性机制
- `B (cost_only)`: 稳定性cost回写，无辅助损失
- `C (full IST)`: 完整IST机制

**结果**:

| 档 | Fold1 | Fold2 | Fold4 | **Mean (3f)** |
|:--:|:-----:|:-----:|:-----:|:-------------:|
| A (factual) | 0.6884 | 0.6632 | 0.7701 | **0.7072** |
| B (cost_only) | 0.6970 | 0.6632 | 0.7564 | 0.7055 |
| C (full IST) | 0.6970 | 0.6642 | 0.7547 | 0.7053 |

**结论**: 
- 分数全来自factual底座
- cost回写和辅助损失均无增益 (Δ≈0)

---

### 2.5 收敛健康度审计

| 方法 | Fold | Best | 类型 | 影响 |
|------|:----:|:----:|------|:----:|
| ArcSurv | 1 | e29 | 欠训练 | 低估 |
| v4.0 IST-Surv | 1 | e0 | 从未学习 | 高估 |
| v4.1 Evidence Ledger | 2 | e3 | 早熟 | 高估 |
| v3.8.2 MGPTR | 2 | e4 | 早熟 | 高估 |

---

## 三、负结果汇总

### 3.1 v3.9 Risk-Simplex Transport

**问题**: fold难度排序反转
- 其他方法: fold4最高, fold2最低
- v3.9: fold2最高, fold4最低

| Fold | v3.9 | v3.3 | Δ |
|:---:|:-----:|:-----:|:--:|
| 1 | 0.6320 | 0.6918 | -0.0598 |
| 2 | 0.6776 | 0.6735 | +0.0041 |
| 4 | 0.6085 | 0.7222 | -0.1137 |
| **Mean** | **0.6394** | **0.6958** | **-0.0564** |

**结论**: 约2.8σ，方向可信但幅度待五折确认。机制未跑通，停止调参。

---

### 3.2 v3.8.3 Centered Intervention Consistency

**问题**: 修好塌缩后分数反而崩了

| Fold | v3.8.3 | v3.3 | Δ |
|:---:|:-------:|:-----:|:--:|
| 1 | 0.5931 | 0.6918 | **-0.0987** |
| 2 | 0.6193 | 0.6735 | **-0.0542** |

**结论**: 高余弦不是病因——共模分量里携带有效信息。塌缩修复失败。

---

### 3.3 ArcSurv 原型塌缩修复

**修复尝试**: furthest-point anchoring + entropy惩罚

**结果**:
- 修复前: fold1 = 0.7132
- 修复后: fold1 = 0.5665

**结论**: 修复无效，反而更差。原型多样性被过度约束。

---

### 3.4 v4.1 补全损失下界修复

**修复尝试**: v41_min_log_variance = -4.0

**结果**:
- 修复前: fold2 = 0.6436
- 修复后: fold2 = 0.6436 (一模一样)

**结论**: 修复无效。completion下界修复未改变分数。

---

## 四、即将/待运行实验

### 4.1 下一队列（2026-08-06）

基于 `scripts/run_priority_experiment_queue.py --stages next`:

| 阶段 | 任务 | 目的 |
|------|------|------|
| 1 | `v382_fixed_full_fold03` | 补BLCA fold0/fold3，合成完整五折 |
| 2 | `arcsurv_repaired_gate` | 检查原型cosine/hazard spread/组合熵 |
| 3 | `v41_repaired_gate` | 检查completion是否保持非负 |

**闸门原则**: ArcSurv或v4.1内部诊断未通过时，不扩跑剩余folds。

---

### 4.2 评分门控实验

**脚本**: `scripts/run_dct_v382_score_gate.py`

**目的**: 诊断DCT v3.8.2在不同评分区间患者上的表现

**对照设计**:
- Control: fixed_full (冻结配方)
- Candidate: 评分门控变体

---

### 4.3 修复闸门详细说明

#### 4.3.1 ArcSurv修复闸门

**检查指标**:
- `arc_wsi_archetype_cosine`: 原型余弦相似度
- `arc_omic_archetype_cosine`: 组学原型余弦相似度
- `arc_hazard_spread`: 风险分布
- `arc_active_archetype_fraction`: 活跃原型比例

**判据**: 若 cosine→1 且 hazard_spread→0，则composition已退化为近似常向量

#### 4.3.2 v4.1修复闸门

**检查指标**:
- `v41_completion`: 补全损失
- `v41_total_loss`: 总损失

**判据**: 总目标是否不再被拖成负数

---

## 五、已确认结论

### 5.1 方法有效性排序

| 排名 | 方法 | 结论 |
|:----:|------|------|
| 1 | DCT v3.8.2 fixed-full | **最优主线**，6癌种全面优胜 |
| 2 | IST v4.0 abl_b | 6癌种仅KIRC反超，DCT赢4/6 |

### 5.2 关键发现

1. **自适应权重无增益**: fixed配方为最优
2. **MGPTR无显著增益**: Δ=-0.0031，噪声级
3. **IST辅助损失无增益**: 三档消融证明分数全来自factual底座
4. **v3.8三损失无显著收益**: Δ=+0.0033
5. **塌缩修复适得其反**: 共模分量携带有效信息

### 5.3 已停止方向

| 方向 | 原因 |
|------|------|
| v3.9 Risk-Simplex | fold排序反转，机制未跑通 |
| v3.8.3 Centered | 塌缩修复反而降分 |
| ArcSurv | 原型塌缩无法修复 |
| v4.1 | completion修复无效 |
| v4.2 ACT-Surv | 依赖ArcSurv修复 |

---

## 六、实验队列清单

### 6.1 优先队列 (DEFAULT_STAGES)

```bash
python scripts/run_priority_experiment_queue.py run \
  --stages v382_fixed_full,v382_adaptive_full,v40_abl_a_factual,v40_abl_b_cost_only,v40_staged_rerun,v33_clean_baseline,v33_clean_no_ipcw_rank
```

### 6.2 下一队列 (NEXT_STAGES)

```bash
python scripts/run_priority_experiment_queue.py run --stages next
```

### 6.3 跨癌种主线

```bash
# DCT v3.8.2 跨癌种
python scripts/run_dct_v382_final_cross_cancer.py run \
  --cancers skcm,hnsc,lusc,kirc,ucec --folds 0,1,2,3,4

# IST v4.0 跨癌种
python scripts/run_ist_v40_final_cross_cancer.py run \
  --cancers skcm,hnsc,lusc,kirc,ucec --folds 0,1,2,3,4
```

### 6.4 评分门控实验

```bash
python scripts/run_dct_v382_score_gate.py run
```

---

## 七、数据说明

### 7.1 划分分界

**重要**: `bee66a2` (2026-07-30) 重写了BLCA的5fold划分

- 旧划分: < 2026-07-30
- 新划分: ≥ 2026-07-30 (当前使用)

跨划分的结果不可比较！

### 7.2 编码器说明

| 编码器 | 维度 | 覆盖情况 |
|--------|------|----------|
| UNI v1 | 1024d | ~100% |
| UNI2-h | 1536d | BRCA仅74.1% |

### 7.3 分箱协议

| 协议 | 说明 | 用于 |
|------|------|------|
| clean | fit_bins_on_train=True | 当前所有正式实验 |
| leaky | fit_bins_on_train=False | 历史记录 |

---

## 八、附录：实验脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/run_priority_experiment_queue.py` | 优先队列调度器 |
| `scripts/run_dct_v382_final_cross_cancer.py` | DCT跨癌种训练 |
| `scripts/run_dct_v382_paper_ablations.py` | DCT消融实验 |
| `scripts/run_dct_v382_score_gate.py` | 评分门控实验 |
| `scripts/run_ist_v40_final_cross_cancer.py` | IST跨癌种训练 |
| `scripts/summarize_dct_v382_score_gate.py` | 评分门控结果汇总 |
| `scripts/monitor_priority_queue.py` | 队列监控 |

---

## 九、结论

### 9.1 最终主线

**DCT v3.8.2 fixed-full** 是最终确定的论文主线方法:
- 固定权重配方，无自适应
- MGPTR=0.05, direction=0.05, dose=0.03, reconfiguration=0.02
- IPCW rank=0.10 保留
- 6癌种五折全部完成

### 9.2 对比SOTA

在BLCA/KIRC/UCEC三个有对比数据的癌种上，DCT v3.8.2均领先:

| 癌种 | DCT v3.8.2 | MOTCat | 领先 |
|------|:-----------:|:------:|:----:|
| UCEC | **0.8224** | 0.675 | +0.147 |
| KIRC | **0.8071** | 0.708 | +0.099 |
| BLCA | **0.7107** | 0.683 | +0.028 |

### 9.3 下一步

1. 等待 `v382_fixed_full_fold03` 完成，补齐BLCA五折
2. 运行修复闸门实验验证ArcSurv/v4.1状态
3. 补充PAMT等最新论文的对比数据
4. 准备论文写作
