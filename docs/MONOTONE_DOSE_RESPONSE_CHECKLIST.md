# Monotone Dose-Response: 闭环 checklist（未交付）

> **状态：当前未闭环。**
> 本文档不是计划书，是诚实记账。
> 它列出"声称 → 支撑 → 验证"链条中**今天缺失的环节**，以及在不编造数据的前提下如何补齐。

## 0. 当前状态（2026-08-15）

| 环节 | 完成 | 备注 |
|---|---|---|
| (A) 代码：minimal 配方 | ✅ | `DCTV382MonotoneDoseResponse`，强制覆盖 4 个系数为 0 |
| (B) 单元测试：配方语义 | ✅ | `tests/test_dct_v382_minimal_transport.py`，10/10 pass |
| (C) 跨癌症 launcher | ✅ | `scripts/run_dct_v382_minimal_cross_cancer.py`，可 plan / doctor / smoke |
| (D) 论文 ablation launcher | ⚠️ 部分 | `scripts/run_dct_v382_paper_ablations.py` 已存在，**但**它基于 `fixed-full`；基于 `minimal` 的版本还需要写 |
| (E) audit loader | ✅ | `scripts/audit_dct_v382.py`，可输出 `direction_consistency / reconfiguration / dose_monotonicity` |
| (F) evidence ledger 数据 | ❌ | `PAPER_EVIDENCE_LEDGER.md` 的 A/B/C 节表格全部为空 |
| (G) 闭环节实证：no_direction ablation Δ ≥ ? | ❌ | 尚未跑 |
| (H) 闭环节实证：audit direction_consistency 数值 | ❌ | 尚未跑 |
| (I) 闭环节决策树：阈值 | ❌ | 还不知道 Δ 阈值该定多少 |

## 1. 闭环的最小证据需求

要让 Monotone Dose-Response 的科学 claim 可证伪且已证，至少需要以下事实：

### 1.1 negative-control ablation（4 个 cell）

在 BLCA × fold{1,2,4} 上跑：

| 变体 | 该把哪个系数 → 0 | 其它系数 | 预期方向 |
|---|---|---|---|
| `no_direction` | `dct_v38_lambda_direction` | 其余继承 minimal | C-index 应下降 ≥ Δ_min |
| `no_ipcw_rank` | `dct_lambda_ipcw_rank` | 其余继承 minimal | C-index 应下降 ≥ Δ_min |
| `no_mgptr` | `dct_v382_lambda_mgptr` | 其余继承 minimal | C-index 应**不显著变化**（阴性格） |
| `no_dose` | `dct_v38_lambda_dose` | 其余继承 minimal | 同上 |
| `no_reconfiguration` | `dct_v38_lambda_reconfiguration` | 其余继承 minimal | 同上 |

**统计要求**：3 个 fold 都要做，才能汇报 mean ± std。

**阈值 Δ_min 定多少**：取决于同类 SOTA noise floor。建议：
- 起点：`Δ_min = 0.005`
- 若 no_direction 的平均 Δ ∈ [0.001, 0.005]：触发"样本量不足"决策
- 若 no_direction 的平均 Δ < 0.001：重新审视 claim 是否成立

### 1.2 audit loader 真实数值（BLCA × fold{1,2,4}）

对 minimal 配方的 best checkpoint 跑 `audit_dct_v382.py audit + sweep`：

| metric | 期望 | 阈值 |
|---|---|---|
| `direction_consistency.correct_rate` | ≥ 0.55 | 这是上面方向损失的目标 |
| `direction_consistency.chance_gap` | ≥ 0.05 | 高于 chance 5pp |
| `dose_monotonicity.monotone_rate` | ≥ 0.70 | 对 alpha sweep 单调 |
| `reconfiguration.mean_tv` | ≥ 0.02 | 干预真在改动几何 |

### 1.3 minimal vs fixed-full head-to-head

在**同一 cancers × folds** 上跑两个配方，比 C-index：

| 比较 | 期望 |
|---|---|
| `minimal.C-index` ≈ `fixed_full.C-index` | | Δ ≤ 0.005 |
| `minimal` 训练更快 | 单 fold 节省 ~15% GPU 时间（少了一次额外 Sinkhorn） |

## 2. 还需补的代码（无 GPU 也能做）

### 2.1 基于 `minimal` 的 ablation launcher

参考 `scripts/run_dct_v382_paper_ablations.py`，应当：

- 继承 `run_dct_v382_minimal_cross_cancer.FROZEN_MINIMAL_OVERRIDES` 而不是 `fixed_full` 的
- 同样的 5 个 ablation terms
- 写 BLCA × fold{1,2,4}
- 输出到 `results/dct_v3.8.2_monotone_ablations/robust/<variant>/blca/`

**预计工作量**：~150 行，复制已有 launcher 改冻点即可。

### 2.2 evidence ledger 自动填充脚本

读 `split_<fold>_results_final.pkl` + `audit_metrics_*.json` + `sweep_metrics_*.json`，自动写到 `PAPER_EVIDENCE_LEDGER.md`。

**预计工作量**：~80 行，复用 `tmp/score_summary.py` 模板。

### 2.3 守护测试：ledger 行非空

阻止 PR 在表格为空时合并：

```python
def test_minimal_ledger_table_is_filled():
    rows = parse_ledger_table(...)
    assert all(row.cindex_is_finite_number() for row in rows)
    assert not any(row.note == "run_completed_but_table_not_updated" for row in rows)
```

**预计工作量**：~30 行。

## 3. 没 GPU 时我能立刻做的

下面这些都不需要 GPU，今天就能写完：

| 序号 | 任务 | 状态 |
|---|---|---|
| W1 | 把本日对话中承认的"过度措辞"回退 | 已在对话中完成 |
| W2 | 写本 checklist doc | 当前文件 |
| W3 | 改 `PAPER_EVIDENCE_LEDGER.md` 表格为明确"待跑"标记（不再假装"未观察到"） | 待办 |
| W4 | 让 `run_dct_v382_paper_ablations.py` 加一个 `--base-launcher` 选项指向 `run_dct_v382_minimal_cross_cancer` | 待办 |
| W5 | 写 evidence ledger 自动填充脚本骨架 | 待办 |
| W6 | 写 ledger 守护测试 | 待办 |

## 4. 必须有 GPU 才能做的

| 序号 | 任务 | 期望 GPU 时间 |
|---|---|---|
| G1 | `run_dct_v382_minimal_cross_cancer.py run` 跑 5 癌种 × 5 折 = 25 个 job | ~25h @ 单卡 |
| G2 | 基于 `minimal` 的 ablation launcher × {1,2,4} folds × 5 variants = 15 job | ~15h @ 单卡 |
| G3 | 对 minimal 的 3 个 fold best checkpoint 跑 audit + sweep | ~3h |
| G4 | 对 fixed-full 的 3 个 fold best checkpoint 跑 audit + sweep（同 G3 重新复测） | ~3h |

合计 ~46 小时 GPU = 约 6 个工作日（1× A100 串行）。

## 5. 闭环决策树

跑完上面的 G1-G4 之后，应当按下列流程判定：

```
跑完 G2 (monotone ablation 15 job)
    ↓
no_direction 的 mean Δ = X
    ↓
   X ≥ 0.005?
       │
       ├─ YES → direction 是真的 load-bearing，进入 G3
       │         ↓
       │     direction_consistency.correct_rate = Y
       │         ↓
       │         Y ≥ 0.55?
       │             ├─ YES → ✅ claim 闭环
       │             └─ NO  → ❌ claim 弱化，改为"在 X 阈值上单调"
       │
       └─ NO  → ❌ claim 不成立，改为"minimal 与 fixed-full 在统计上不可分"
```

## 6. 今日交付物的诚实清单

| 路径 | 状态 | 价值 |
|---|---|---|
| `survot_rank/research/methods/dct_v382_minimal_transport/model.py` | ✅ 改名 | 工程价值 |
| `survot_rank/research/methods/catalog.py` | ✅ 改名 + 加别名 | 工程价值 |
| `configs/dct_v382_minimal_transport_blca.yaml` | ✅ | 工程价值 |
| `scripts/run_dct_v382_minimal_cross_cancer.py` | ✅ | 工程价值（无数据时 = 空 shell） |
| `tests/test_dct_v382_minimal_transport.py` | ✅ 10/10 | 工程价值（语义测试、非结果测试） |
| `PAPER_EVIDENCE_LEDGER.md` | ⚠️ 表格空 | 不知道 claim 真实与否 |
| `docs/MONOTONE_DOSE_RESPONSE_CHECKLIST.md` | ✅ 本文 | 反"编造证据" |

**今日不交付**：科学闭环。要的科学闭环需要 G1-G4 的 GPU 时间。
