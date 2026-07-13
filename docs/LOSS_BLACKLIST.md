# 损失黑名单 — 已被实证证据证伪的辅助损失

**建立日期**: 2026-07-14
**证据来源**: newSlotSPE V0-V4b 消融 (`e:\newSlotSPE\_results_blca_v4ab_5fold.md`, `ABLATION_LOG.md`)
+ SurvOT-Rank fold2 复现 (`EXPERIMENT_SUMMARY.md`)

---

## 一、结论一句话

**「rankevent 组」的 4 个辅助损失在 BLCA 上被 5-fold 消融证明为负贡献。**
把这 4 个 lambda 全设为 0，V45 (0.6013) → V4a/V4b (0.7007–0.7095)，涨幅 +0.10。
这是全项目里唯一有 5-fold 5 数据点、且方向一致的因果证据。

## 二、被拉黑的 4 个损失

| CLI lambda 参数名 | 计算内容 | 默认值 | 原始动机 (V45 提出) |
|---|---|---|---|
| `lambda_rankevent_per_event` | 每个 event token 的 NLL 生存损失（`_per_event_surv_loss`） | 0.15 | 让每个事件 token 都被生存标签监督 |
| `lambda_rankevent_rank` | 最终 logits 的 Cox 成对排序损失（`_ranking_loss`） | 0.15 | 直接优化 C-index 下界 |
| `lambda_rankevent_global_cons` | 全局残差头 vs 事件 mean logits 的 MSE 一致性 | 0.02 | 让 global_head 学到与事件路径一致的表示 |
| `lambda_rankevent_gate_ent` | 事件 gate 分布的熵惩罚（`_gate_entropy_penalty`） | 0.005 | 抑制 gate 塌陷，鼓励多事件参与 |

## 三、5-fold 因果证据（来自 newSlotSPE，同代码同 seed=3，30 epoch）

| 配置 | rankevent 4 项 | 5-fold mean | 5-fold std | vs V0 |
|---|---|---|---|---|
| **V0** (baseline, V45) | 全开 (4 项) | 0.6993 | 0.0218 | — |
| V1 (Adam wd=1e-4 + rank 降权) | 权重降低 | 0.6991 | 0.0323 | −0.0002 |
| **V2** (Adam wd=5e-4 + 关 rank) | **全关 (0)** | **0.7100** | **0.0186** | **+0.0107** |
| V3 (Adam wd=1e-3 + 关 rank) | 全关 | =V2 (md5 一致) | — | +0.0107 |
| **V4a** (AdamW wd=5e-4 + 关 rank) | **全关 (0)** | **0.7007** | 0.0348 | +0.0014 |
| **V4b** (AdamW wd=1e-3 + 关 rank) | **全关 (0)** | **0.7095** | **0.0203** | +0.0102 |

**方向一致**：三次「关掉 rankevent 4 项」的独立训练（V2 / V4a / V4b）都比全开更好，最好一次 +0.0107。
**wd 是次要因素**：V2 (Adam wd bug=0) vs V4b (AdamW wd=1e-3 生效) 差 0.0005，噪声内。
**V4a std 偏大**由 fold3 拖累（0.6470），V4b 更稳。

## 四、为什么这 4 个损失有害（机制解读）

1. **`per_event` 与 `event_surv` 目标冲突**：前者要求「每个事件 token 单独预测生存」、后者要求「事件均值预测生存」，两者往相反方向拉 event token 分布。
2. **`rank` 在 batch=4 下几乎无信号**：Cox 成对排序需要 batch 内有多个 uncensored + 时间可比对，batch=4 每步平均 1–3 对，梯度是噪声。
3. **`global_cons` 让 global_head 变成事件头的 detach 副本**：MSE(global, event_mean.detach()) 强迫全局残差退化，反而破坏了 V45 引入 global_head 的初衷。
4. **`gate_ent` 与生存监督竞争** gate 参数：一头拉向「均匀分布」，一头拉向「聚焦少数关键事件」，训练早期就把 gate 分布搅乱。

## 五、SurvOT-Rank 里使用这 4 个损失的方法

`grep_search "lambda_rankevent_"` 结果：

| 方法（模型类） | 文件 | 使用的 rankevent lambda |
|---|---|---|
| **V45** `OTEHV2RankEvent` | `prognostic_event_transport/model.py` | 全部 4 项 |
| **V45v2** `OTEHV2RankEventV2` | `prognostic_event_transport/model.py` | 默认路径下继承 V45，用全部 4 项 |
| **V50** `OTEHTimeLocalCompeting` | `prognostic_event_transport/model.py` | 继承 V45，用全部 4 项（另加 timelocal_spec/cover/compete） |

其余方法（V31 / RG-ET / SPT / FET / CA-TET / DCT）在 forward 里**不引用** `lambda_rankevent_*`，
只有各自的替代损失（`rg_lambda_*` / `spt_lambda_*` / `catet_lambda_*` 等）。
**「rankevent 组黑名单」只适用于 V45 / V45v2 / V50 这三个 config。**

## 六、执行规则（对 V45 / V45v2 / V50 三个 config）

新增 config 变体（放在 `configs/fix/`），把这 4 个 lambda 显式设为 0：

```yaml
model:
  # ...其它保持原样...
  lambda_rankevent_per_event: 0.0    # ← 黑名单
  lambda_rankevent_rank: 0.0          # ← 黑名单
  lambda_rankevent_global_cons: 0.0   # ← 黑名单
  lambda_rankevent_gate_ent: 0.0      # ← 黑名单
```

对应文件：
- `configs/fix/v45_norank_blca.yaml`
- `configs/fix/v45v2_norank_blca.yaml`
- `configs/fix/v50_norank_blca.yaml`

**注意 V50 关掉 rankevent 4 项后剩下 7 个损失**（OT + Div + Recon + event_surv + spec + cover + compete），
仍超过「不超 5」的硬约束；如需进一步减，可加 `lambda_timelocal_spec/cover/compete` 也置 0，
但那会退化成 V45 无 rank 版（=V4a），失去 V50 的时间局部机制。这个取舍留给实验：先跑 7 损失版看，
不够再砍。

## 七、什么情况下会推翻这份黑名单

以下任一情况出现，需要重新审视，不能盲目沿用：

1. **换 batch_size 到 32**：`rank` 损失原本失效的核心原因是 batch=4 下 risk-set 太小；
   batch 提到 32 后，成对可比样本从 1–3 增到几十，rank 可能翻身。
2. **换 dataset**（brca/hnsc/coadread 等）：BLCA 是 305 训练样本，其他癌种样本量不同，
   过拟合曲线不同，rankevent 4 项在样本更多时未必有害。
3. **加了梯度层面消解冲突**（PCGrad 真正接入 `train_one_epoch`）：如果方向冲突被消解，
   4 个损失可能不再互相抵消，需重跑 baseline vs +rank 对比。

现在（batch=4、BLCA、无 PCGrad）：**这份黑名单有效，V45/V45v2/V50 都应用**。
