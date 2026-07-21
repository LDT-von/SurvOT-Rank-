# DCT v3.5：网络与损失受控筛选

> 状态：代码已就绪，等待服务器结果。开发阶段固定只跑 fold 0/2；最终候选仍须补齐 5-fold。

## 1. 为什么停止 v3.4

DCT v3.4 同时启用了 25% 事件有放回采样、`alpha_surv=2/3` 和 64-case
rank memory。BRCA fold0 的自然 DSS 事件率约 8.4%，三项叠加大幅改变了原始队列目标，
并重复使用稀有事件患者。旧 Slot Attention 还会在 `eval()` 中重新采样随机 slots，导致
同一 checkpoint 重评不一致。

v3.5 固定以下共同协议：

- batch size 8；
- survival bins 仅在当前训练折拟合；
- 患者无放回，每轮恰好使用一次；
- 事件尽量分散到不同 batch，但不改变队列事件率；
- `alpha_surv=0.15`；
- `NLL + 0.10 * IPCW ranking`；
- IPCW rank memory 关闭；
- 所有旧 OT/anchor/stage-risk/coordinate 辅助损失关闭；
- 50 epochs、seed 3、AdamW、学习率 `5e-4`；
- 正式结果与 smoke 结果完全隔离。

## 2. 四个版本

| 版本 | 唯一研究变量 | 目的 |
|---|---|---|
| v3.5R | 训练时随机、验证时固定且互异的 slot 初值 | 正确性基线 |
| v3.5Q | R 的固定初值改为每个 slot 独立 learned query | 检验随机可交换 slots 是否是瓶颈 |
| v3.5G | R + evidence marginal strength `0.25` | 检验无约束 evidence marginals 是否过拟合 |
| v3.5L | R + projection 128、Transformer 1 层 | 检验 30M 级容量是否过大 |
| U（2026 follow-up） | R + RTEM geometry reliability | 检验跨 OT 几何冲突时是否应收缩 evidence marginals |
| M（2026 follow-up） | R + epoch 内 IPCW memory 64 | 检验小 batch/稀事件风险集是否需要跨 batch 上下文 |

第一阶段禁止组合 Q/G/L。单变量胜出后，才允许增加组合版本。

## 3. 服务器运行

先拉取并检查：

```bash
git pull
python3 scripts/run_dct_v35_screen.py doctor
python3 scripts/run_dct_v35_screen.py plan --variants r --cancers all --folds 0,2
```

先做一次独立 smoke，不会污染正式目录：

```bash
python3 scripts/run_dct_v35_screen.py smoke --variants r --cancers brca --folds 0 --gpu 0
```

第一轮先运行修复基线，覆盖十癌种 fold0/2：

```bash
mkdir -p logs
nohup python3 -u scripts/run_dct_v35_screen.py run \
  --variants r --cancers all --folds 0,2 --gpu 0 \
  > logs/dct_v35_r_all_fold02.log 2>&1 &
```

R 完成并检查后，再逐个运行 Q/G/L，避免一次排队 80 folds：

```bash
nohup python3 -u scripts/run_dct_v35_screen.py run --variants q --cancers brca,ucec,blca,luad --folds 0,2 --gpu 0 > logs/dct_v35_q_phase1.log 2>&1 &
nohup python3 -u scripts/run_dct_v35_screen.py run --variants g --cancers brca,ucec,blca,luad --folds 0,2 --gpu 0 > logs/dct_v35_g_phase1.log 2>&1 &
nohup python3 -u scripts/run_dct_v35_screen.py run --variants l --cancers brca,ucec,blca,luad --folds 0,2 --gpu 0 > logs/dct_v35_l_phase1.log 2>&1 &
```

脚本默认跳过已有 `split_<fold>_results_final.pkl` 的正式 fold。确需覆盖时显式加
`--force`。

## 4. 输出目录

```text
results/dct_v3.5_screen/{r,q,g,l}/{cancer}/
results/dct_v3.5_smoke/{r,q,g,l}/{cancer}/
```

U/M 是后续单变量诊断，不是新的论文方法，也不应早于 R/Q/G/L 运行。对应结果目录沿用同一规则：`results/dct_v3.5_screen/{u,m}/{cancer}/`。

各 fold 会保存 best checkpoint、patient results、epoch curve、partial summary 和日志。

## 5. 入选规则

开发期不能只按两个 fold 的最高点挑版本。至少同时满足：

1. 同一 checkpoint 重评完全一致；
2. fold0/2 mean best C-index 高于或不劣于 R；
3. `best - last5` gap 不恶化；
4. 无 NaN/Inf；
5. evidence marginal entropy、IPCW pair 数没有异常塌缩；
6. BRCA/UCEC 提升不能以明显损害 BLCA/LUAD 为代价。

fold0/2 只负责淘汰。最终版本需跑完整 5-fold，并补充 NLL-only 对照、三种子、置信区间
和结构消融。
