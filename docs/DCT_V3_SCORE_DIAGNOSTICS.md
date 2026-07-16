# DCT v3 分数诊断

这套入口用于回答一个窄问题：DCT v3 的 anchor、共享坐标和 evidence-conditioned cost，分别是在提高真实泛化分数，还是只改变了某一折的峰值。它冻结当前主线的训练配方（50 epochs、batch size 16、seed 3、AdamW），不改正式 DCT 默认配置。

## 四个对照

| 变体 | 与 full 的唯一区别 | 要检验的问题 |
| --- | --- | --- |
| `full` | 无 | 当前 v3 完整机制 |
| `no_anchor` | `dct_lambda_anchor=0` | IPCW anchor 是否真的带来分数增益 |
| `no_coordinate` | `dct_lambda_coordinate=0` | 全局 prototype 坐标约束是否压低主任务 |
| `no_evidence_cost` | `dct_evidence_cost_weight=0` | evidence 同时改 cost/marginal 是否过强 |

默认只跑 fold 0、2、3：0/3 是已知不稳定折，2 是相对稳定折。共 12 次训练；它是定位瓶颈，不是正式报告。三折信号清楚后，才让胜出的版本跑完整五折。

## 运行

先确认环境和数据路径：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_dct_v3_score_diagnostics.ps1 -Mode doctor
```

先做一个单 epoch 冒烟检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_dct_v3_score_diagnostics.ps1 -Mode smoke
```

跑完整四变体三折：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_dct_v3_score_diagnostics.ps1 -Mode run -Variant all -Folds 0,2,3
```

Linux 服务器可用：

```bash
bash scripts/run_dct_v3_score_diagnostics.sh run all
```

独立重新汇总：

```powershell
python scripts/summarize_dct_v3_score_diagnostics.py --root results/dct_v3_score_diagnostics --expected-folds 0,2,3
```

汇总会输出 `results/dct_v3_score_diagnostics/dct_v3_score_summary.csv`，每折记录 best epoch、best C-index、best 附近三点均值、末五 epoch 均值、峰值-末五差、IPCW C-index、IBS、iAUC。

## 判读与下一步

- `full` 要至少在 2/3 折胜过某个删减版，并且 `best3` 或 `last5` 不更差，才说明该机制是稳定的分数贡献，而非尖峰。
- 若 `no_anchor` 更好或 `full` 只抬高 best 却扩大 `best_gap`，先回退或重做 anchor；不要先扫 intervention 强度。
- 若 `no_coordinate` 更好，优先保留 v3 的训练折 IPCW 逻辑，而把全局 prototype 坐标改为残差式或可学习映射。
- 若 `no_evidence_cost` 更好，说明 gate 同时控制 cost 和 marginals 过强；先只保留其中一条作用路径。
- 只有当 `full` 或一个删减版在三折上稳定后，才补 fold 1、4 形成正式五折；这时应报告 best、best3、last5 和选点规则，而不只报告每折峰值。
