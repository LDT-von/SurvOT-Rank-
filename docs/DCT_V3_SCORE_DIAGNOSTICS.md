# DCT v3 分数诊断

这套入口用于判断新的 score-first 目标是否真的提高 BLCA 泛化分数。当前
主线固定为 50 epochs、batch size 8、seed 3、AdamW，并保持每折最早达到
最佳验证 C-index 的 checkpoint。

## 四个对照

| 变体 | 与 `full` 的区别 | 检验问题 |
| --- | --- | --- |
| `full` | 无 | NLL + IPCW 可比较样本排序 |
| `nll_only` | `dct_lambda_ipcw_rank=0` | 排序监督是否真正提高 C-index |
| `unweighted_rank` | 关闭 IPCW 排序，启用旧普通排序 | IPCW 是否优于未校正删失的排序 |
| `legacy_six_loss` | 恢复旧 OT/普通排序/anchor/阶段/坐标五个辅助项 | 多目标冲突是否是低分来源 |

`full` 的默认目标只有：

```text
survival NLL + 0.10 * IPCW comparable-pair rank
```

OT、anchor 和 prototype 坐标仍参与前向结构与反事实解释，但不再各自争夺
预测梯度。训练日志会额外记录 `ipcw_rank`、`ipcw_pairs`、OT 距离、anchor
覆盖率等诊断量。

## 运行

先做环境检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_dct_v3_score_diagnostics.ps1 -Mode doctor
```

再做单 epoch 冒烟检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_dct_v3_score_diagnostics.ps1 -Mode smoke
```

正式诊断三个代表折：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_dct_v3_score_diagnostics.ps1 -Mode run -Variant all -Folds 0,2,3
```

Linux：

```bash
bash scripts/run_dct_v3_score_diagnostics.sh run all
```

独立汇总：

```powershell
python scripts/summarize_dct_v3_score_diagnostics.py --root results/dct_v3_score_diagnostics --expected-folds 0,2,3
```

汇总文件为 `results/dct_v3_score_diagnostics/dct_v3_score_summary.csv`，包含
每折 best epoch、best C-index、best 附近三点均值、末五轮均值、峰值差、
IPCW C-index、IBS 和 iAUC。

## 判读

- `full` 应在多数折超过 `nll_only`，否则将排名权重继续降到 0.05。
- `full` 若胜过 `unweighted_rank`，说明 train-fold IPCW 校正有效。
- `legacy_six_loss` 若更差且 `best_gap` 更大，就能直接确认旧多目标配方造成
  过拟合或梯度冲突。
- 先用 0/2/3 折定位，再让胜出配方跑完整五折；最终同时报告 best、best3、
  last5，避免只靠单 epoch 尖峰下结论。
