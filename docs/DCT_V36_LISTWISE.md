# DCT v3.6：Transport-Conditioned Listwise 筛选

> 状态：实验候选，不替代 DCT v3.3/v3.5 主线，不构成已经验证的论文创新。

## 科学问题

直接在最终风险上加入 Plackett–Luce 风险集目标与 Cox partial likelihood
高度相邻，因此 v3.6 必须同时运行两个受控变体：

- **GPL**：在最终 factual risk 上做 censor-aware risk-set listwise，作为普通
  listwise control；
- **TCL**：在患者事件时间所属阶段的 factual transport event
  representation 上做同一个风险集目标。

只有 TCL 稳定优于 GPL，才能说明收益与 DCT 的 stagewise transport geometry
有关。两者都固定为 `NLL + 0.1 * listwise`，不叠加 IPCW pairwise 或 ETAR。

## 运行入口

本地仅检查命令，不启动癌种训练：

```bash
python scripts/run_dct_v36_listwise_screen.py plan \
  --variants gpl,tcl --cancers blca,brca --folds 0,2
```

训练服务器先运行单 batch smoke：

```bash
python scripts/run_dct_v36_listwise_screen.py smoke \
  --variants gpl,tcl --cancers blca,brca --folds 0,2
```

再运行六个匹配对照：

```bash
python scripts/run_dct_v36_listwise_screen.py run \
  --variants all --cancers blca,brca --folds 0,2
```

结果分别写入：

```text
results/dct_v3.6_listwise/<variant>/<cancer>
results/dct_v3.6_listwise_smoke/<variant>/<cancer>
```

脚本默认跳过已经存在 `split_<fold>_results_final.pkl` 的折，不覆盖旧 DCT
结果。汇总和预注册晋级判断：

```bash
python scripts/summarize_dct_v36_listwise.py
```

## 解释链路

v3.6 在评估阶段保存以下完整链路：

```text
WSI patch
  -> patient-local WSI slot
  -> global WSI prototype
  -> stage-specific factual/high/low OT coupling
  -> global omics prototype
  -> pathway token
```

病例级 patch–pathway transport mass 为：

\[
E^s=(A^{WSI})^\top P_s A^{omics}.
\]

训练结束后使用最佳 checkpoint 导出，`--set` 必须与该 checkpoint 的
GPL/TCL 配置一致：

```bash
python scripts/export_dct_v36_explanations.py \
  --config configs/distributional_counterfactual_transport_blca.yaml \
  --checkpoint results/dct_v3.6_listwise/tcl/blca/model_best_s0.pth \
  --fold 0 \
  --set dct_listwise_mode=stage_transport
```

每个病例输出：

- `summary.json`：factual/high/low risk、剂量响应、随机 anchor、固定 coupling
  对照和 patch 删除结果；
- `prototype_patch.csv`：prototype 对应的 top WSI patches；
- `stage_patch_pathway.csv`：各阶段 top patch–pathway transport relations；
- `transport_matrices.npz`：cost、coupling、边际及 assignment 原始矩阵。

只有 patch index 时不会声称生成 WSI 热图。提供匹配坐标目录后生成空间散点图；
同时提供 `.svs` 且环境安装 OpenSlide 时才生成真正的 WSI overlay。

## 晋级规则

fold0/fold2 只用于筛选。TCL 必须同时满足：

1. 所有折无 NaN，listwise score 与梯度均为有限值；
2. 每个癌种 mean Best 不低于 IPCW baseline 超过 `0.005`；
3. 总体 Last5 提升至少 `0.01`，或 Best–Last gap 缩小至少 `0.02`；
4. TCL 相对 GPL 至少赢三折，或总体 mean C-index 高 `0.005`。

未通过时保留 DCT v3.3/v3.5 主线，不把普通 listwise loss 包装成 DCT 创新。
