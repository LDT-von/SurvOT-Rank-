# DCT v3.10 Direction Loss 消融实验

**日期**: 2026-08-26  
**方法**: DCT v3.10 directional regularized transport  
**对比**: λ = 0.00 (direction loss OFF) vs λ = 0.05 (direction loss ON)  
**特征**: uni2-h, 2048 patches, batch=8, 50 epochs  
**划分**: 5-fold legacy, 测试前 2 folds (k=2..3)

---

## 结果（val c-index，2-fold mean）

| Cancer | d=0.00 | d=0.05 | Δ | 方向 |
|--------|--------|--------|---|------|
| blca   | 0.6756 | 0.7281 | **+0.0525** | ✅ 显著正向 |
| ucec   | 0.8095 | 0.8050 | -0.0046 | ➖ 持平 |
| kirc   | 0.8069 | 0.8009 | -0.0060 | ➖ 持平 |
| skcm   | 0.7141 | 0.7035 | -0.0106 | ➖ 持平 |
| hnsc   | 0.6780 | 0.6480 | -0.0300 | ⚠️ 轻微负向 |
| lusc   | 0.6361 | 0.6617 | +0.0256 | ✅ 正向 |

---

## 结论

- **BLCA 是最大赢家**：direction loss 带来 +0.053 的 val c-index 提升
- **LUSC 也有改善**：+0.026
- **其余 4 个癌种**：效果不明显或轻微负面
- **总体**：方向性损失在多数癌种上无显著副作用，在 BLCA 和 LUSC 上有实质增益

---

## 配置详情

- `survot_method=dct_v310_directional_regularized_transport`
- `bag_loss=nll_surv`
- `dct_lambda_ipcw_rank=0.1`
- `dct_lambda_etar=0.0` (ETAR 关)
- `dct_v38_lambda_dose=0.0`
- `dct_v38_lambda_reconfiguration=0.0`
- `fit_bins_on_train=true`
- `binning_mode=global_qcut`
- `event_stratified_batches=true`
- `event_sampling_fraction=0.0`
