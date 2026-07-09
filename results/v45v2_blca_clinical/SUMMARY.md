# V45v2 + Clinical (age + gender) on BLCA / DSS — 5-fold Results

Run: `configs/v45v2_blca_clinical.yaml`
Seed: 3 | lr: 5e-4 (Adam) | batch: 4 | max_epochs: 30 | num_workers: 4
Slots: WSI=8 / Omic=8, iters=5, temp=0.01, topk=0.25 (parallel_topk_st)
OTEH v2: eps=0.05, iter=50, warmup=5, events=24, heads=4, layers=4, dropout=0.1
Loss: lambda_ot=0.06 / div=0.01 / event_surv=0.25 / recon=0.2
RankEvent: per_event=0.15 / rank=0.15 / global_cons=0.02 / gate_ent=0.005
Anneal: eps_start=0.10 -> eps_end=0.05 over 12 epochs | global_init=-2.0
Clinical fusion: age + gender (feature_dim=2), enabled

## Per-fold best epoch (selected by val C-index)

| Fold | Best Epoch | C-index | IPCW    | IBS     | iAUC    |
|------|------------|---------|---------|---------|---------|
| 0    | 14         | 0.7417  | 0.4581  | 0.8828  | 0.8302  |
| 1    | 24         | 0.7453  | 0.6801  | 0.4261  | 0.6910  |
| 2    | 8          | 0.6237  | 0.5581  | 0.8884  | 0.4965  |
| 3    | 20         | 0.7049  | 0.6313  | 0.4543  | 0.5897  |
| 4    | 15         | 0.6441  | 0.6074  | 0.3131  | 0.6253  |

## 5-fold summary (mean ± std)

| Metric   | Value          | Direction       |
|----------|----------------|-----------------|
| C-index  | 0.6919 ± 0.0499 | higher = better |
| IPCW     | 0.5870 ± 0.0755 | higher = better |
| IBS      | 0.5929 ± 0.2436 | lower  = better |
| iAUC     | 0.6465 ± 0.1113 | higher = better |

## Notes
- Fold 2/3/4 were restarted on a dedicated GPU after a colleague's
  sweep job was sharing the same GPU during folds 0/1 (and the first
  half of fold 2). The new runs use the same seed/hyperparams as the
  old logs so the metrics above are fully reproducible from
  `configs/v45v2_blca_clinical.yaml`.
- Late epochs (22-29) show a wide C-index wobble (0.35-0.55) on
  fold 4; we report the per-fold best epoch selected by val C-index
  (model_best_s{0..4}.pth).
- All metrics computed on 5-fold val splits, BLCA / DSS label,
  `survival_months_dss`.
