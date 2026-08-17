# DCT v3.10: Directionally Regularized Transport (DCT-Reg)

## Frozen method

The paper-facing version is registered as
`dct_v310_directional_regularized_transport` and uses exactly

\[
\mathcal L_{\mathrm{DCT-Reg}}
=
\mathcal L_{\mathrm{NLL}}
+0.10\,\mathcal L_{\mathrm{IPCW-rank}}
+0.05\,\mathcal L_{\mathrm{direction}}.
\]

The class forces ETAR, listwise, dose, reconfiguration and MGPTR weights to
zero. Adaptive auxiliary weighting is disabled. The historical structural
warmup/ramp is also disabled, so the direction coefficient is not silently
changed by epoch.

The defensible claim is that DCT-Reg **learns a directionally consistent risk
response to prognostic ground-cost interventions after re-solving Sinkhorn**.
It does not claim that an unconstrained OT plan naturally has prognostic
semantics, and it is not a causal treatment-effect model.

## Required experiments

### E1. Matched 2 x 2 objective ablation (mandatory)

Run all four variants on BLCA, five paired folds, with identical data, split,
seed, optimizer, epoch budget and model-selection rule:

| Variant | Objective | Question |
|---|---|---|
| `nll_only` | NLL | Prediction-only baseline |
| `ipcw_only` | NLL + 0.10 IPCW | Does censoring-aware ranking help? |
| `direction_only` | NLL + 0.05 direction | Does direction work without rank? |
| `full` | NLL + 0.10 IPCW + 0.05 direction | Is the complete frozen method necessary? |

Use `scripts/run_dct_v310_experiments.py`. Report every fold, paired fold
differences, mean, standard deviation and a paired confidence interval. Do not
infer contribution by subtracting results from different splits or encoders.

Read-only queue check:

```powershell
python scripts/run_dct_v310_experiments.py plan
```

### E2. Final prediction benchmark (mandatory)

Run DCT v3.10 on BLCA, UCEC, KIRC, HNSC, SKCM and LUSC, five folds each, using
`scripts/run_dct_v310_final_cross_cancer.py`. Report Harrell C-index,
IPCW C-index, IBS and time-dependent AUC. Archive per-patient predictions,
training curves, checkpoints, resolved config, split hash and Git revision.

To show that the direction term generalizes rather than only the whole model,
also run `ipcw_only` on UCEC and LUSC for all five folds and compare it against
the matching v3.10 folds from the final queue. Expanding this paired comparison
to all six cancers is preferable if compute permits.

For a paper-grade estimate, choose checkpoints on an inner validation set and
evaluate once on the untouched outer fold. The outer fold must not be used for
epoch selection and then renamed as a test set.

### E3. Transport-mechanism controls (mandatory)

Run at least BLCA, UCEC and LUSC on folds 1, 2 and 4:

| Control | Required observation |
|---|---|
| `fixed_coupling` | Replaying the factual coupling should weaken the learned intervention response if re-Sinkhorn is load-bearing |
| `noisy_batch_mean_anchors` | Non-prognostic noisy batch-mean anchors should not reproduce the true-anchor direction response |
| `permuted_reference` | Reference-time permutation should move audit statistics toward their null distribution |
| shuffled/uniform feasible plans | Prediction or risk response should change if the decoder functionally uses the coupling |

The first three controls are launchable through
`scripts/run_dct_v310_experiments.py`. Shuffled/uniform feasible plans are a
held-out checkpoint audit and must preserve marginals before decoding.

The default experiment queue is E1 only. Build the required E3 queue explicitly:

```powershell
python scripts/run_dct_v310_experiments.py plan --cancers blca,ucec,lusc --folds 1,2,4 --variants fixed_coupling,noisy_batch_mean_anchors,permuted_reference
```

### E4. Continuous intervention audit (mandatory)

On held-out patients, sweep intervention dose
\(\alpha\in\{0,0.25,0.5,0.75,1\}\) and report:

- direction-consistency rate and high/low risk deltas;
- plan total variation relative to the factual coupling;
- marginal error for every re-solved plan;
- patient-level response curves and confidence intervals.

Dose monotonicity is an audit metric only; it is not another training loss.

### E5. Predictive baselines and statistics (mandatory for submission)

Compare under the same outer folds and features against a non-OT fusion
baseline and representative survival fusion/OT baselines. Report paired
bootstrap or permutation confidence intervals and correct for multiple cancer
comparisons. Parameter count, inference time and peak memory should also be
reported because DCT-Reg performs extra Sinkhorn solves during training.

## Historical evidence boundary

- DCT v3.3 is NLL + IPCW only. Its scores are historical motivation and a
  no-direction precursor, not DCT v3.10 results.
- The v3.8 BLCA `direction` fold-0 score is preliminary single-fold evidence.
  It cannot replace E1.
- DCT v3.8.2 fixed-full six-cancer scores remain a historical comparator. They
  include dose, reconfiguration and MGPTR and cannot be relabelled as v3.10.
- Tests and smoke runs establish wiring and numerical health, not predictive or
  mechanistic performance.
