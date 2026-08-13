# three_method_v2 launcher — push protocol

## What it does

Runs the three v2 methods (CA-PSA v2 / ArcSurv v2 / CATET v2) under one
queue, mirroring `run_three_method_final_cross_cancer.py` but pointing at
`*_v2_blca.yaml` configs and pushing the v2 hyperparameters.  Results
land under `results/three_method_v2/` to stay isolated from the existing
`three_method_final` runs.

## Why a new launcher

The previous `run_three_method_final_cross_cancer.py` was locked to the
**final** model code and the **final** hyperparameters.  v2 changes
both: cohort ArchetypeBank anchors, hard top-K gates, CohortAnchoredRouter,
safe euclidean cost.  Re-running the old launcher would silently revert
to the final models because it overrode `--set` with old final values.

This launcher targets the v2 model code + v2 configs + v2 hyperparameters
in a single queue.  If the v2 results are weak we can still cross-check
against `three_method_final` runs since the result roots do not overlap.

## Push sequence (suggested)

```
# 1. on remote box, confirm v2 launcher is wired
python scripts/run_three_method_v2_cross_cancer.py doctor --cancers blca --folds 0

# 2. print the queue to verify
python scripts/run_three_method_v2_cross_cancer.py plan \
    --cancers blca --folds 0,1,2,3,4

# 3. first smoke check (cheap: 2 epochs, 2 batches)
python scripts/run_three_method_v2_cross_cancer.py smoke \
    --gpu 0 --cancers blca --folds 0

# 4. full BLCA five-fold run
python scripts/run_three_method_v2_cross_cancer.py run \
    --gpu 0 --cancers blca --folds 0,1,2,3,4

# 5. extend to other cancers once BLCA is healthy
python scripts/run_three_method_v2_cross_cancer.py run \
    --gpu 0 --cancers ucec,kirc,skcm,hnsc,lusc --folds 0,1,2,3,4
```

## v2 hyperparameter summary

| method      | key change vs final                                                  | file                                         |
|-------------|----------------------------------------------------------------------|----------------------------------------------|
| capsa_v2    | threshold gate (capsa_full), ArchetypeBank anchors, slot-attn hazard | configs/capsa_v2_blca.yaml                   |
| arcsurv_v2  | hard top-K gate (CA-PSA), KL faithfulness audit (CATET), no volume   | configs/arcsurv_v2_blca.yaml                 |
| catet_v2    | CohortAnchoredRouter, lazy archetype prior, safe euclidean cost     | configs/catet_v2_blca.yaml                   |

All three v2 models pass `tests/test_three_method_v2_smoke.py`:
single-step forward + backward + eval mode.  CATET v2 multi-step
stability is verified by `tmp/trace_8steps.py` (8 Adam steps, 0 NaN
gradients after the `_safe_euclidean_cost` fix).

## Stop conditions

- `[already-running]` — another queue is active on the same GPU.
- `[BLOCKED]` — doctor check failed (no UNI2-h features, no 5fold_uni2h split).
- `[ERROR]` — child process returned non-zero; queue halts.

Re-running with `--force` re-launches every job regardless of cached
pickles.