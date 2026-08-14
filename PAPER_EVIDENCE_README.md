# Paper-evidence launcher chain (DCT v3.8.2)

Five mechanism-targeted ablations plus a post-hoc audit. Every launcher
inherits the frozen v3.8.2 `fixed-full` recipe and changes exactly one
mechanism. Run order below matches the recommended priority on 2026-08-14.

## 0. Sanity (no GPU required)

```bash
py -3 -m pytest tests/test_dct_v382_paper_evidence.py -v
```

Eight smoke tests prove the ablation switches and audit loader do not perturb
the production recipe when their flags are off.

## 1. Plan every queue (no GPU, no training)

```bash
py -3 scripts/run_dct_v382_paper_evidence.py plan --python python
py -3 scripts/run_dct_v382_cross_cancer_prototype.py plan --python python
```

Each ``plan`` invocation prints the full command list with the slot it
writes to. Adjust cancers / folds with ``--cancers`` and ``--folds``.

## 2. Mechanism ablations (4 variants × 3 cancers × fold 1 = 12 jobs)

```bash
py -3 scripts/run_dct_v382_paper_evidence.py run \
    --python python --gpu 0 --cancers ucec,blca,lusc --folds 1
```

Variants and what each one proves (all default to off in production):

| Variant | Flag changed | What it probes |
|---|---|---|
| `fixed_coupling` | `dct_fixed_coupling=True` | OT intervention chain — does re-solving Sinkhorn matter? |
| `random_anchors` | `dct_random_anchors=True` | Risk-set anchor evidence — IPCW vs random perturbation |
| `null_calibration` | `dct_perm_labels_seed=1` | Audit specificity — label shuffling should null out the signal |
| `stage_randomization` | `dct_stage_jitter_fraction=0.30` | Stage-edge evidence — does exact edge placement matter? |

Default schedule is one fold per cancer (fold 1). For null calibration
you can repeat with seeds 2–5 by passing ``--set dct_perm_labels_seed=N``
once per seed.

## 3. Cross-cancer prototype transfer (BLCA → KIRC/UCEC/LUSC = 6 jobs)

```bash
py -3 scripts/run_dct_v382_cross_cancer_prototype.py run \
    --python python --gpu 0 --pairs blca->kirc,blca->ucec,blca->lusc
```

Each pair schedules a `source` job (BLCA, 50 epochs) followed by a
`target` job (50 epochs) that loads the source checkpoint and freezes
the two shared prototype tensors via the new
`dct_freeze_source_prototype=<path>` flag.

If the target cancer's C-index stays above ~60% of the source value, the
shared prototypes are cross-cancer semantic. If it collapses, the
"global coordinate" claim is weakened.

## 4. Post-hoc audit (one launch per fold per cancer)

```bash
py -3 scripts/audit_dct_v382.py audit \
    --config configs/distributional_counterfactual_transport_blca.yaml \
    --checkpoint results/dct_v3.8.2/robust/fixed_full/blca/model_best_s1.pth \
    --fold 1 \
    --output-dir results/dct_v3.8.2_paper_evidence/audit/blca_fold1
```

Captures ``factual_risk``, ``low_risk_counterfactual``,
``high_risk_counterfactual`` plus factual vs intervened transport
distances. Emits ``audit_metrics_fold<N>.json`` with three metrics:

- `direction_consistency.correct_rate` (label-aware monotonicity)
- `direction_consistency.chance_gap` (correct_rate − 0.5)
- `reconfiguration.above_margin_rate`

For dose monotonicity, also run:

```bash
py -3 scripts/audit_dct_v382.py sweep \
    --config configs/distributional_counterfactual_transport_blca.yaml \
    --checkpoint results/dct_v3.8.2/robust/fixed_full/blca/model_best_s1.pth \
    --fold 1 \
    --output-dir results/dct_v3.8.2_paper_evidence/audit/blca_fold1
```

## 5. Summarise into one ledger

```bash
py -3 scripts/summarize_paper_evidence.py
cat PAPER_EVIDENCE_LEDGER.md
```

The ledger merges ablation C-index numbers, audit JSON, and
cross-cancer transfer numbers. The pass criteria block at the bottom
shows what each variant must achieve for the paper claim to hold.

## What these ablations and audits prove

Mapping back to the DCT v3.8.2 paper claims (see
`paper_drafts/DCT/DCT_主张与证据台账.md`):

| Claim | Mechanism evidence source |
|---|---|
| C1 (factual coupling is not enough) | `fixed_coupling` ablation + audit `reconfiguration.above_margin_rate` |
| C2 (shared semantic coordinates) | `cross_cancer_prototype` transfer |
| C4 (risk-set anchors carry signal) | `random_anchors` ablation + audit `direction_consistency` |
| C5 (intervention chain is auditable) | audit `direction_consistency.correct_rate` |
| C6 (three structural losses add value) | leave-one-out is in `run_dct_v382_paper_ablations.py` (already shipped) |
| C8 (MGPTR is auxiliary, not core) | `no_mgptr` ablation in `run_dct_v382_paper_ablations.py` |

The chain is intentionally auditable: each ablation has a one-flag
delta from the frozen recipe, and the audit loader writes raw JSON that
the summariser turns into the paper ledger.
