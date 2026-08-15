# CATET — Censoring-Aware Stage Re-Transport

## Idea

> **SOTA problem:** Multimodal survival OT methods (MOTCat, MMP, ProtoPathway, PIBD, …)
> run one balanced OT and consume the plan as a fusion feature.  No method audits
> whether the chosen transport geometry actually carries through to the risk
> prediction.

> **CATET claim:** Build a stage-specific OT cost where stage identity enters the
> cost *before* Sinkhorn, then re-solve balanced OT after a cost offset and
> compare the factual vs. counterfactual risk.  The intervention is *model-
> internal: distribution-shape sensitivity*, not a treatment causal effect.

## Mechanism

For stage `s`:

```
C_s       = C_base + λ_prog * C_risk(pair, stage_s)
C_keep_s  = C_s + λ_cf * (1 − gate_s)
C_drop_s  = C_s + λ_cf * gate_s
P_s       = Sinkhorn-IPFP(C_s)           ← factual prediction plan
P_keep_s  = Sinkhorn-IPFP(C_keep_s)      ← keep intervention plan
P_drop_s  = Sinkhorn-IPFP(C_drop_s)      ← drop intervention plan
```

`P_s` is the only plan that feeds the hazard head.  `P_keep_s` and `P_drop_s`
are explanation audits: their risk scores are compared with the factual risk to
check sufficiency (keep ≈ factual) and comprehensiveness (drop differs).  The
audit gate is direction-free.

All three plan families preserve uniform row and column marginals (IPFP enforces
this; tolerance ≤ 5 × 10⁻⁴).  The decisive structural differences from the
SOTA are:

| | SOTA (MOTCat / MMP / ProtoPathway) | CATET |
|---|---|---|
| OT solves | One plan per patient | One plan *per stage* per patient |
| Stage identity | Not in cost | In cost before Sinkhorn |
| Intervention | None | Cost offset + re-Sinkhorn |
| Edge risk | Aggregated post-OT | Learns cost bias before Sinkhorn |
| Censoring | Standard NLL or IPCW | Train-fold KM IPCW on *final* risk |

## Closed objective

```
L = L_surv
    + λ_ot   * (L_ot + L_plan_diversity)
    + λ_rank * L_ipcw_rank        ← supervised on factual_logits → risk_score
    + λ_stage * L_censored_stage  ← observed: event stage NLL; censored: tail mass
    + λ_intervention * (L_sufficiency + L_comprehensiveness + 0.1 * gate_budget)
```

`configure_train_reference()` fits two fold-local references *before* the first
forward pass:

- `stage_edges`: observed-event quantiles → stage boundary percentiles.
- `censor_KM`: censoring Kaplan-Meier → IPCW weight = KM⁻¹.

No test-fold information reaches the auxiliary terms.  The method claims
model-faithful evidence counterfactuals, not causal treatment effects.

## What this file does NOT contain

`CohortAnchoredRouter`, `archetype_prior_per_stage`, and the route-consistency
KL are the v2 lattice additions.  They are not part of the CATET claim: the OT
operates on the native slot × slot cost matrix.  They are intentionally removed.
If cohort routing is needed, it belongs under CA-PSA.

## Exact explanations

`explain_last_batch()` returns 22 tensors: stage edge risks, evidence gates,
factual/keep/drop plans, event-stage probabilities, risks, sufficiency/
comprehensiveness gaps, adjacent-stage plan distances, and row/column marginal
errors for all three plan families.  These are the exact plans and predictions
used in the forward pass — no post-hoc approximation.

## Audit checklist

**Do not start the cross-cancer queue unless all of the following hold:**

| Check | Tolerance | How to verify |
|---|---|---|
| Stage costs are genuinely different | `stage[:,0] not allclose stage[:,1]` | `test_catet_final_has_true_stage_costs_…` |
| Row/column marginal error | ≤ 5 × 10⁻⁴ (all three plan families) | `test_catet_final_…` assertions |
| Factual risk = `_risk_score(logits)` | Exact match | `test_catet_final_…` assertion |
| IPCW pairs nonzero | `last_training_losses["catet_ipcw_pairs"] > 0` | `test_catet_final_…` |
| IPCW weight nontrivial | `_ipcw(t=16.0) > 1.0` after `configure_train_reference` | `test_catet_training_reference_…` |
| Censored stage loss prefers correct stage | observed-correct < observed-wrong | `test_catet_stage_nll_prefers_…` |
| No non-finite values | `last_training_losses["catet_finite"] == 1` | `test_catet_final_…` |
| All hyperparameters valid | ValueError on 4 invalid configs | `test_catet_rejects_invalid_…` |

**Required mechanism ablations (in order of priority):**

1. `shared_stage_cost` — collapse stage embedding → equivalent to v1 expand bug.
   Expected: drop in direction-consistency rate.
2. `no_ipcw` — remove train-fold KM reference, IPCW degrades to uniform weights.
   Expected: ranking loss gradient noisier, C-index may drop.
3. `no_censored_stage` — remove `L_censored_stage`, keep only IPCW ranking.
   Expected: censored patients contribute less signal.
4. `masked_plan` — keep/remove = plan × gate renormalization (v1 behavior).
   Expected: marginal errors > 5e-4, sufficiency gap widens.
5. `random_gate` — evidence gate replaced by uniform random [0,1].
   Expected: sufficiency gap ≈ 0, comprehensiveness gap ≈ 0.
6. `final_model` — all three fixes simultaneously → this implementation.

**Audit script (to be written):**

```bash
python scripts/audit_catet.py --config configs/catet_final_blca.yaml --fold 0
# Reports: direction rate / dose monotone rate / plan conservation /
#          sufficiency / comprehensiveness / random-gate baseline per fold
```

## Run

```bash
python -m survot_rank.cli train \
  --config configs/censoring_aware_temporal_evidence_transport_blca.yaml
```
