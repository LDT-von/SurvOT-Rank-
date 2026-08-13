# ArcSurv Final: Shared Cohort Prognostic Simplex

## Problem and claim

ArcSurv represents each patient as a convex mixture of cohort-supported
prognostic extreme states.  The final version uses one shared bank for WSI and
omics.  This removes the old ambiguity in which WSI archetype `k` and omics
archetype `k` could be convex combinations of unrelated patients.

```text
A = softmax(Beta) @ Z_train
pi_w = softmax(-distance(wsi_state, A) / temperature)
pi_o = softmax(-distance(omic_state, A) / temperature)
pi   = availability-weighted mean(pi_w, pi_o)
logits = pi @ archetype_hazard_logits + bias
```

`A` is always a row-stochastic convex combination of the fold-local bank.  The
state encoder is frozen when the bank freezes, so later optimization cannot
move patients into a coordinate system no longer represented by the hull.
Hard furthest-point seeding is an explicit option and is disabled in the final
recipe.

## Closed objective

```text
L = L_surv + lambda_recon * L_hull_reconstruction
           + lambda_align * JS(pi_w, pi_o)
           + lambda_balance * L_cohort_usage
           + lambda_volume * L_simplex_volume
           + lambda_rank * L_survival_rank
```

The survival predictor has no bypass around the simplex.  The optional
sharpness term is zero in the final recipe because forced one-hot mixtures are
not required for interpretability.

## Exact explanations

`explain_last_batch()` returns modality and patient compositions, the shared
archetypes, support weights, archetype hazard/survival curves, and additive
archetype logit contributions.  Contributions plus the shared bias exactly
reconstruct the prediction.

## Run and required evidence

```bash
python -m survot_rank.cli train \
  --config configs/archetypal_risk_composition_blca.yaml
```

Report convex-hull row errors, archetype cosine, simplex volume, hazard spread,
active fraction, composition entropy, and cross-fold matched archetype
stability.  The novelty claim is the shared multimodal prognostic simplex and
its direct hazard composition, not archetypal analysis or Slot Attention alone.
