# CA-PSA Final: Identifiable Cohort Routes

## Problem and claim

Ordinary patient-wise slots are exchangeable and a fixed slot count gives every
patient the same latent capacity.  CA-PSA Final defines a cohort route anchor
`a_k`, updates that same route separately from WSI and omics, and activates a
patient-specific subset.  Route identity is claimed only up to a global
permutation; cross-fold comparison therefore requires optimal matching.

For modality `m` and patient `i`:

```text
z_i,k,m = normalize(a_k) * anchor_scale + state_i,k,m
```

Tokens compete across the shared route indices.  Same-index WSI and omics
routes are fused, emit time-bin logits, and receive time-specific mixture
weights.  A hard-concrete estimator is used during training; evaluation keeps
the top `round(target_active_ratio * K_max)` routes deterministically.

## Closed objective

```text
L = L_surv + lambda_identity * (L_contrastive_identity + L_anchor_separation)
           + lambda_budget * L_gate_budget
```

- Bidirectional contrastive identity makes WSI route `k` match omics route `k`
  more strongly than every off-index route.
- Anchor separation penalizes off-diagonal anchor cosine above a margin.
- Gate budget matches both per-patient capacity and cohort route usage to the
  target ratio.  Unlike expected-L0 minimization, it has no all-off optimum.

## Exact explanations

`explain_last_batch()` returns gates, open probabilities, active counts,
attention maps, anchor cosine, identity matching diagnostics, and
`route_time_contribution`.  Summing the last tensor over routes exactly
reconstructs the logits used by survival NLL.

## Run and required evidence

```bash
python -m survot_rank.cli train \
  --config configs/cohort_anchored_adaptive_prognostic_slot_attention_blca.yaml
```

Before a paper claim, report cross-fold anchor matching, identity accuracy,
route usage, gate counts, and the ablations `no identity`, `no budget`, fixed
all-on routes, and independent modality anchors.  A route is not called a
biological pathway without external enrichment evidence.
