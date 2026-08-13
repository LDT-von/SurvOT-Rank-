# Three-method final implementation plan

This document freezes one paper-facing version for each of CA-PSA, ArcSurv,
and CATET.  A version is considered final only when its prediction path, its
training objective, and its exported explanation all describe the same
mathematical object.

## Shared protocol and non-negotiable rules

- The external survival objective remains the discrete-time survival NLL.
- Method-specific losses may shape the claimed mechanism, but may not replace
  the survival prediction path with an unrelated proxy.
- Every explanation is computed from the exact tensors used for prediction.
- Missing modalities are explicit masks; zero-filled missing WSI features are
  not accepted as real observations.
- Hyperparameters are fixed across cancers.  Cancer-specific tuning is not a
  final method.
- A local smoke test proves numerical and structural correctness only.  It
  does not establish a C-index improvement; that requires the unified
  cross-cancer five-fold server protocol.

## CA-PSA Final: identifiable cohort routes

### Claim

Each cohort anchor defines a prognostic route.  Patient-specific WSI and omics
states update the same route identity, while a budgeted hard-concrete gate
selects the routes used by that patient.

### Closed objective

`L = L_surv + lambda_identity * (L_cross_modal + L_anchor_separation)
              + lambda_budget * L_gate_budget`

- `L_cross_modal` is a bidirectional contrastive loss: the WSI state for route
  `k` must match omics route `k` more than all other route indices.
- `L_anchor_separation` prevents different cohort anchors from collapsing to
  the same direction.
- `L_gate_budget` targets a patient-level active-route ratio and therefore
  removes the all-off optimum of the old expected-L0 penalty.

The identity claim is explicitly "identifiable up to a global permutation".
Cross-fold comparisons must use optimal matching rather than raw slot index.

### Required explanation and audits

- per-patient gate probability and hard active count;
- per-route, per-time-bin hazard contribution used in the final logits;
- same-index versus off-index cross-modal similarity and matching accuracy;
- anchor pairwise cosine and effective route usage.

## ArcSurv Final: shared cohort prognostic simplex

### Claim

WSI and omics patient states are mapped into one common coordinate system and
composed over one shared cohort archetype bank.  Patient hazard logits are the
convex composition of archetype-specific hazard curves.

### Closed objective

`L = L_surv + lambda_recon * L_hull
              + lambda_align * L_composition_JS
              + lambda_volume * L_simplex_volume
              + lambda_balance * L_cohort_usage
              + lambda_rank * L_survival_rank`

- There is one bank and one `Beta`, so WSI archetype `k` and omics archetype
  `k` cannot silently refer to different cohort supports.
- Archetypes remain row-stochastic convex combinations of the bank.
- The state encoder is frozen when the bank freezes, preventing the learned
  coordinate system from drifting away from its cohort support.
- Hard furthest-point seeding is optional and disabled in the final recipe;
  it is not silently applied after warmup.

### Required explanation and audits

- patient archetype composition and entropy;
- archetype hazard/survival curves and patient-wise additive contributions;
- shared bank support weights;
- pairwise archetype cosine, simplex volume, hazard spread, active fraction;
- exact convex-hull reconstruction and row-sum invariants.

## CATET Final: censoring-aware stage-conditioned re-transport

### Claim

Each survival stage changes the WSI-omics transport geometry itself.  Evidence
keep/remove interventions modify the cost and re-solve balanced OT, so all
factual and counterfactual plans preserve the same row and column marginals.

### Closed objective

`L = L_surv + lambda_ot * L_transport
              + lambda_rank * L_IPCW_rank
              + lambda_stage * L_censored_stage
              + lambda_intervention * (L_sufficiency + L_comprehensiveness)`

- Stage embeddings enter the edge-cost network before Sinkhorn; stage costs
  are not copies made with `expand`.
- Factual logits are predicted from factual plans.  Keep/remove plans are used
  only as faithful interventions on that same predictor.
- Keep and remove interventions are new Sinkhorn solutions with penalized
  costs, not elementwise plan masking.
- IPCW weights and stage edges are fitted from the training fold only through
  `configure_train_reference`.
- The stage objective treats observed events as event-stage targets and
  censored observations as survival-beyond-stage constraints.

### Required explanation and audits

- per-stage edge risk, evidence gate, plan, and event weight;
- per-stage pairwise plan distance proving temporal non-identity;
- factual/keep/remove patient risk and sufficiency/comprehensiveness gaps;
- row/column marginal errors for all three plan families;
- IPCW pair count, censoring weights, stage NLL, and finite-value flag.

## Acceptance gates before GPU training

1. Targeted tests cover every invariant above and all gradients are finite.
2. Evaluation is deterministic for identical inputs.
3. A synthetic diagnostic makes each claimed mechanism react in the expected
   direction (identity matching, archetype mixture, or stage re-transport).
4. No explanation is reconstructed post hoc from tensors outside prediction.
5. A two-epoch smoke run completes on one supported fold before any five-fold
   queue is launched.

## Server experiment order

1. CA-PSA Final: BLCA fold0 mechanism screen, then six supported cancers x five
   folds only if all identity/gate audits pass.
2. ArcSurv Final: BLCA fold0 mechanism screen, then the same queue only if the
   shared simplex remains non-collapsed and uses at least half its archetypes.
3. CATET Final: BLCA fold0 mechanism screen, then the same queue only if stage
   plans differ, balanced marginals hold, and remove interventions measurably
   change risk without numerical instability.

The previous CA-PSA full, ArcSurv staged/hard-repair, and CATET repaired runs
remain historical baselines.  They must not be mixed with these final methods.
