# CATET Final: Censoring-Aware Stage Re-Transport

## Problem and claim

An attention map does not prove that selected pathology-pathway relations are
used by a survival predictor.  CATET Final makes stage identity part of the OT
cost before Sinkhorn and tests evidence through new balanced OT solutions.

For stage `s`, factual cost and interventions are:

```text
C_s       = C_base + lambda_prog * C_risk(pair, stage_s)
C_keep_s  = C_s + lambda_cf * (1 - evidence_gate_s)
C_drop_s  = C_s + lambda_cf * evidence_gate_s
P_s       = Sinkhorn(C_s)
P_keep_s  = Sinkhorn(C_keep_s)
P_drop_s  = Sinkhorn(C_drop_s)
```

All plan families preserve the same uniform row and column marginals.  Factual
logits are decoded from `P_s`; keep/drop logits are explanation audits and
never replace the factual survival prediction.

## Closed objective

```text
L = L_surv + lambda_ot * (L_transport + L_stage_plan_diversity)
           + lambda_rank * L_IPCW_rank
           + lambda_stage * L_censored_stage
           + lambda_intervention * (L_sufficiency + L_comprehensiveness)
```

`configure_train_reference()` fits stage edges and the censoring Kaplan-Meier
curve from the current training fold only.  Observed patients supervise their
event stage with IPCW; a censored patient contributes survival probability
beyond the censoring stage.  The method claims model-faithful evidence
counterfactuals, not causal treatment effects.

## Exact explanations

`explain_last_batch()` returns stage edge risks, evidence gates, factual/keep/
drop plans, event-stage probabilities, risks, intervention gaps, adjacent-stage
plan distances, and row/column marginal errors.  These are the exact plans and
predictions used in the forward pass.

## Run and required evidence

```bash
python -m survot_rank.cli train \
  --config configs/censoring_aware_temporal_evidence_transport_blca.yaml
```

Do not start the cross-cancer queue unless stage plans are non-identical,
marginal error stays below tolerance, IPCW pairs are nonzero, and evidence
removal changes risk without non-finite values.  Required ablations are shared
stage cost, no IPCW, no censored-stage term, masked-plan intervention, random
intervention, and the complete final model.
