# CA-TET: Censoring-Aware Temporal Evidence Transport

## Positioning

CA-TET is the new mainline method for SurvOT-Rank. It addresses a specific
problem in multimodal WSI-pathway survival modeling: an attention map can look
plausible without proving that the selected pathology-pathway relation is used
by the survival predictor, while right censoring makes ordinary pairwise
supervision unreliable.

## Mechanism

1. A learned stage-specific edge risk score changes the OT cost before
   Sinkhorn optimization. The evidence therefore changes the transport plan,
   rather than being computed from a detached attention map.
2. Each temporal stage has an evidence gate over WSI-slot/pathway-slot pairs.
   The gated plan feeds the event token and the factual hazard head.
3. The risk-set transport loss only anchors comparisons at observed events and
   uses the empirical at-risk-set size as a censoring-aware weight. Censored
   patients remain context in the risk set but are not treated as observed
   failures.
4. The same plan is intervened on by keeping or removing gated edges. The
   intervention term rewards sufficiency and measurable comprehensiveness
   without forcing a preselected low-risk/high-risk direction.

## Objective

The outer training loop supplies the discrete-time survival NLL. CA-TET adds
only three compact terms:

```text
L = L_survival + lambda_ot L_ot
                  + lambda_rank L_risk-set-transport
                  + lambda_int L_evidence-intervention
```

The counterfactual output is a model-faithful intervention diagnostic, not a
causal treatment recommendation.

## Required evidence for a paper claim

The method should not be claimed as superior until the following are reported:

- shared OT versus stage-specific OT;
- stage-specific OT without edge-risk cost;
- edge-risk cost without risk-set supervision;
- full CA-TET;
- keep/remove/random evidence interventions;
- patch-level and pathway-level stability across seeds and folds;
- mean and standard deviation over the complete five-fold protocol.

The clean novelty claim is: **risk-set supervision is injected into the
stage-specific OT geometry and evaluated through interventions on that same
transport evidence under right censoring**. This is a methodological claim,
not a claim of causal discovery.
