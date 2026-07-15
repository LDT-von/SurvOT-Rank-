# Risk-Anchored Counterfactual Evidence Transport

`distributional_counterfactual_transport` is an experimental multimodal survival
line for **model-based counterfactual sensitivity analysis**.  It is not a causal
treatment-effect estimator.

## Mechanism

1. WSI patches and pathway tokens are pooled by separate global prototype
   dictionaries.  Prototype index therefore defines a stable coordinate across
   patients before population anchors are formed.
2. At the start of every fold, only that fold's training labels fit time-stage
   upper boundaries and a censoring Kaplan--Meier curve.
3. Each stage builds two IPCW-weighted transport-cost anchors during training:
   observed events within the stage form the high-risk anchor; patients known to
   survive past the stage boundary form the low-risk anchor.  The latter includes
   late-censored patients but excludes patients censored before that boundary.
4. Evidence gates modify both the transport energy and the OT marginals.  Weak
   semantic slots may therefore carry less mass instead of being forced to use a
   uniform balanced marginal.
5. To query a patient, factual stage costs are moved toward either anchor and
   Sinkhorn is solved again under the intervened cost.  A numerical projection
   verifies the resulting coupling's evidence-conditioned marginals.
6. Factual, low-anchor, and high-anchor couplings use the same survival decoder.
   Counterfactual risk direction is measured after training; it is never imposed
   as a training target.

```text
WSI patches / pathway tokens
        -> global semantic prototype coordinates
        -> stage-specific evidence-conditioned OT cost
        -> IPCW high-event / low-risk-set cost anchors
        -> cost-space intervention
        -> re-optimised Sinkhorn coupling
        -> shared survival decoder and post-hoc sensitivity metrics
```

## Objective

The external survival NLL remains the prediction objective.  The model adds:

1. OT cost regularisation;
2. censoring-aware continuous ranking;
3. risk-anchor contrast on observed event and risk-set membership; and
4. shared-prototype coordinate diversity.

There is deliberately no `low_risk_counterfactual < factual_risk <
high_risk_counterfactual` loss.

## Required validation

Report, per fold and across seeds:

- train-fold stage edges and anchor coverage;
- coupling marginal residuals;
- risk delta and survival-curve difference across `alpha` values;
- directional-consistency and monotonicity-violation rates;
- stability of prototype-coordinate assignments; and
- ablations of shared coordinates, IPCW anchors, evidence-conditioned marginals,
  and re-optimised cost-space intervention.
