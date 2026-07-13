# Distributional Counterfactual Transport

`distributional_counterfactual_transport` is an independent experimental method line. It keeps the previous methods unchanged and adds a latent, model-faithful counterfactual explanation layer.

## Core idea

The method learns stage-wise low-risk and high-risk evidence distributions over WSI-slot/pathway-slot transport plans. For each patient, it interpolates the factual plan toward both distributions and recomputes the survival prediction:

```text
factual evidence plan
        -> low-risk counterfactual plan -> risk change
        -> high-risk counterfactual plan -> risk change
```

The explanation exposes the factual evidence, both risk prototypes, the two counterfactual risks, and the risk deltas. The counterfactual lives in the learned evidence space; it must not be described as a causal treatment effect.

## Objective

The total objective contains the survival NLL outside the model plus four compact auxiliary terms:

1. OT regularization;
2. censoring-aware continuous ranking;
3. bidirectional counterfactual risk ordering;
4. sparse risk-prototype entropy.

## Required evidence

Before claiming an improvement, compare the factual prediction with low/high counterfactual risk, measure the transport distance required to cross a risk margin, and test stability across folds and seeds. Report model-faithful latent counterfactuals rather than causal intervention claims.
