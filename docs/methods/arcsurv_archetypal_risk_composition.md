# ArcSurv: Archetypal Risk Composition

## Main claim

ArcSurv models a patient as a convex composition of cohort-level prognostic
extreme states. It is not a prototype-centre classifier and does not claim that
archetypal analysis is new to cancer or multi-omics.

The method changes the prediction object from patient-specific event slots to a
shared prognostic simplex:

```text
WSI patches -> WSI slots ----> WSI archetype composition ---+
                                                             +-> patient composition -> hazard logits
pathways    -> omics slots --> omics archetype composition -+
```

Each modality-specific archetype is constrained to be a convex combination of
a fold-local training memory:

```text
A_m = Beta_m Z_m,  Beta_m >= 0,  rows(Beta_m) sum to 1
```

The patient composition is also row-stochastic. Its convex combination of
archetype-specific hazard curves directly produces the final time-bin logits.
There is no unconstrained residual prediction path that can bypass the
simplex.

## Objectives

The external training loop still supplies the standard censored survival NLL.
The model returns four compact auxiliary terms:

1. archetypal reconstruction of modality states;
2. Jensen-Shannon agreement between WSI and omics compositions;
3. a weak cohort balance regularizer to prevent immediate simplex collapse;
4. censor-aware pairwise ranking.

## Fold and leakage rule

The archetype memory belongs to one model instance and is filled only while the
model is in training mode. Validation and test forwards never update it. Since
the framework creates a fresh model per fold, no test patient can enter the
archetype hull.

## Missing modalities

Dual-modality patients average the two simplex compositions. Single-modality
patients use the available composition. Patients with neither modality use a
learned missing-input hazard vector. Slot masks are respected during patient
state pooling.

## Entry points

```bash
python -m survot_rank.cli train --config configs/archetypal_risk_composition_blca.yaml
```

Public method names:

- `archetypal_risk_composition`
- `arcsurv`
- `arc_surv`

## Required ablations

- unconstrained learnable prototypes instead of `Beta @ memory`;
- no cross-modal composition agreement;
- one-modality compositions;
- different numbers of archetypes;
- archetype stability across folds and bootstrap resamples;
- comparison against MMP-style prototype aggregation.

The paper-facing novelty boundary is the prognostic extreme-state composition
and its direct hazard parameterization, not the use of Slot Attention.
