# V70 Patient-Specific Prognostic Circuits (PSPC-Surv)

`v70_patient_specific_prognostic_circuits` is an independent WSI--omics
survival hypothesis. It does not inherit V50/V60, instantiate Slot Attention,
run optimal transport, or create a fixed bank of event queries.

## Research question

V50 assumes fixed WSI slots, fixed omics slots, and fixed event tokens for every
patient. V70 asks a different question: can a cohort share a reusable library of
prognostic computations while each patient activates and connects only the
subset needed by their histology and molecular state?

The design is inspired by the sparse modularity of Neural Attentive Circuits,
but it is not a copy of the complete NAC architecture. The survival-specific
hypothesis is a patient-conditioned multimodal circuit with a time-local hazard
readout.

## Forward path

```text
WSI patches -> WSI projection -----------+
                                           +-> patient context
omics/pathways -> pathway encoders -------+
                                           |
                                           +-> module gates g_i
                                           +-> directed adjacency A_i

[WSI tokens, omics tokens, null token]
                |
                v
reusable module queries -- read-in attention
                |
                v
patient-specific sparse circuit execution
                |
                v
module x time hazard contributions -> patient hazard logits
```

The null token guarantees a numerically valid route when a modality is missing.
`wsi_available`, `omics_available`, `wsi_token_mask`, and `omics_token_mask`
are supported explicitly.

## Dynamic capacity and topology

`K_max` is a maximum reusable module capacity, not an assumed number of tumor
components. A hard-concrete gate produces a patient-specific active subset and
keeps at least one route open. A low-rank patient-conditioned edge generator
creates a directed graph over the active modules. Self-loops are guaranteed;
off-diagonal edges are learned with a straight-through hard decision.

The interpretation boundary is strict: a learned module is a computational
prognostic mechanism, not automatically a biological pathway or causal factor.

## Objective

The complete training objective has three terms:

```text
L = L_survival_NLL
  + lambda_node * E[active module fraction]
  + lambda_edge * E[active off-diagonal edge density]
```

The common trainer supplies `L_survival_NLL` exactly once. V70 returns only the
two structural regularizers. There is no OT, ranking, reconstruction,
specialization, coverage, or protective auxiliary loss in this first version.

## Run

```bash
python -m survot_rank.cli train \
  --config configs/v70_patient_specific_prognostic_circuits_blca.yaml
```

## Required evidence before paper claims

1. Same five BLCA folds as V50 and V60.
2. V70 versus a dense all-module/all-edge executor with matched dimensions.
3. Fixed graph versus patient-conditioned graph.
4. Capacity sensitivity for `K_max` in `{8, 16, 32}`.
5. Active-module and active-edge distributions across patients and folds.
6. Modality ablations and missing-modality tests.

Until these experiments are complete, V70 is a testable method hypothesis, not
a demonstrated improvement or a claim of first use.
