# Stagewise Prognostic Transport

`stagewise_prognostic_transport` is a separate experimental method line. It does not replace V31, V45, V50, or RG-ET.

## Core idea

RG-ET learns one prognostic pair cost shared by all event tokens. This method predicts a different WSI-omics pair cost for each event stage and solves a separate three-cost Sinkhorn transport problem for each stage:

```text
WSI slots + omics slots
        -> stage-specific prognostic costs
        -> stage-specific cosine/euclidean/dot OT plans
        -> matching stage event token
        -> Transformer hazard head
```

The stage index is represented by a fixed ordered embedding, while the auxiliary objective contains OT regularization, censoring-aware continuous ranking, and stage-order regularization.

## Positioning

This is a hypothesis-driven extension, not yet a proven contribution. The required ablations are:

1. shared RG-ET transport;
2. stage-specific transport without prognostic cost;
3. stage-specific transport with prognostic cost;
4. stage-specific transport with and without stage-order regularization.

Report repeated seeds and a last-five-epoch robust summary before making a novelty or performance claim.
