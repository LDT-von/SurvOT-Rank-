# Faithful Evidence Transport

`faithful_evidence_transport` is a separate method line for interpretable WSI-pathway survival modeling. It leaves the previous method implementations unchanged.

## Evidence path

```text
WSI patches -> WSI slots
pathways    -> pathway slots
                  |
       stage-specific OT plan
                  |
       evidence gate on slot pairs
                  |
       event-stage hazard
```

The model stores `stage_slot_pair_evidence`, WSI-to-slot assignments, pathway-to-slot assignments, and event gates in `model.explain_last_batch()`. These tensors can be mapped to patch clusters and pathway names without treating raw attention weights as explanations.

## Faithfulness objective

The evidence gate changes the transport plan used by the event token. A counterfactual plan suppresses the selected evidence, and the auxiliary objective requires the prediction to change by at least a margin. An entropy term encourages a compact evidence set.

## Required evaluation

Report sufficiency and comprehensiveness tests: retaining the top evidence should preserve the prediction, removing it should change the prediction more than removing random evidence, and explanations should be stable across seeds. Do not claim causal explanations; this method provides model-faithful prognostic evidence.
