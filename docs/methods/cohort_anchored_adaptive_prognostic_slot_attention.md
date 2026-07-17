# Cohort-Anchored Adaptive Prognostic Slot Attention (CA-PSA)

`cohort_anchored_adaptive_prognostic_slot_attention` is an experimental
WSI--omics survival method. Its target is a specific weakness of ordinary Slot
Attention in cohort modelling: independently sampled, exchangeable slots have
no stable meaning across patients or modalities, and a fixed slot count asserts
the same latent complexity for every patient.

## Mechanism

For patient `i`, modality `m`, and cohort identity `k`, CA-PSA constructs

```text
z_i,k,m = a_k + s_i,k,m
```

where `a_k` is one learnable cohort anchor shared by WSI and omics, while
`s_i,k,m` is produced by a modality-specific competitive recurrent update. The
two modalities therefore start from the exact same indexed identity without a
post-hoc `K x K` OT matching step.

The same-index WSI and omics slots are fused as

```text
[z_w, z_o, z_w * z_o, abs(z_w - z_o)] -> slot feature
```

Each slot emits discrete-time hazard logits and time-specific mixture weights.
A hard-concrete gate selects a patient-specific subset from `K_max`; a one-slot
safety constraint prevents an empty prediction route. A learnable ordered
capacity prior avoids an all-on or all-off initial validation pass, while the
patient feature determines the posterior activation. `K_max=16` is a capacity
limit, not a claim that every patient has sixteen prognostic factors.

## Objective

The complete training objective has exactly three terms:

```text
L = L_NLL + lambda_sparse * L_expected_L0
          + lambda_align * L_same_identity_alignment
```

`L_NLL` is supplied once by the common trainer. The model returns only the two
auxiliary terms. Alignment is computed on modality-specific patient states, not
on the shared anchor itself, and is restricted to patients with both modalities
available.

## Run

```bash
python -m survot_rank.cli train \
  --config configs/cohort_anchored_adaptive_prognostic_slot_attention_blca.yaml
```

The first score-oriented defaults use batch size 8, 50 epochs, AdamW, five
warm-up epochs, gradient clipping, and a small auxiliary scale. Recommended
ablation order is: fixed all-on gates; independent modality anchors; no state
alignment; and the complete CA-PSA model.

## Novelty boundary and claims

CA-PSA is influenced by dynamic slot selection, learned-query initialization,
and identity/state separation in prior Slot Attention research. Its proposed
contribution is the unified survival-specific combination of cohort-shared
cross-modal identity, patient state, and adaptive prognostic capacity. It does
not claim that a learned slot is a biological pathway, a causal mechanism, or a
globally unique factor without external validation. The code should be treated
as a new hypothesis until full five-fold results and the listed ablations are
available.
