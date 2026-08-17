# DCT v3.10 paper evidence ledger

This ledger collects the C-index numbers and audit metrics produced by the paper-evidence ablation launchers and the audit loader. It is the canonical snapshot for DCT v3.10. Numbers are pulled from raw ``split_<fold>_results_final.pkl`` files and audit JSON outputs; if a cell is empty the run has not completed yet.

## 0. Frozen-recipe inventory

| Recipe | Method key | File | Aux-loss set |
|---|---|---|---|
| **v3.10 DCT-Reg (paper mainline)** | `dct_v310_directional_regularized_transport` | `configs/dct_v310_directional_regularized_transport.yaml` | NLL + 0.10·IPCW + 0.05·direction |
| v3.8.2 frozen-full (historical) | `dct_v382_prognostic_transport_reconstruction` | `configs/distributional_counterfactual_transport_<cancer>.yaml` | NLL + IPCW + direction + dose + reconfiguration + MGPTR |
| v3.8.2 minimal (development precursor) | `dct_v382_minimal_transport` | `configs/dct_v382_minimal_transport_blca.yaml` | NLL + IPCW + direction |

DCT v3.10 freezes the smallest selected objective under a new method identity.
Historical results remain historical until the matched v3.10 experiment queue
finishes.

## A. Score-first ablation C-index (3 cancers × fold 1)

| Variant | Cancer | Fold | C-index | C-index IPCW | n |
|---|---|---:|---:|---:|---:|

## B. Audit metrics (factual risk vs counterfactual risk)

_No audit results available yet. Run::

    python scripts/audit_dct_v382.py audit --config <cfg> --checkpoint <ckpt> --fold <f> --output-dir <out>_

## C. Cross-cancer shared prototype transfer (BLCA source → ?)

| Pair | Target cancer | Fold | C-index (target) |
|---|---|---:|---:|

## D. Evaluation criteria (paper-facing)

- Compare paired outer-fold differences with confidence intervals; do not use arbitrary post-hoc drop thresholds.
- `fixed_coupling`, `noisy_batch_mean_anchors` and `permuted_reference` must weaken the held-out directional response if their corresponding mechanism is load-bearing.
- Every re-solved or replayed coupling must satisfy the same marginal-error tolerance.
- Shuffled/uniform feasible-plan audits must test whether the decoder functionally uses the coupling.

## E. Frozen one-sentence claim

> **DCT v3.10 actively regularizes the re-solved transport-risk response to be directionally consistent under training-fold prognostic cost interventions.**

The claim remains pending until the matched objective and mechanism experiments
in `docs/DCT_V310_DCT_REG.md` are completed.
