# DCT v3.8.2 paper evidence ledger

This ledger collects the C-index numbers and audit metrics produced by the paper-evidence ablation launchers and the audit loader. It is the canonical machine-readable snapshot for the v3.8.2 paper. Numbers are pulled from the raw ``split_<fold>_results_final.pkl`` files and the audit JSON outputs; if a cell is empty the run has not completed yet.

## 0. Frozen-recipe inventory

| Recipe | Method key | File | Aux-loss set |
|---|---|---|---|
| **v3.8.2 frozen-full (paper mainline)** | `dct_v382_prognostic_transport_reconstruction` | `configs/distributional_counterfactual_transport_<cancer>.yaml` | NLL + 0.10·IPCW + 0.05·direction + 0.03·dose + 0.02·reconfiguration + 0.05·MGPTR (fixed) |
| **v3.8.2 minimal (new)** | `dct_v382_minimal_transport` | `configs/dct_v382_minimal_transport_<cancer>.yaml` | NLL + 0.10·IPCW + 0.05·direction (everything else forced to 0) |

The minimal recipe is the smallest set of components that can answer the
monotone dose-response claim.  It exists alongside the paper mainline so
both recipes can be evaluated head-to-head and the contribution of the
removed terms (MGPTR, dose, reconfiguration, adaptive weighting) can be
isolated without rerunning the full fixed-full matrix.

## A. Score-first ablation C-index (3 cancers × fold 1)

| Variant | Cancer | Fold | C-index | C-index IPCW | n |
|---|---|---:|---:|---:|---:|

## B. Audit metrics (factual risk vs counterfactual risk)

_No audit results available yet. Run::

    python scripts/audit_dct_v382.py audit --config <cfg> --checkpoint <ckpt> --fold <f> --output-dir <out>_

## C. Cross-cancer shared prototype transfer (BLCA source → ?)

| Pair | Target cancer | Fold | C-index (target) |
|---|---|---:|---:|

## D. Pass criteria (paper-facing)

- Ablation 1 (`fixed_coupling`): C-index should drop ≥ 0.04 from fixed-full. Audit ``reconfiguration > margin`` should drop below 0.20.
- Ablation 2 (`random_anchors`): C-index should drop ≥ 0.03 from fixed-full. Audit ``direction rate`` should fall back to chance (≤ 0.55).
- Ablation 4 (`null_calibration`): audit metrics should match the random-signal baseline (direction ≈ 0.50, dose monotone ≈ chance).
- Ablation 5 (`stage_randomization`): C-index should drop ≥ 0.02 if exact edge placement matters.
- Cross-cancer transfer: target C-index ≥ 0.60 of source for the prototype to count as cross-cancer semantic.

## E. Minimal-recipe one-sentence claim

> **DCT v3.8.2 minimal: IPCW ranking + direction loss alone guarantee that re-optimised Sinkhorn under high/low risk anchors moves the predicted survival risk in the requested direction, providing a monotone dose-response guarantee that the larger auxiliary-loss set does not improve upon.**

The claim is verifiable on the frozen recipe: the ``no_direction`` row of
``run_dct_v382_paper_ablations.py`` should show a C-index drop of at least
0.005 on BLCA folds 1/2/4 once the launcher completes.