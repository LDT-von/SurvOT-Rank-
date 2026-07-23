# DCT v3.7-UNI2H

DCT v3.7-UNI2H keeps the score-first DCT objective unchanged and changes the
WSI input protocol to the extracted UNI2-h features.

## Fixed protocol

- WSI encoder: `uni2-h`
- WSI feature dimension: `1536`
- WSI input: HDF5 `features` with shape `(1, N, 1536)` or `(N, 1536)`
- Patches sampled per patient: `2048`
- Survival bins: train-fold-only event quantiles
- Slots: deterministic evaluation
- Objective: `NLL + 0.1 * IPCW ranking`
- ETAR, listwise losses, event-stratified batching, and legacy auxiliary losses: off

The eleven extracted TCGA archives map to ten DCT study keys because COAD and
READ are evaluated together as `coadread`.

## Commands

No cancer training is run by these checks:

```bash
python scripts/run_dct_v37_uni2h_screen.py doctor
python scripts/run_dct_v37_uni2h_screen.py plan --cancers blca,brca --folds 0,2
```

Server smoke screening:

```bash
python scripts/run_dct_v37_uni2h_screen.py smoke --cancers blca,brca --folds 0,2
```

Fold0/fold2 screening across all available cancers:

```bash
python scripts/run_dct_v37_uni2h_screen.py run
```

Results are isolated under `results/dct_v3.7_uni2h/<cancer>`. Existing DCT
v3.3, v3.5, Recovery, ETAR, and v3.6 results are neither reused nor overwritten.
