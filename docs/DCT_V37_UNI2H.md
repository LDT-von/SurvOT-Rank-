# DCT v3.7-UNI2H

DCT v3.7-UNI2H keeps the score-first DCT objective unchanged and changes the
WSI input protocol to the extracted UNI2-h features. Its default `highscore`
variant matches the historical v3.3 and prior-work full-cohort discretization
protocol so that UNI2-h is the only intended experimental change.

## Shared protocol

- WSI encoder: `uni2-h`
- WSI feature dimension: `1536`
- WSI input: HDF5 `features` with shape `(1, N, 1536)` or `(N, 1536)`
- Patches sampled per patient: `2048`
- Objective: `NLL + 0.1 * IPCW ranking`
- ETAR, listwise losses, event-stratified batching, and legacy auxiliary losses: off

## Variants

| Variant | Survival bins | Slots | Purpose |
|---|---|---|---|
| `highscore` (default) | full-cohort event qcut | gaussian | v3.3 high-score protocol with only UNI2-h changed |
| `clean` | train-fold-only event qcut | deterministic | strict leakage audit control |

The eleven extracted TCGA archives map to ten DCT study keys because COAD and
READ are evaluated together as `coadread`.

## Commands

No cancer training is run by these checks:

```bash
python scripts/run_dct_v37_uni2h_screen.py doctor
python scripts/run_dct_v37_uni2h_screen.py plan --variants highscore --cancers blca,brca --folds 0,2
```

Server smoke screening:

```bash
python scripts/run_dct_v37_uni2h_screen.py smoke --variants highscore --cancers blca,brca --folds 0,2
```

Fold0/fold2 screening across all available cancers:

```bash
python scripts/run_dct_v37_uni2h_screen.py run
```

The default command runs only `highscore`. To run the audit control explicitly:

```bash
python scripts/run_dct_v37_uni2h_screen.py run --variants clean
```

Results are isolated under `results/dct_v3.7_uni2h/<variant>/<cancer>`.
Existing DCT v3.3, v3.5, Recovery, ETAR, and v3.6 results are neither reused nor
overwritten.
