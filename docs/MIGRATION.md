# Migration Notes

The original repository used many root-level shell scripts and method folders.
To avoid breaking old results, the first cleanup pass adds a clean layer instead
of rewriting every import.

## Old to New

| Old style | New style |
|---|---|
| `bash run_v45_final.sh` | `python -m survot_rank.cli train --config configs/v45_blca.yaml` |
| `bash run_best_blca.sh` | `python -m survot_rank.cli train --config configs/v45_best_blca.yaml` |
| edit shell variables | edit `configs/*.yaml` or use `--set key=value` |
| direct old V45 `ensemble_eval.py` | `python -m survot_rank.cli ensemble --dirs ...` |

## What Is Still Legacy

- `-m survot_rank.training.train_runner` remains the training backend.
- `survot_rank/training/model_factory.py` resolves the main method from
  `survot_rank/research/methods/prognostic_event_transport`.
- PET model layers are copied into `survot_rank/research/components`.
- The old SlotSPE code is reduced to a minimal runtime at
  `survot_rank/research/legacy/slotspe_runtime`.

The next cleanup pass can move model implementations under a single
`survot_rank/models/` namespace after the paper-facing mainline is settled.
