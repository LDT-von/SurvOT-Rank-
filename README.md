# SurvOT-Rank

SurvOT-Rank is a cleaned framework for WSI-pathway survival modeling. The
current paper-facing method is **V45 / `otehv2_rankevent`**, which builds latent
prognostic event tokens from cross-modal optimal transport and trains them with
event-level survival supervision plus pairwise ranking.

The repository keeps the original SlotSPE-based experiment backend for
reproducibility, but new work should use the unified package, YAML configs, and
short script wrappers.

## Clean Entry Points

```bash
# Check whether the expected project files exist
python -m survot_rank.cli doctor

# Default V45 BLCA run
python -m survot_rank.cli train --config configs/v45_blca.yaml

# Tuned BLCA run from the fold-3 smoke search
python -m survot_rank.cli train --config configs/v45_best_blca.yaml

# Quick smoke run
python -m survot_rank.cli train --config configs/smoke_v45_blca.yaml

# Override config values without editing YAML
python -m survot_rank.cli train --config configs/v45_blca.yaml --set seed=5 --set gpu=1

# Ensemble evaluation for multiple seed result folders
python -m survot_rank.cli ensemble --dirs results/seed3 results/seed5
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_v45_blca.ps1 -Config configs/v45_blca.yaml -Gpu 0 -Seed 3
```

Linux helper:

```bash
bash scripts/train_v45_blca.sh configs/v45_blca.yaml
```

## Directory Layout

```text
survot_rank/                  clean Python package, CLI, and research code
configs/                      experiment YAML files
scripts/                      short run wrappers
scripts/legacy/               old root-level run_*.sh scripts
docs/                         framework and migration notes
survot_rank/training/         training runner, args, paths, and model factory
survot_rank/research/methods/prognostic_event_transport/
                              main PET method, formerly V45
survot_rank/research/methods/ot_event_hazard_v2/
                              parent OT-event model, formerly V31
survot_rank/research/components/
                              copied-in model components used by PET
survot_rank/research/legacy/slotspe_runtime/
                              minimal SlotSPE runtime for data/loss compatibility
tools/                        old sweep, monitor, and data utilities
important_outputs/            packaged reproduction artifacts
閲嶈鏂囦欢/                     historical experiment notes
```

## Main Code Path

```text
configs/*.yaml
  -> survot_rank.cli train
  -> survot_rank.training.train_runner
  -> survot_rank.training.model_factory
  -> prognostic_event_transport.OTEHV2RankEvent
  -> ot_event_hazard_v2.OTEventHazardV2Survival
  -> SlotSPE dataset/loss utilities
```

## Data Expectations

The default configs expect:

```text
survot_rank/research/legacy/slotspe_runtime/dataset_csv/
  clinical/all/blca.csv
  raw_rna_data_inter/blca_rna_inter.csv
  splits/5fold/blca/fold_{0..4}.csv

/data/CPathPatchFeature/blca/uni/pt_files/*.pt
```

Adjust `data_root_dir` and `data_path` in YAML if your data lives elsewhere.

## Notes

- PET no longer imports model layers from SlotSPE. Its slot attention and omics
  encoder live in `survot_rank/research/components/`.
- `survot_rank/research/legacy/slotspe_runtime/` is kept only for the old
  dataset, loss, and metric helpers still used by the training loop.
- New experiments should be added as `configs/*.yaml`, not new root-level shell
  scripts.
- See `docs/FRAMEWORK.md` and `docs/MIGRATION.md` for the cleanup map.
