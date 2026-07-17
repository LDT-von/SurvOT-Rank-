# SurvOT-Rank

SurvOT-Rank is a cleaned framework for WSI-pathway survival modeling. The
current compact method is **V60 / `v60_ot_event_rank`**. V60 keeps the stable
OT-event backbone from V9/V45, uses log-domain Sinkhorn transport and an OT
auxiliary loss, then adds only per-event survival supervision and censor-aware
Cox-style ranking.

V45 / `otehv2_rankevent` remains available as the legacy composite reference.
It includes additional global residual, consistency, gate-entropy, and epsilon
annealing components, so it is useful for historical comparison but is not the
compact paper-facing default.

The repository keeps the original SlotSPE-based experiment backend for
reproducibility, but new work should use the unified package, YAML configs, and
short script wrappers.

## Clean Entry Points

```bash
# Check whether the expected project files exist
python -m survot_rank.cli doctor

# Default V60 BLCA run
python -m survot_rank.cli train --config configs/v60_ot_event_rank_blca.yaml

# Legacy V45 BLCA reference
python -m survot_rank.cli train --config configs/v45_blca.yaml

# Tuned BLCA run from the fold-3 smoke search
python -m survot_rank.cli train --config configs/v45_best_blca.yaml

# V45 quick smoke run
python -m survot_rank.cli train --config configs/smoke_v45_blca.yaml

# Override config values without editing YAML
python -m survot_rank.cli train --config configs/v45_blca.yaml --set seed=5 --set gpu=1

# Experimental CA-PSA: shared cohort slot identities + patient-adaptive count
python -m survot_rank.cli train --config configs/cohort_anchored_adaptive_prognostic_slot_attention_blca.yaml

# Ensemble evaluation for multiple seed result folders
python -m survot_rank.cli ensemble --dirs results/seed3 results/seed5
```

V60 method alias:

```bash
python -m survot_rank.cli train --config configs/v60_ot_event_rank_blca.yaml --set survot_method=60
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
survot_rank/research/methods/v60_ot_event_rank/
                              compact V60 paper-facing method
survot_rank/research/methods/cohort_anchored_adaptive_prognostic_slot_attention/
                              experimental cohort-anchored adaptive slots
survot_rank/research/methods/prognostic_event_transport/
                              legacy composite PET method, formerly V45
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
  -> v60_ot_event_rank.V60OTEventRank
  -> ot_event_hazard_v2.OTEventHazardV2Survival
  -> SlotSPE dataset/loss utilities
```

## V60 Objective

```text
L = L_survival
  + lambda_ot * L_OT
  + lambda_per_event * L_per_event_NLL
  + lambda_rank * L_censor_aware_rank
```

The OT plans use log-domain Sinkhorn updates. V60 does not use the V45 global
residual head, gate-entropy penalty, global consistency loss, or epsilon
annealing schedule.

## Missing Modalities

V60 accepts optional availability and slot masks:

```text
wsi_available:  [B]
omics_available: [B]
wsi_slot_mask:  [B, num_wsi_slots]
omics_slot_mask: [B, num_omics_slots]
```

Complete samples use masked OT with renormalized valid-slot marginals. Samples
with only one modality use a modality-specific fallback head; samples with no
available modality use a learned missing-data fallback. Invalid slots receive
zero transport mass. If the masks are omitted, all modalities and slots are
treated as available for backward compatibility.

## Verification

The V60 implementation is covered by focused forward, backward, registration,
masked-OT, missing-modality, and censoring edge-case tests. The current full
test suite passes with 189 tests. This is code-level verification; real-data
five-fold performance for V60 still needs to be run before making a paper
performance claim.

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
