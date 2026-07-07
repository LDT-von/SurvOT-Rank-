# SurvOT-Rank Framework

This repository now has a clean outer framework around the original experiment
code. The old files are kept as the legacy backend so historical runs can still
be reproduced.

## Directory Roles

```text
survot_rank/                 clean Python entrypoints and research code
configs/                     YAML experiment configs
scripts/                     short run wrappers
survot_rank/training/        training runner, args, paths, and model factory
survot_rank/research/methods/prognostic_event_transport/
                             main PET method, formerly V45
survot_rank/research/methods/ot_event_hazard_v2/
                             parent OT-event model, formerly V31
survot_rank/research/components/
                             copied-in model components used by PET
survot_rank/research/legacy/slotspe_runtime/
                             minimal SlotSPE runtime for data/loss compatibility
tools/                       utility scripts for old sweeps and monitoring
important_outputs/           packaged reproduction artifacts
閲嶈鏂囦欢/                    historical experiment notes
```

## Main Commands

Check expected files:

```bash
python -m survot_rank.cli doctor
```

Run the default V45 BLCA experiment:

```bash
python -m survot_rank.cli train --config configs/v45_blca.yaml
```

Run the tuned BLCA config:

```bash
python -m survot_rank.cli train --config configs/v45_best_blca.yaml
```

Override a field without editing YAML:

```bash
python -m survot_rank.cli train --config configs/v45_blca.yaml --set seed=5 --set gpu=1
```

Evaluate a multi-seed ensemble:

```bash
python -m survot_rank.cli ensemble --dirs results/seed3 results/seed5
```

## Development Rule

New experiments should start from `configs/*.yaml` and the `survot_rank.cli`
entrypoint. Keep old shell scripts only as historical references. This makes the
paper-facing path easy to audit:

```text
config -> unified CLI -> survot_rank.training.train_runner -> model_factory -> PET model
```
