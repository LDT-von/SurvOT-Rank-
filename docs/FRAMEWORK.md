# SurvOT-Rank Framework

The repository uses one Python package and one public CLI around the preserved
SlotSPE-compatible runtime. Historical implementations remain available for
reproducibility, while current method identity is maintained in a single
catalog.

## Directory Roles

```text
survot_rank/                  Python package and public CLI
survot_rank/training/         training runner, argument parser, model factory
survot_rank/research/methods/ method implementations and catalog.py
survot_rank/research/components/
                              shared research components
survot_rank/research/legacy/slotspe_runtime/
                              minimal SlotSPE data/loss compatibility layer
configs/                      experiment instances; see configs/INDEX.md
scripts/                      queues, screens, monitors and exporters
docs/                         documentation map in docs/README.md
tools/                        data and historical utility programs
important_outputs/            packaged reproduction artifacts
重要文件/                       historical experiment notes
```

Method folders are executable code locations, not paper-priority labels. Use
`docs/METHODS.md` or the catalog command for current roles.

## Main Commands

```bash
# List canonical method names, aliases and current research roles
python -m survot_rank.cli methods

# Check package, data compatibility layer and all catalog code paths
python -m survot_rank.cli doctor

# Run any YAML-backed experiment
python -m survot_rank.cli train --config <config.yaml>

# Override a field without editing the YAML
python -m survot_rank.cli train --config configs/v45_blca.yaml --set seed=5 --set gpu=1

# Evaluate a multi-seed ensemble
python -m survot_rank.cli ensemble --dirs results/seed3 results/seed5
```

DCT v3.10 DCT-Reg is the current paper mainline. Its exact three-term objective
is frozen by both the model class and the dedicated cross-cancer launcher. V45 and V60
OT Event Rank remain useful config-driven references; they are not the current
paper-priority label.

## Main Code Path

```text
YAML or a frozen launcher
  -> survot_rank.cli / generated argparse arguments
  -> survot_rank.training.train_runner
  -> survot_rank.training.model_factory
  -> survot_rank.research.methods.catalog
  -> selected method implementation
  -> SlotSPE-compatible dataset and survival losses
```

The catalog owns canonical names, aliases, class paths and current research
roles. `extended_args.py` derives its valid method choices from the same
catalog, so adding a method no longer requires maintaining a second list.

## Development Rule

A new method is complete only when it has:

1. one independent implementation directory;
2. one entry in `survot_rank/research/methods/catalog.py`;
3. a representative config or a launcher that records all dynamic overrides;
4. a mechanism document linked from `docs/METHODS.md`;
5. focused registration/forward tests.

Do not add a standalone `run_*.py` without registering and documenting the
method. Do not move existing script modules without compatibility wrappers,
because formal queues and tests import several of them by their current paths.
