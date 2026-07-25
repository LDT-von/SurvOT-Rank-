#!/usr/bin/env python3
"""Export exact IST-Surv v4.0 patch-pathway risk decompositions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from survot_rank.config import apply_overrides, config_to_argv, load_config  # noqa: E402
from survot_rank.research.methods.dct_listwise_transport.explanations import (  # noqa: E402
    build_patch_metadata,
)
from survot_rank.research.methods.intervention_stable_survival_transport.explanations import (  # noqa: E402
    export_case_explanations,
)
from survot_rank.training.extended_args import process_args_extended  # noqa: E402
from survot_rank.training.model_factory import get_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-pairs", type=int, default=100)
    parser.add_argument("--coordinate-root")
    parser.add_argument("--slide-root")
    parser.add_argument("--skip-deletion-study", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _feature_length(dataset, slide_id: str) -> int:
    feature_path = dataset._resolve_wsi_feature_path(slide_id)
    if feature_path is None:
        return int(dataset.dataset_factory.num_patches)
    feature_path = Path(feature_path)
    if feature_path.suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(feature_path, "r") as handle:
            shape = tuple(handle["features"].shape)
        return int(shape[-2])
    loaded = torch.load(feature_path, map_location="cpu")
    if isinstance(loaded, dict):
        loaded = loaded["features"]
    return int(loaded.shape[-2])


def _deterministic_patch_metadata(dataset, row) -> list[dict[str, object]]:
    slide_ids = [
        item.strip()
        for item in str(row["wsi"]).split(",")
        if item.strip()
    ]
    slide_lengths = [
        _feature_length(dataset, slide_id) for slide_id in slide_ids
    ]
    total = int(sum(slide_lengths))
    target = int(dataset.dataset_factory.num_patches)
    real_count = min(target, total)
    if real_count:
        selected = np.floor(
            np.arange(real_count) * total / real_count
        ).astype(np.int64)
    else:
        selected = np.empty(0, dtype=np.int64)
    if real_count < target:
        selected = np.concatenate(
            [
                selected,
                np.arange(
                    total,
                    total + target - real_count,
                    dtype=np.int64,
                ),
            ]
        )
    return build_patch_metadata(slide_ids, slide_lengths, selected)


def _slice_batch_mapping(
    mapping: dict[str, torch.Tensor],
    index: int,
    batch_size: int,
    *,
    global_keys: set[str],
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in mapping.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu()
        if (
            key not in global_keys
            and tensor.ndim > 0
            and tensor.size(0) == batch_size
        ):
            tensor = tensor[index : index + 1]
        result[key] = tensor
    return result


def main() -> int:
    cli_args = build_parser().parse_args()
    os.chdir(PROJECT_ROOT)

    config = apply_overrides(load_config(cli_args.config), cli_args.set)
    parsed = process_args_extended(config_to_argv(config))
    parsed.survot_method = "intervention_stable_survival_transport"
    parsed.newslot_method = parsed.survot_method
    parsed.k_start = cli_args.fold
    parsed.k_end = cli_args.fold + 1
    parsed.cur_fold = cli_args.fold
    parsed.cur_epoch = 0
    parsed.num_workers = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu)

    from survot_rank.training.train_runner import (
        SurvivalDatasetFactory,
        _process_data_and_forward,
        get_split,
    )

    clinical_feature_cols = None
    if getattr(parsed, "clinical_feature_cols", None):
        clinical_feature_cols = [
            value.strip()
            for value in parsed.clinical_feature_cols.split(",")
            if value.strip()
        ]
    dataset_factory = SurvivalDatasetFactory(
        study=parsed.study,
        data_path=parsed.data_path,
        rna_format=parsed.rna_format,
        signature=parsed.signature,
        n_bins=parsed.n_classes,
        label_col=parsed.label_col,
        num_genes=parsed.num_genes,
        num_patches=parsed.num_patches,
        clinical_feature_cols=clinical_feature_cols,
        binning_mode=getattr(parsed, "binning_mode", "global_qcut"),
    )
    if parsed.rna_format in ("Pathways", "RNASeq", "GeneEmbedding"):
        rna_cases = set(dataset_factory.gene_data_df.columns)
        dataset_factory.clinical_df = dataset_factory.clinical_df[
            dataset_factory.clinical_df["case id"].isin(rna_cases)
        ].reset_index(drop=True)

    train_data, val_data, _, val_loader = get_split(
        parsed, dataset_factory, cli_args.fold
    )
    del train_data
    parsed.omic_sizes = dataset_factory.omic_sizes
    parsed.omic_names = dataset_factory.omic_names
    parsed.pathway_names = getattr(dataset_factory, "pathway_names", None)
    if parsed.rna_format == "RNASeq":
        omics_input_dim = (
            dataset_factory.num_genes
            if dataset_factory.num_genes is not None
            else dataset_factory.omic_sizes
        )
    elif parsed.rna_format == "GeneEmbedding":
        omics_input_dim = 768
    else:
        omics_input_dim = None

    model = get_model(
        parsed.survot_method,
        parsed,
        omic_input_dim=omics_input_dim,
        omic_names=parsed.omic_names,
        pathway_names=parsed.pathway_names,
    )
    state = torch.load(cli_args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    output_root = Path(
        cli_args.output_dir
        or (
            Path("results")
            / "ist_surv_v4.0_explanations"
            / parsed.study
            / f"fold{cli_args.fold}"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    pathway_names = list(getattr(dataset_factory, "pathway_names", []) or [])

    exported = 0
    dataset_index = 0
    with torch.no_grad():
        for data in val_loader:
            out, _, _, _ = _process_data_and_forward(
                parsed, model, data, device, test=True
            )
            logits, _ = out
            batch_size = int(logits.size(0))
            all_explanations = model.explain_last_batch()
            all_sweep = (
                {}
                if cli_args.skip_deletion_study
                else model.deletion_sweep(seed=1729 + dataset_index)
            )

            for local_index in range(batch_size):
                if cli_args.max_cases > 0 and exported >= cli_args.max_cases:
                    break
                row = val_data.label_df.iloc[dataset_index]
                case_id = str(row["case id"])
                explanations = _slice_batch_mapping(
                    all_explanations,
                    local_index,
                    batch_size,
                    global_keys={"stage_bias"},
                )
                sweep = _slice_batch_mapping(
                    all_sweep,
                    local_index,
                    batch_size,
                    global_keys={"fractions"},
                )
                export_case_explanations(
                    case_id,
                    explanations,
                    output_root,
                    patch_metadata=_deterministic_patch_metadata(
                        val_data, row
                    ),
                    pathway_names=pathway_names or None,
                    deletion_sweep=sweep or None,
                    top_pairs=cli_args.top_pairs,
                    coordinate_root=cli_args.coordinate_root,
                    slide_root=cli_args.slide_root,
                    force=cli_args.force,
                )
                exported += 1
                dataset_index += 1
                print(f"[export] {case_id} -> {output_root / case_id}")
            if cli_args.max_cases > 0 and exported >= cli_args.max_cases:
                break

    print(f"[done] exported {exported} cases to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
