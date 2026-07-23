#!/usr/bin/env python3
"""Export patient-level DCT v3.6 transport explanations from a saved checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.research.methods.dct_listwise_transport.explanations import (
    build_patch_metadata,
    compose_patch_pathway_transport,
    export_case_explanations,
)
from survot_rank.training.extended_args import process_args_extended
from survot_rank.training.model_factory import get_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--epoch",
        type=int,
        help="Sinkhorn annealing epoch; defaults to the checkpoint's best curve epoch",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-patches", type=int, default=50)
    parser.add_argument("--top-pathways", type=int, default=20)
    parser.add_argument("--top-pairs", type=int, default=100)
    parser.add_argument("--coordinate-root")
    parser.add_argument("--slide-root")
    parser.add_argument("--skip-deletion-study", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _load_feature_lengths(dataset, slide_ids: list[str]) -> list[int]:
    lengths = []
    for slide_id in slide_ids:
        # Match the legacy loader's feature filename convention exactly.
        feature_path = Path(dataset.wsi_path) / f"{slide_id.rstrip('.svs')}.pt"
        if feature_path.exists():
            feature = torch.load(feature_path, map_location="cpu")
            lengths.append(int(feature.size(0)))
        else:
            lengths.append(int(dataset.dataset_factory.num_patches))
    return lengths


def _deterministic_patch_metadata(dataset, row) -> list[dict[str, object]]:
    slide_ids = [item.strip() for item in str(row["wsi"]).split(",") if item.strip()]
    slide_lengths = _load_feature_lengths(dataset, slide_ids)
    total_patches = int(sum(slide_lengths))
    target = int(dataset.dataset_factory.num_patches)
    real_count = min(target, total_patches)
    if real_count > 0:
        selected = np.floor(
            np.arange(real_count) * total_patches / real_count
        ).astype(np.int64)
    else:
        selected = np.empty(0, dtype=np.int64)
    if real_count < target:
        padded = np.arange(
            total_patches, total_patches + (target - real_count), dtype=np.int64
        )
        selected = np.concatenate([selected, padded])
    return build_patch_metadata(slide_ids, slide_lengths, selected)


def _risk(logits: torch.Tensor) -> torch.Tensor:
    hazards = torch.sigmoid(logits)
    return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)


def _resolve_checkpoint_epoch(
    checkpoint: str | Path,
    fold: int,
    explicit_epoch: int | None,
) -> int:
    if explicit_epoch is not None:
        if explicit_epoch < 0:
            raise ValueError("epoch must be non-negative")
        return int(explicit_epoch)
    curve_path = Path(checkpoint).parent / f"epoch_curve_fold{fold}.csv"
    if not curve_path.exists():
        raise FileNotFoundError(
            f"cannot infer checkpoint epoch without {curve_path}; pass --epoch"
        )
    curve = pd.read_csv(curve_path)
    if curve.empty or not {"epoch", "val_cindex"}.issubset(curve.columns):
        raise ValueError(f"invalid checkpoint epoch curve: {curve_path}")
    values = pd.to_numeric(curve["val_cindex"], errors="coerce")
    if not np.isfinite(values.to_numpy()).any():
        raise ValueError(f"epoch curve has no finite validation C-index: {curve_path}")
    best_position = int(np.nanargmax(values.to_numpy()))
    return int(curve["epoch"].iloc[best_position])


@torch.no_grad()
def _deletion_study(
    args,
    model,
    data,
    explanations,
    process_data_and_forward,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    factual_transport = compose_patch_pathway_transport(
        explanations["wsi_patch_to_global_pooling"],
        explanations["factual_stage_couplings"],
        explanations["omic_pathway_to_global_pooling"],
    )
    transport_scores = factual_transport.sum(dim=(1, 3))[0]
    attention_scores = explanations["wsi_patch_to_global_pooling"].sum(dim=1)[0]
    token_count = int(transport_scores.numel())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_scores = torch.rand(token_count, generator=generator).to(
        transport_scores.device
    )
    rankings = {
        "transport": torch.argsort(transport_scores, descending=True),
        "attention": torch.argsort(attention_scores, descending=True),
        "random": torch.argsort(random_scores, descending=True),
    }
    fractions = (0.05, 0.10, 0.20)
    results = {
        "deletion_fraction": transport_scores.new_tensor(fractions),
    }
    device = next(model.parameters()).device
    for name, order in rankings.items():
        risks = []
        for fraction in fractions:
            count = max(1, int(round(token_count * fraction)))
            modified = list(data)
            modified[0] = data[0].clone()
            modified[0][:, order[:count], :] = 0.0
            out, _, _, _ = process_data_and_forward(
                args, model, modified, device, test=True
            )
            risks.append(_risk(out[0]))
        results[f"{name}_deleted_risk"] = torch.stack(risks, dim=1)
    return results


def main() -> int:
    cli_args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    config = apply_overrides(load_config(cli_args.config), cli_args.set)
    parsed = process_args_extended(config_to_argv(config))
    parsed.survot_method = "dct_listwise_transport"
    parsed.newslot_method = parsed.survot_method
    parsed.k_start = cli_args.fold
    parsed.k_end = cli_args.fold + 1
    parsed.cur_fold = cli_args.fold
    parsed.cur_epoch = _resolve_checkpoint_epoch(
        cli_args.checkpoint, cli_args.fold, cli_args.epoch
    )
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
        method=parsed.survot_method,
        args=parsed,
        omic_input_dim=omics_input_dim,
        omic_names=parsed.omic_names,
        pathway_names=parsed.pathway_names,
    )
    model.configure_train_reference(
        train_data.label_df[dataset_factory.label_col].to_numpy(),
        train_data.label_df[dataset_factory.censorship_var].to_numpy(),
    )
    state_dict = torch.load(cli_args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    output_root = Path(
        cli_args.output_dir
        or (
            Path("results")
            / "dct_v3.6_explanations"
            / parsed.dct_listwise_mode
            / parsed.study
            / f"fold{cli_args.fold}"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    pathway_names = list(getattr(dataset_factory, "pathway_names", []))

    exported = 0
    for batch_idx, data in enumerate(val_loader):
        if cli_args.max_cases > 0 and exported >= cli_args.max_cases:
            break
        out, _, _, _ = _process_data_and_forward(
            parsed, model, data, device, test=True
        )
        logits, _ = out
        explanations = {
            key: value.detach().cpu()
            for key, value in model.explain_last_batch().items()
            if isinstance(value, torch.Tensor)
        }
        sweep = {
            key: value.detach().cpu()
            for key, value in model.counterfactual_sweep().items()
        }
        if not cli_args.skip_deletion_study:
            sweep.update(
                {
                    key: value.detach().cpu()
                    for key, value in _deletion_study(
                        parsed,
                        model,
                        data,
                        explanations,
                        _process_data_and_forward,
                        seed=1729 + batch_idx,
                    ).items()
                }
            )

        row = val_data.label_df.iloc[batch_idx]
        case_id = str(row["case id"])
        patch_metadata = _deterministic_patch_metadata(val_data, row)
        export_case_explanations(
            case_id,
            explanations,
            output_root,
            patch_metadata=patch_metadata,
            pathway_names=pathway_names or None,
            sweep=sweep,
            top_patches=cli_args.top_patches,
            top_pathways=cli_args.top_pathways,
            top_pairs=cli_args.top_pairs,
            coordinate_root=cli_args.coordinate_root,
            slide_root=cli_args.slide_root,
            force=cli_args.force,
        )
        exported += 1
        print(
            f"[export] {case_id}: risk={float(_risk(logits).item()):.6f} "
            f"-> {output_root / case_id}"
        )

    print(f"[done] exported {exported} cases to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
