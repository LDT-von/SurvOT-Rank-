"""Auditable case-level exports for intervention-stable survival transport."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from survot_rank.research.methods.dct_listwise_transport.explanations import (
    find_coordinate_file,
    load_patch_coordinates,
)


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _json_values(value) -> list:
    return _numpy(value).tolist()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_signed_patch_heatmap(
    coordinates: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
    *,
    slide_path: str | Path | None = None,
) -> bool:
    """Render a signed coordinate map, optionally over the source WSI."""

    import matplotlib.pyplot as plt

    coords = np.asarray(coordinates, dtype=float)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coordinates must have shape [patches, >=2]")
    if coords.shape[0] != values.size:
        raise ValueError("coordinate and score counts do not match")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8))
    used_overlay = False
    plot_x, plot_y = coords[:, 0], coords[:, 1]
    if slide_path is not None and Path(slide_path).exists():
        try:
            import openslide

            slide = openslide.OpenSlide(str(slide_path))
            thumbnail = slide.get_thumbnail((1600, 1600))
            width, height = slide.dimensions
            axis.imshow(thumbnail)
            plot_x = coords[:, 0] * thumbnail.size[0] / float(width)
            plot_y = coords[:, 1] * thumbnail.size[1] / float(height)
            used_overlay = True
        except (ImportError, OSError):
            pass

    limit = max(float(np.max(np.abs(values))), 1e-8)
    scatter = axis.scatter(
        plot_x,
        plot_y,
        c=values,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        s=18,
        alpha=0.78,
        linewidths=0,
    )
    if not used_overlay:
        axis.invert_yaxis()
    axis.set_axis_off()
    figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return used_overlay


def export_case_explanations(
    case_id: str,
    explanations: Mapping[str, torch.Tensor],
    output_dir: str | Path,
    *,
    patch_metadata: Sequence[Mapping[str, object]],
    pathway_names: Sequence[str] | None = None,
    deletion_sweep: Mapping[str, torch.Tensor] | None = None,
    top_pairs: int = 100,
    coordinate_root: str | Path | None = None,
    slide_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Export exact stage-logit decompositions and transport diagnostics."""

    required = {
        "factual_cost",
        "factual_plan",
        "stable_plan",
        "stable_cost",
        "transport_stability_score",
        "transport_reliability",
        "edge_values",
        "stage_edge_contribution",
        "stage_patch_contribution",
        "stage_pathway_contribution",
        "stage_logits",
        "stage_bias",
        "hazards",
        "survival",
        "risk",
        "completeness_error",
        "marginal_error",
        "wsi_valid_mask",
        "omic_valid_mask",
    }
    missing = sorted(required - set(explanations))
    if missing:
        raise KeyError(f"missing IST explanation tensors: {', '.join(missing)}")

    case_dir = Path(output_dir) / str(case_id)
    if case_dir.exists() and any(case_dir.iterdir()) and not force:
        raise FileExistsError(
            f"explanation directory exists; use force to replace it: {case_dir}"
        )
    case_dir.mkdir(parents=True, exist_ok=True)

    plan = _numpy(explanations["stable_plan"])[0]
    cost = _numpy(explanations["stable_cost"])[0]
    stability = _numpy(explanations["transport_stability_score"])[0]
    reliability = _numpy(explanations["transport_reliability"])[0]
    edge_values = _numpy(explanations["edge_values"])[0]
    contributions = _numpy(explanations["stage_edge_contribution"])[0]
    patch_contributions = _numpy(
        explanations["stage_patch_contribution"]
    )[0]
    pathway_contributions = _numpy(
        explanations["stage_pathway_contribution"]
    )[0]
    valid_patches = _numpy(explanations["wsi_valid_mask"])[0].astype(bool)
    valid_pathways = _numpy(explanations["omic_valid_mask"])[0].astype(bool)

    if len(patch_metadata) != plan.shape[0]:
        raise ValueError("patch metadata count does not match WSI token count")
    if pathway_names is None or len(pathway_names) != plan.shape[1]:
        pathway_names = [
            f"pathway_{index}" for index in range(plan.shape[1])
        ]

    pair_rows: list[dict[str, object]] = []
    for stage in range(contributions.shape[0]):
        score = np.abs(contributions[stage])
        valid = valid_patches[:, None] & valid_pathways[None, :]
        flat = np.flatnonzero(valid.reshape(-1))
        ranked = flat[
            np.argsort(-score.reshape(-1)[flat])[
                : min(int(top_pairs), flat.size)
            ]
        ]
        for rank, flat_index in enumerate(ranked, start=1):
            patch_index, pathway_index = np.unravel_index(
                int(flat_index), score.shape
            )
            contribution = float(
                contributions[stage, patch_index, pathway_index]
            )
            pair_rows.append(
                {
                    "case_id": str(case_id),
                    "stage": stage,
                    "rank": rank,
                    "sampled_token": int(patch_index),
                    "pathway_index": int(pathway_index),
                    "pathway": str(pathway_names[pathway_index]),
                    "signed_logit_contribution": contribution,
                    "direction": (
                        "risk_increasing"
                        if contribution > 0
                        else "risk_decreasing"
                        if contribution < 0
                        else "neutral"
                    ),
                    "transport_mass": float(plan[patch_index, pathway_index]),
                    "edge_value": float(
                        edge_values[stage, patch_index, pathway_index]
                    ),
                    "stable_cost": float(cost[patch_index, pathway_index]),
                    "stability_score": float(
                        stability[patch_index, pathway_index]
                    ),
                    "reliability": float(
                        reliability[patch_index, pathway_index]
                    ),
                    **dict(patch_metadata[patch_index]),
                }
            )
    _write_rows(case_dir / "stage_patch_pathway.csv", pair_rows)

    patch_rows: list[dict[str, object]] = []
    for stage in range(patch_contributions.shape[0]):
        for patch_index in np.flatnonzero(valid_patches):
            value = float(patch_contributions[stage, patch_index])
            patch_rows.append(
                {
                    "case_id": str(case_id),
                    "stage": stage,
                    "sampled_token": int(patch_index),
                    "signed_logit_contribution": value,
                    "absolute_contribution": abs(value),
                    **dict(patch_metadata[int(patch_index)]),
                }
            )
    _write_rows(case_dir / "stage_patch_attribution.csv", patch_rows)

    pathway_rows: list[dict[str, object]] = []
    for stage in range(pathway_contributions.shape[0]):
        for pathway_index in np.flatnonzero(valid_pathways):
            value = float(pathway_contributions[stage, pathway_index])
            pathway_rows.append(
                {
                    "case_id": str(case_id),
                    "stage": stage,
                    "pathway_index": int(pathway_index),
                    "pathway": str(pathway_names[pathway_index]),
                    "signed_logit_contribution": value,
                    "absolute_contribution": abs(value),
                }
            )
    _write_rows(case_dir / "stage_pathway_attribution.csv", pathway_rows)

    np.savez_compressed(
        case_dir / "transport_matrices.npz",
        factual_cost=_numpy(explanations.get("factual_cost"))[0],
        factual_plan=_numpy(explanations.get("factual_plan"))[0],
        stable_cost=cost,
        stable_plan=plan,
        transport_stability_score=stability,
        transport_reliability=reliability,
        edge_values=edge_values,
        stage_edge_contribution=contributions,
        stage_patch_contribution=patch_contributions,
        stage_pathway_contribution=pathway_contributions,
        intervention_plans=_numpy(explanations["intervention_plans"])[0],
        intervention_row_masks=_numpy(
            explanations["intervention_row_masks"]
        )[0],
        intervention_col_masks=_numpy(
            explanations["intervention_col_masks"]
        )[0],
    )

    spatial_available = False
    overlay_available = False
    slides = sorted(
        {
            str(item.get("slide_id"))
            for item in patch_metadata
            if item.get("slide_id") not in (None, "")
        }
    )
    for slide_id in slides:
        coordinate_file = find_coordinate_file(
            coordinate_root, Path(slide_id).stem
        )
        if coordinate_file is None:
            continue
        coordinates = load_patch_coordinates(coordinate_file)
        selected = [
            (token_index, int(item["slide_patch_index"]))
            for token_index, item in enumerate(patch_metadata)
            if str(item.get("slide_id")) == slide_id
            and item.get("slide_patch_index") is not None
            and not bool(item.get("padded", False))
            and int(item["slide_patch_index"]) < coordinates.shape[0]
        ]
        if not selected:
            continue
        token_indices = np.asarray([item[0] for item in selected], dtype=int)
        coordinate_indices = np.asarray([item[1] for item in selected], dtype=int)
        slide_path = None
        if slide_root is not None:
            candidate = Path(slide_root) / slide_id
            if candidate.exists():
                slide_path = candidate
        for stage in range(patch_contributions.shape[0]):
            used_overlay = render_signed_patch_heatmap(
                coordinates[coordinate_indices],
                patch_contributions[stage, token_indices],
                case_dir
                / f"{Path(slide_id).stem}_stage{stage}_signed_heatmap.png",
                slide_path=slide_path,
            )
            spatial_available = True
            overlay_available = overlay_available or used_overlay

    summary = {
        "case_id": str(case_id),
        "stage_logits": _json_values(explanations["stage_logits"][0]),
        "stage_bias": _json_values(explanations["stage_bias"]),
        "hazards": _json_values(explanations["hazards"][0]),
        "survival": _json_values(explanations["survival"][0]),
        "risk": float(_numpy(explanations["risk"])[0]),
        "completeness_error": float(
            _numpy(explanations["completeness_error"])[0]
        ),
        "marginal_error": float(_numpy(explanations["marginal_error"])[0]),
        "mean_transport_reliability": float(reliability[valid_patches][:, valid_pathways].mean()),
        "spatial_coordinates_available": spatial_available,
        "wsi_overlay_available": overlay_available,
        "explanation_semantics": (
            "signed contributions exactly sum to each hazard logit minus its bias"
        ),
    }
    if deletion_sweep is not None:
        summary["deletion_sweep"] = {
            key: _json_values(value) for key, value in deletion_sweep.items()
        }
    with (case_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return case_dir
