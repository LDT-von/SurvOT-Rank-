"""Case-level explanation composition and export for DCT v3.6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


def compose_patch_pathway_transport(
    wsi_patch_pooling: torch.Tensor,
    stage_couplings: torch.Tensor,
    omic_pathway_pooling: torch.Tensor,
) -> torch.Tensor:
    """Compose patch→prototype, prototype OT, and prototype→pathway weights.

    Args:
        wsi_patch_pooling: ``[B, K_w, N_patch]``.
        stage_couplings: ``[B, stages, geometries, K_w, K_o]``.
        omic_pathway_pooling: ``[B, K_o, N_pathway]``.

    Returns:
        Geometry-averaged mass ``[B, stages, N_patch, N_pathway]``.
    """

    if wsi_patch_pooling.ndim != 3 or omic_pathway_pooling.ndim != 3:
        raise ValueError("patch/pathway pooling tensors must be three-dimensional")
    if stage_couplings.ndim != 5:
        raise ValueError("stage_couplings must have shape [B,S,G,Kw,Ko]")
    if (
        wsi_patch_pooling.size(0) != stage_couplings.size(0)
        or omic_pathway_pooling.size(0) != stage_couplings.size(0)
        or wsi_patch_pooling.size(1) != stage_couplings.size(3)
        or omic_pathway_pooling.size(1) != stage_couplings.size(4)
    ):
        raise ValueError("pooling and coupling dimensions are incompatible")

    mean_coupling = stage_couplings.mean(dim=2)
    contribution = torch.einsum(
        "bkn,bskl,blp->bsnp",
        wsi_patch_pooling,
        mean_coupling,
        omic_pathway_pooling,
    )
    return contribution.clamp_min(0.0)


def _to_numpy(value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value) -> float:
    array = _to_numpy(value).reshape(-1)
    return float(array[0]) if array.size else float("nan")


def load_patch_coordinates(path: str | Path) -> np.ndarray:
    """Load CLAM-style coordinates from optional common formats."""

    coord_path = Path(path)
    suffix = coord_path.suffix.lower()
    if suffix == ".npy":
        coordinates = np.load(coord_path)
    elif suffix == ".npz":
        archive = np.load(coord_path)
        key = "coords" if "coords" in archive else archive.files[0]
        coordinates = archive[key]
    elif suffix == ".csv":
        frame = pd.read_csv(coord_path)
        if {"x", "y"}.issubset(frame.columns):
            coordinates = frame[["x", "y"]].to_numpy()
        else:
            coordinates = frame.iloc[:, :2].to_numpy()
    elif suffix in {".pt", ".pth"}:
        loaded = torch.load(coord_path, map_location="cpu")
        if isinstance(loaded, Mapping):
            loaded = loaded.get("coords", next(iter(loaded.values())))
        coordinates = _to_numpy(loaded)
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError(
                "h5py is required to read .h5 coordinate files"
            ) from error
        with h5py.File(coord_path, "r") as handle:
            key = "coords" if "coords" in handle else next(iter(handle.keys()))
            coordinates = handle[key][:]
    else:
        raise ValueError(f"unsupported coordinate format: {coord_path}")

    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"coordinates must have shape [patches, >=2]: {coord_path}")
    return coordinates[:, :2].astype(float, copy=False)


def find_coordinate_file(
    coordinate_root: str | Path | None,
    slide_stem: str,
) -> Path | None:
    if coordinate_root is None:
        return None
    root = Path(coordinate_root)
    for suffix in (".h5", ".hdf5", ".npy", ".npz", ".csv", ".pt", ".pth"):
        candidate = root / f"{slide_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def render_patch_heatmap(
    coordinates: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
    *,
    slide_path: str | Path | None = None,
) -> bool:
    """Render a coordinate scatter, or a true WSI overlay when OpenSlide exists."""

    import matplotlib.pyplot as plt

    coords = np.asarray(coordinates, dtype=float)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if coords.shape[0] != values.size:
        raise ValueError("coordinate and score counts do not match")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8))
    used_slide = False
    if slide_path is not None and Path(slide_path).exists():
        try:
            import openslide

            slide = openslide.OpenSlide(str(slide_path))
            thumbnail = slide.get_thumbnail((1600, 1600))
            width, height = slide.dimensions
            axis.imshow(thumbnail)
            x_scale = thumbnail.size[0] / float(width)
            y_scale = thumbnail.size[1] / float(height)
            plot_x = coords[:, 0] * x_scale
            plot_y = coords[:, 1] * y_scale
            used_slide = True
        except (ImportError, OSError):
            plot_x, plot_y = coords[:, 0], coords[:, 1]
    else:
        plot_x, plot_y = coords[:, 0], coords[:, 1]

    scatter = axis.scatter(
        plot_x,
        plot_y,
        c=values,
        cmap="magma",
        s=18,
        alpha=0.75,
        linewidths=0,
    )
    axis.invert_yaxis() if not used_slide else None
    axis.set_axis_off()
    figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return used_slide


def export_case_explanations(
    case_id: str,
    explanations: Mapping[str, torch.Tensor],
    output_dir: str | Path,
    *,
    patch_metadata: Sequence[Mapping[str, object]] | None = None,
    pathway_names: Sequence[str] | None = None,
    sweep: Mapping[str, torch.Tensor] | None = None,
    top_patches: int = 50,
    top_pathways: int = 20,
    top_pairs: int = 100,
    coordinate_root: str | Path | None = None,
    slide_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Write compact, auditable explanation artifacts for one evaluation case."""

    required = (
        "wsi_patch_to_global_prototype",
        "wsi_patch_to_global_pooling",
        "omic_pathway_to_global_prototype",
        "omic_pathway_to_global_pooling",
        "factual_stage_costs",
        "low_stage_costs",
        "high_stage_costs",
        "factual_stage_couplings",
        "low_stage_couplings",
        "high_stage_couplings",
    )
    missing = [key for key in required if key not in explanations]
    if missing:
        raise KeyError(f"missing DCT explanation tensors: {', '.join(missing)}")

    case_dir = Path(output_dir) / str(case_id)
    if case_dir.exists() and any(case_dir.iterdir()) and not force:
        raise FileExistsError(
            f"explanation directory already exists; pass force to replace files: {case_dir}"
        )
    case_dir.mkdir(parents=True, exist_ok=True)

    wsi_membership = explanations["wsi_patch_to_global_prototype"][:1]
    wsi_pooling = explanations["wsi_patch_to_global_pooling"][:1]
    omic_membership = explanations["omic_pathway_to_global_prototype"][:1]
    omic_pooling = explanations["omic_pathway_to_global_pooling"][:1]
    factual_coupling = explanations["factual_stage_couplings"][:1]
    low_coupling = explanations["low_stage_couplings"][:1]
    high_coupling = explanations["high_stage_couplings"][:1]

    factual_contribution = compose_patch_pathway_transport(
        wsi_pooling, factual_coupling, omic_pooling
    )[0]
    low_contribution = compose_patch_pathway_transport(
        wsi_pooling, low_coupling, omic_pooling
    )[0]
    high_contribution = compose_patch_pathway_transport(
        wsi_pooling, high_coupling, omic_pooling
    )[0]

    patch_count = wsi_pooling.size(-1)
    pathway_count = omic_pooling.size(-1)
    if patch_metadata is None:
        patch_metadata = [
            {"sampled_patch_index": index} for index in range(patch_count)
        ]
    if len(patch_metadata) != patch_count:
        raise ValueError("patch metadata count does not match captured patch tokens")
    if pathway_names is None:
        pathway_names = [f"pathway_{index}" for index in range(pathway_count)]
    if len(pathway_names) != pathway_count:
        raise ValueError("pathway name count does not match captured omics tokens")

    prototype_rows = []
    wsi_pool_np = _to_numpy(wsi_pooling[0])
    wsi_member_np = _to_numpy(wsi_membership[0])
    for prototype_idx in range(wsi_pool_np.shape[0]):
        selected = np.argsort(-wsi_pool_np[prototype_idx])[
            : min(top_patches, patch_count)
        ]
        prototype_mass = float(wsi_member_np[prototype_idx].mean())
        for rank, patch_idx in enumerate(selected, start=1):
            metadata = dict(patch_metadata[int(patch_idx)])
            prototype_rows.append(
                {
                    "case_id": case_id,
                    "prototype": prototype_idx,
                    "prototype_patient_mass": prototype_mass,
                    "rank": rank,
                    "sampled_token": int(patch_idx),
                    "pooling_weight": float(wsi_pool_np[prototype_idx, patch_idx]),
                    "membership_probability": float(
                        wsi_member_np[prototype_idx, patch_idx]
                    ),
                    **metadata,
                }
            )
    pd.DataFrame(prototype_rows).to_csv(
        case_dir / "prototype_patch.csv", index=False
    )

    pair_rows = []
    factual_np = _to_numpy(factual_contribution)
    low_np = _to_numpy(low_contribution)
    high_np = _to_numpy(high_contribution)
    for stage_idx in range(factual_np.shape[0]):
        pathway_mass = factual_np[stage_idx].sum(axis=0)
        allowed_pathways = np.argsort(-pathway_mass)[
            : min(top_pathways, pathway_count)
        ]
        masked = np.full_like(factual_np[stage_idx], -np.inf)
        masked[:, allowed_pathways] = factual_np[stage_idx][
            :, allowed_pathways
        ]
        selected_pairs = np.argsort(-masked.reshape(-1))[
            : min(top_pairs, patch_count * len(allowed_pathways))
        ]
        for rank, flat_idx in enumerate(selected_pairs, start=1):
            patch_idx, pathway_idx = np.unravel_index(
                int(flat_idx), masked.shape
            )
            metadata = dict(patch_metadata[int(patch_idx)])
            factual_value = float(factual_np[stage_idx, patch_idx, pathway_idx])
            low_value = float(low_np[stage_idx, patch_idx, pathway_idx])
            high_value = float(high_np[stage_idx, patch_idx, pathway_idx])
            pair_rows.append(
                {
                    "case_id": case_id,
                    "stage": stage_idx,
                    "rank": rank,
                    "sampled_token": int(patch_idx),
                    "pathway_index": int(pathway_idx),
                    "pathway": str(pathway_names[pathway_idx]),
                    "factual_transport_mass": factual_value,
                    "low_transport_mass": low_value,
                    "high_transport_mass": high_value,
                    "low_minus_factual": low_value - factual_value,
                    "high_minus_factual": high_value - factual_value,
                    **metadata,
                }
            )
    pd.DataFrame(pair_rows).to_csv(
        case_dir / "stage_patch_pathway.csv", index=False
    )

    np.savez_compressed(
        case_dir / "transport_matrices.npz",
        factual_stage_costs=_to_numpy(explanations["factual_stage_costs"][0]),
        low_stage_costs=_to_numpy(explanations["low_stage_costs"][0]),
        high_stage_costs=_to_numpy(explanations["high_stage_costs"][0]),
        factual_stage_couplings=_to_numpy(factual_coupling[0]),
        low_stage_couplings=_to_numpy(low_coupling[0]),
        high_stage_couplings=_to_numpy(high_coupling[0]),
        factual_row_marginals=_to_numpy(
            explanations["factual_row_marginals"][0]
        ),
        factual_col_marginals=_to_numpy(
            explanations["factual_col_marginals"][0]
        ),
        wsi_patch_to_global_prototype=_to_numpy(wsi_membership[0]),
        wsi_patch_to_global_pooling=_to_numpy(wsi_pooling[0]),
        omic_pathway_to_global_prototype=_to_numpy(omic_membership[0]),
        omic_pathway_to_global_pooling=_to_numpy(omic_pooling[0]),
    )

    summary = {
        "case_id": str(case_id),
        "factual_risk": _scalar(explanations["factual_risk"]),
        "low_risk": _scalar(explanations["low_risk_counterfactual"]),
        "high_risk": _scalar(explanations["high_risk_counterfactual"]),
        "low_risk_delta": _scalar(explanations["counterfactual_risk_delta_low"]),
        "high_risk_delta": _scalar(explanations["counterfactual_risk_delta_high"]),
        "low_transport_distance": _scalar(
            explanations["counterfactual_transport_distance_low"]
        ),
        "high_transport_distance": _scalar(
            explanations["counterfactual_transport_distance_high"]
        ),
        "factual_marginal_error": _scalar(
            explanations["factual_coupling_marginal_error"]
        ),
        "low_marginal_error": _scalar(
            explanations["low_coupling_marginal_error"]
        ),
        "high_marginal_error": _scalar(
            explanations["high_coupling_marginal_error"]
        ),
        "spatial_coordinates_available": False,
        "wsi_overlay_available": False,
    }
    if sweep is not None:
        summary["counterfactual_sweep"] = {
            key: _to_numpy(value).reshape(-1).tolist()
            for key, value in sweep.items()
        }

    patch_scores = factual_np.sum(axis=(0, 2))
    slides = sorted(
        {
            str(item.get("slide_id"))
            for item in patch_metadata
            if item.get("slide_id") not in (None, "")
        }
    )
    rendered_any = False
    overlay_any = False
    for slide_id in slides:
        slide_stem = Path(slide_id).stem
        coord_file = find_coordinate_file(coordinate_root, slide_stem)
        if coord_file is None:
            continue
        coordinates = load_patch_coordinates(coord_file)
        selected_rows = [
            (idx, item)
            for idx, item in enumerate(patch_metadata)
            if str(item.get("slide_id")) == slide_id
            and item.get("slide_patch_index") is not None
        ]
        valid = [
            (idx, int(item["slide_patch_index"]))
            for idx, item in selected_rows
            if int(item["slide_patch_index"]) < coordinates.shape[0]
        ]
        if not valid:
            continue
        token_indices = np.asarray([item[0] for item in valid], dtype=int)
        coord_indices = np.asarray([item[1] for item in valid], dtype=int)
        slide_path = None
        if slide_root is not None:
            candidate = Path(slide_root) / slide_id
            if candidate.exists():
                slide_path = candidate
        used_overlay = render_patch_heatmap(
            coordinates[coord_indices],
            patch_scores[token_indices],
            case_dir / f"{slide_stem}_transport_heatmap.png",
            slide_path=slide_path,
        )
        rendered_any = True
        overlay_any = overlay_any or used_overlay

    summary["spatial_coordinates_available"] = rendered_any
    summary["wsi_overlay_available"] = overlay_any
    with open(case_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return case_dir


def build_patch_metadata(
    slide_ids: Sequence[str],
    slide_lengths: Sequence[int],
    sampled_indices: Iterable[int],
) -> list[dict[str, object]]:
    """Map concatenated feature indices back to slide-local patch indices."""

    if len(slide_ids) != len(slide_lengths):
        raise ValueError("slide_ids and slide_lengths must have equal length")
    boundaries = np.cumsum([0, *[int(length) for length in slide_lengths]])
    metadata = []
    for sampled_token, global_idx_value in enumerate(sampled_indices):
        global_idx = int(global_idx_value)
        slide_idx = int(np.searchsorted(boundaries[1:], global_idx, side="right"))
        if slide_idx >= len(slide_ids):
            metadata.append(
                {
                    "sampled_patch_index": global_idx,
                    "slide_id": None,
                    "slide_patch_index": None,
                    "padded": True,
                }
            )
            continue
        metadata.append(
            {
                "sampled_patch_index": global_idx,
                "slide_id": str(slide_ids[slide_idx]),
                "slide_patch_index": global_idx - int(boundaries[slide_idx]),
                "padded": False,
            }
        )
    return metadata
