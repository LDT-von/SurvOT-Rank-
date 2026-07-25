import json
from types import SimpleNamespace

import numpy as np
import torch

from scripts.run_v40_intervention_stable_transport import (
    CANCERS,
    PROTOCOLS,
    VARIANTS,
    build_parser,
    build_train_command,
)
from scripts.run_recent_transport_5fold import (
    build_commands as build_recent_commands,
    build_parser as build_recent_parser,
)
from survot_rank.research.methods.dct_listwise_transport.explanations import (
    build_patch_metadata,
)
from survot_rank.research.methods.intervention_stable_survival_transport.explanations import (
    export_case_explanations,
)
from survot_rank.research.methods.intervention_stable_survival_transport.model import (
    InterventionStableSurvivalTransport,
    masked_log_sinkhorn_plan,
)
from survot_rank.training.model_factory import get_model, list_methods


def make_args(**overrides):
    values = {
        "omic_sizes": [4, 5, 3],
        "n_classes": 4,
        "encoding_dim": 10,
        "wsi_projection_dim": 8,
        "rna_format": "Pathways",
        "ist_eps": 0.08,
        "ist_sinkhorn_iters": 30,
        "ist_num_interventions": 3,
        "ist_keep_ratio": 0.70,
        "ist_stability_beta": 1.0,
        "ist_stability_strength": 0.10,
        "ist_lambda_plan": 0.05,
        "ist_lambda_attribution": 0.05,
        "ist_lambda_risk": 0.02,
        "ist_edge_value_scale": 4.0,
        "ist_eval_seed": 17,
        "ist_deletion_penalty": 8.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_batch(seed=3, batch_size=3):
    generator = torch.Generator().manual_seed(seed)
    x_wsi = torch.randn(batch_size, 7, 10, generator=generator)
    x_wsi[:, -1] = 0.0
    return {
        "x_wsi": x_wsi,
        "x_omic1": torch.randn(batch_size, 4, generator=generator),
        "x_omic2": torch.randn(batch_size, 5, generator=generator),
        "x_omic3": torch.randn(batch_size, 3, generator=generator),
    }


def test_masked_sinkhorn_respects_visible_marginals():
    torch.manual_seed(1)
    cost = torch.rand(2, 5, 4)
    row_mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, True]]
    )
    col_mask = torch.tensor(
        [[True, False, True, True], [False, True, True, False]]
    )
    plan, rows, cols = masked_log_sinkhorn_plan(
        cost, row_mask, col_mask, eps=0.10, max_iter=80
    )
    assert torch.isfinite(plan).all()
    assert torch.equal(
        plan.masked_select(
            ~(row_mask.unsqueeze(2) & col_mask.unsqueeze(1))
        ),
        torch.zeros_like(
            plan.masked_select(
                ~(row_mask.unsqueeze(2) & col_mask.unsqueeze(1))
            )
        ),
    )
    assert torch.allclose(plan.sum(dim=2), rows, atol=2e-4)
    assert torch.allclose(plan.sum(dim=1), cols, atol=2e-4)


def test_v40_forward_is_exactly_additive_and_has_finite_gradients():
    torch.manual_seed(5)
    model = InterventionStableSurvivalTransport(make_args())
    assert not hasattr(model, "slot_attention_wsi")
    model.train()
    logits, aux_loss = model(**make_batch())
    explanations = model.explain_last_batch()

    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert explanations["stage_edge_contribution"].shape == (3, 4, 7, 3)
    reconstructed = (
        model.stage_bias
        + explanations["stage_edge_contribution"].sum(dim=(2, 3))
    )
    assert torch.allclose(logits, reconstructed, atol=1e-6)
    assert explanations["completeness_error"].max() <= 1e-6
    assert explanations["marginal_error"].max() < 1e-3
    assert torch.equal(
        explanations["stage_patch_contribution"],
        explanations["stage_edge_contribution"].sum(dim=3),
    )
    assert torch.equal(
        explanations["stage_pathway_contribution"],
        explanations["stage_edge_contribution"].sum(dim=2),
    )

    objective = logits.square().mean() + aux_loss
    objective.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.wsi_stage_value[-1].weight.grad is not None
    assert model.omic_stage_value[-1].weight.grad is not None
    assert model.edge_pair_scale.grad is not None


def test_v40_eval_is_deterministic_and_deletion_reoptimises_transport():
    torch.manual_seed(9)
    model = InterventionStableSurvivalTransport(make_args()).eval()
    values = make_batch(seed=11, batch_size=2)
    with torch.no_grad():
        first_logits, _ = model(**values)
        first_plans = model.explain_last_batch()["intervention_plans"].clone()
        second_logits, _ = model(**values)
        second_plans = model.explain_last_batch()["intervention_plans"].clone()
    assert torch.equal(first_logits, second_logits)
    assert torch.equal(first_plans, second_plans)

    sweep = model.deletion_sweep(fractions=(0.10, 0.20), seed=23)
    assert sweep["top_deleted_logits"].shape == (2, 2, 4)
    assert sweep["random_deleted_logits"].shape == (2, 2, 4)
    assert torch.isfinite(sweep["top_deleted_risk"]).all()
    assert torch.isfinite(sweep["random_deleted_risk"]).all()
    assert (sweep["top_plan_shift"] > 0).all()
    assert (sweep["random_plan_shift"] > 0).all()
    assert sweep["positive_deleted_target_logits"].shape == (2, 2, 4)
    assert sweep["negative_deleted_target_logits"].shape == (2, 2, 4)
    assert sweep["positive_direction_ok"].dtype == torch.bool
    assert sweep["negative_direction_ok"].dtype == torch.bool


def test_v40_export_writes_signed_complete_case_artifacts(tmp_path):
    torch.manual_seed(13)
    model = InterventionStableSurvivalTransport(make_args()).eval()
    with torch.no_grad():
        model(**make_batch(batch_size=1))
    explanations = model.explain_last_batch()
    metadata = build_patch_metadata(
        ["slide-a.svs"], [6], [0, 1, 2, 3, 4, 5, 6]
    )
    case_dir = export_case_explanations(
        "case-a",
        explanations,
        tmp_path,
        patch_metadata=metadata,
        pathway_names=["p1", "p2", "p3"],
        deletion_sweep=model.deletion_sweep(fractions=(0.10,)),
        top_pairs=5,
    )
    assert (case_dir / "summary.json").exists()
    assert (case_dir / "stage_patch_pathway.csv").exists()
    assert (case_dir / "stage_patch_attribution.csv").exists()
    assert (case_dir / "stage_pathway_attribution.csv").exists()
    assert (case_dir / "transport_matrices.npz").exists()
    summary = json.loads(
        (case_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["completeness_error"] <= 1e-6
    assert summary["spatial_coordinates_available"] is False
    pair_text = (case_dir / "stage_patch_pathway.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "signed_logit_contribution" in pair_text
    archive = np.load(case_dir / "transport_matrices.npz")
    assert archive["stage_edge_contribution"].shape == (4, 7, 3)


def test_v40_registry_and_five_fold_launcher_are_isolated():
    assert "intervention_stable_survival_transport" in list_methods()
    assert get_model("v40", make_args()).__class__.__name__ == (
        "InterventionStableSurvivalTransport"
    )
    assert get_model("ist_surv", make_args()).__class__.__name__ == (
        "InterventionStableSurvivalTransport"
    )
    defaults = build_parser().parse_args([])
    assert defaults.mode == "plan"
    assert defaults.cancers == list(CANCERS)
    assert defaults.folds == [0, 1, 2, 3, 4]
    assert defaults.protocols == ["highscore", "clean"]
    assert defaults.variants == ["full"]
    assert set(PROTOCOLS) == {"highscore", "clean"}
    assert set(VARIANTS) == {"factual", "stable_plan", "full"}

    command, result_dir = build_train_command(
        "python3",
        "blca",
        "clean",
        "full",
        4,
        "0",
        "4",
        "/data1/TCGA-UNI2-h-features",
    )
    rendered = " ".join(command)
    assert "survot_method=intervention_stable_survival_transport" in rendered
    assert "fit_bins_on_train=true" in rendered
    assert "ist_lambda_attribution=0.05" in rendered
    assert "k_start=4" in rendered
    assert result_dir.as_posix() == "results/ist_surv_v4.0/clean/full/blca"

    recent_args = build_recent_parser().parse_args([])
    recent = dict(build_recent_commands(recent_args))
    assert set(recent) == {
        "v3.6-TCL",
        "v3.7-UNI2H",
        "v3.8-transport-consistency",
        "v4.0-IST-Surv",
    }
    assert "0,1,2,3,4" in recent["v4.0-IST-Surv"]
    assert "blca,brca" in recent["v4.0-IST-Surv"]
