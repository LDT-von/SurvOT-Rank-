import json
from types import SimpleNamespace

import torch

from survot_rank.research.components.slot_attention import MultiHeadSlotAttention
from survot_rank.research.methods.dct_listwise_transport.explanations import (
    build_patch_metadata,
    compose_patch_pathway_transport,
    export_case_explanations,
)
from survot_rank.research.methods.dct_listwise_transport.model import (
    DCTListwiseTransport,
    censor_aware_plackett_luce_loss,
)
from survot_rank.training.model_factory import list_methods


def make_args(mode="stage_transport"):
    return SimpleNamespace(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=3,
        slot_num_omics=3,
        slot_iters=2,
        otehv2_eps=0.05,
        otehv2_iter=20,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.10,
        dct_ipcw_rank_margin=0.02,
        dct_ipcw_rank_temperature=0.50,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=0,
        dct_lambda_etar=0.10,
        dct_etar_margin=0.02,
        dct_etar_uncertainty_weight=0.05,
        dct_etar_temperature=0.50,
        dct_etar_evidence_floor=0.10,
        dct_lambda_ot=0.0,
        dct_lambda_rank=0.0,
        dct_lambda_anchor=0.0,
        dct_lambda_stage_risk=0.0,
        dct_stage_risk_margin=0.02,
        dct_anchor_margin=0.02,
        dct_anchor_momentum=0.90,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_coupling_projection_iters=30,
        dct_coupling_projection_tol=1e-4,
        dct_lambda_coordinate=0.0,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.50,
        dct_slot_init_mode="deterministic",
        dct_slot_eval_seed=991,
        dct_evidence_marginal_strength=1.0,
        dct_geometry_reliability_strength=0.0,
        dct_geometry_reliability_temperature=0.25,
        dct_listwise_mode=mode,
        dct_lambda_listwise=0.10,
        dct_listwise_temperature=0.50,
        dct_listwise_memory_size=4,
        dct_listwise_tie_method="breslow",
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        cur_epoch=0,
    )


def configure_reference(model):
    model.configure_train_reference(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0, 8.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
    )


def test_plackett_luce_prefers_correct_survival_order():
    times = torch.tensor([1.0, 2.0, 3.0])
    censorship = torch.zeros(3)
    correct, correct_diag = censor_aware_plackett_luce_loss(
        torch.tensor([[3.0], [2.0], [1.0]]),
        times,
        censorship,
        temperature=1.0,
    )
    reversed_loss, _ = censor_aware_plackett_luce_loss(
        torch.tensor([[1.0], [2.0], [3.0]]),
        times,
        censorship,
        temperature=1.0,
    )
    assert correct < reversed_loss
    assert correct_diag["list_count"] == 2


def test_plackett_luce_uses_censored_patients_only_in_risk_sets():
    scores = torch.tensor([[2.0], [0.5], [-1.0]])
    times = torch.tensor([1.0, 2.0, 3.0])
    censorship = torch.tensor([0.0, 1.0, 1.0])
    loss, diagnostics = censor_aware_plackett_luce_loss(
        scores, times, censorship, temperature=1.0
    )
    expected = torch.logsumexp(scores[:, 0], dim=0) - scores[0, 0]
    assert torch.allclose(loss, expected)
    assert diagnostics["list_count"] == 1
    assert diagnostics["avg_risk_set_size"] == 3


def test_plackett_luce_breslow_ties_and_all_censored_batch_are_finite():
    tied_scores = torch.tensor([[1.0], [0.5], [-1.0]], requires_grad=True)
    tied_loss, diagnostics = censor_aware_plackett_luce_loss(
        tied_scores,
        torch.tensor([1.0, 1.0, 2.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert torch.isfinite(tied_loss)
    assert diagnostics["list_count"] == 2
    tied_loss.backward()
    assert torch.isfinite(tied_scores.grad).all()

    censored_scores = torch.randn(3, 1, requires_grad=True)
    zero, diagnostics = censor_aware_plackett_luce_loss(
        censored_scores,
        torch.tensor([1.0, 2.0, 3.0]),
        torch.ones(3),
    )
    assert zero == 0
    assert diagnostics["list_count"] == 0
    zero.backward()
    assert torch.equal(censored_scores.grad, torch.zeros_like(censored_scores))


def test_slot_attention_capture_is_opt_in_and_numerically_invariant():
    torch.manual_seed(3)
    module = MultiHeadSlotAttention(
        num_slots=3,
        dim=8,
        heads=2,
        dim_head=4,
        iters=2,
        init_mode="deterministic",
        eval_seed=17,
    )
    module.eval()
    inputs = torch.randn(2, 7, 8)
    baseline = module(inputs)
    assert module.last_token_assignment is None
    module.capture_attention = True
    captured = module(inputs)
    assert torch.equal(baseline, captured)
    assert module.last_token_assignment.shape == (2, 3, 7)
    assert module.last_pooling_attention.shape == (2, 3, 7)
    assert torch.allclose(
        module.last_token_assignment.sum(dim=1),
        torch.ones(2, 7),
        atol=1e-6,
    )
    assert torch.allclose(
        module.last_pooling_attention.sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
    )


def test_tcl_forward_backward_and_epoch_memory_reset():
    torch.manual_seed(5)
    model = DCTListwiseTransport(make_args(), omic_input_dim=20)
    configure_reference(model)
    model.train()
    logits, aux_loss = model(
        x_wsi=torch.randn(4, 6, 16),
        x_omics=torch.randn(4, 5, 20),
        event_time=torch.tensor([1.0, 2.0, 5.0, 7.0]),
        c=torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    assert logits.shape == (4, 4)
    assert torch.isfinite(aux_loss)
    assert model.last_training_losses["listwise_lists"] > 0
    assert model.last_training_losses["listwise_avg_risk_set"] >= 2
    assert model.last_training_losses["listwise_finite_scores"] == 1
    assert model.dct_lambda_ipcw_rank == 0
    assert model.dct_lambda_etar == 0
    aux_loss.backward()
    assert model.stage_listwise_head[-1].weight.grad is not None
    assert torch.isfinite(model.stage_listwise_head[-1].weight.grad).all()
    assert model._listwise_memory_scores.requires_grad is False

    model.args.cur_epoch = 1
    model(
        x_wsi=torch.randn(2, 6, 16),
        x_omics=torch.randn(2, 5, 20),
        event_time=torch.tensor([1.0, 6.0]),
        c=torch.tensor([0.0, 1.0]),
    )
    assert model._listwise_memory_epoch == 1
    assert model._listwise_memory_scores.size(0) == 2


def test_gpl_uses_final_factual_risk_without_stage_head_gradient():
    model = DCTListwiseTransport(make_args(mode="global"), omic_input_dim=20)
    configure_reference(model)
    model.train()
    _, aux_loss = model(
        x_wsi=torch.randn(4, 6, 16),
        x_omics=torch.randn(4, 5, 20),
        event_time=torch.tensor([1.0, 2.0, 5.0, 7.0]),
        c=torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    aux_loss.backward()
    assert torch.isfinite(aux_loss)
    assert model.stage_listwise_head[-1].weight.grad is None
    assert model.last_training_losses["listwise_stage_coverage"] == 1


def _eval_model(batch_size=2):
    torch.manual_seed(9)
    model = DCTListwiseTransport(make_args(), omic_input_dim=20)
    configure_reference(model)
    model.train()
    model(
        x_wsi=torch.randn(4, 6, 16),
        x_omics=torch.randn(4, 5, 20),
        event_time=torch.tensor([1.0, 2.0, 5.0, 7.0]),
        c=torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    model.eval()
    with torch.no_grad():
        model(
            x_wsi=torch.randn(batch_size, 6, 16),
            x_omics=torch.randn(batch_size, 5, 20),
        )
    return model


def test_eval_explanations_reach_patch_pathway_and_preserve_transport_mass():
    model = _eval_model(batch_size=2)
    explanation = model.explain_last_batch()
    assert explanation["wsi_patch_to_global_prototype"].shape == (2, 3, 6)
    assert explanation["omic_pathway_to_global_prototype"].shape == (2, 3, 5)
    assert explanation["factual_stage_couplings"].shape == (2, 4, 3, 3, 3)
    contribution = compose_patch_pathway_transport(
        explanation["wsi_patch_to_global_pooling"],
        explanation["factual_stage_couplings"],
        explanation["omic_pathway_to_global_pooling"],
    )
    assert contribution.shape == (2, 4, 6, 5)
    assert torch.all(contribution >= 0)
    assert torch.allclose(
        contribution.sum(dim=(2, 3)),
        torch.ones(2, 4),
        atol=2e-3,
    )
    for key in (
        "factual_coupling_marginal_error",
        "low_coupling_marginal_error",
        "high_coupling_marginal_error",
    ):
        assert torch.all(explanation[key] < 2e-3)

    sweep = model.counterfactual_sweep()
    assert sweep["low_risk"].shape == (2, 5)
    assert sweep["high_risk"].shape == (2, 5)
    assert sweep["random_anchor_risk"].shape == (2,)
    assert torch.equal(sweep["frozen_coupling_risk"], sweep["factual_risk"])


def test_case_export_writes_compact_auditable_artifacts(tmp_path):
    model = _eval_model(batch_size=1)
    explanation = model.explain_last_batch()
    metadata = build_patch_metadata(
        ["slide-a.svs", "slide-b.svs"],
        [3, 3],
        range(6),
    )
    case_dir = export_case_explanations(
        "case-1",
        explanation,
        tmp_path,
        patch_metadata=metadata,
        pathway_names=[f"pathway-{idx}" for idx in range(5)],
        sweep=model.counterfactual_sweep(),
        top_patches=2,
        top_pathways=3,
        top_pairs=4,
    )
    assert (case_dir / "summary.json").exists()
    assert (case_dir / "prototype_patch.csv").exists()
    assert (case_dir / "stage_patch_pathway.csv").exists()
    assert (case_dir / "transport_matrices.npz").exists()
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["spatial_coordinates_available"] is False
    assert summary["wsi_overlay_available"] is False
    assert len(summary["counterfactual_sweep"]["alpha"]) == 5


def test_new_method_is_registered_without_replacing_old_dct():
    methods = list_methods()
    assert "distributional_counterfactual_transport" in methods
    assert "dct_listwise_transport" in methods
