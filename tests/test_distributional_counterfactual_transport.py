from types import SimpleNamespace

import torch

from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)


def make_args():
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
        otehv2_iter=100,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.1,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.10,
        dct_ipcw_rank_margin=0.02,
        dct_ipcw_rank_temperature=0.50,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=0,
        dct_lambda_etar=0.0,
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
        dct_coupling_projection_iters=1000,
        dct_coupling_projection_tol=1e-4,
        dct_lambda_coordinate=0.0,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.50,
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        cur_epoch=1,
    )


def test_train_fold_reference_uses_late_censoring_as_low_risk_set_context():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(
        torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    low, high = model._stage_membership_weights(
        torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert torch.all(high[:4].sum(dim=1) > 0)
    assert low[4].sum() > 0
    assert low[4, -1] > 0
    assert model._ipcw(torch.tensor([1.0])).item() == 1.0


def test_distributional_counterfactual_transport_uses_feasible_risk_anchored_paths():
    torch.manual_seed(0)
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(
        torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    model.train()
    logits, aux_loss = model(
        x_wsi=torch.randn(5, 6, 16),
        x_omics=torch.randn(5, 5, 20),
        event_time=torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        c=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert logits.shape == (5, 4)
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    assert model.last_training_losses["ipcw_pairs"] > 0
    assert torch.allclose(
        aux_loss,
        model.dct_lambda_ipcw_rank * model.last_training_losses["ipcw_rank"],
    )
    aux_loss.backward()
    assert model.stage_pair_cost[-1].weight.grad is not None
    assert model.slot_attention_wsi.slots_mu.grad is not None
    assert model.slot_attention_omic.slots_mu.grad is not None
    assert model.risk_anchor_seen.all()
    assert not hasattr(model, "risk_prototypes")

    # A second batch keeps the score-aligned IPCW ranking path differentiable;
    # it never imposes a requested ordering on synthetic CF risk predictions.
    model.zero_grad(set_to_none=True)
    _, aux_loss = model(
        x_wsi=torch.randn(5, 6, 16),
        x_omics=torch.randn(5, 5, 20),
        event_time=torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        c=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    aux_loss.backward()
    assert model.stage_pair_cost[-1].weight.grad is not None

    model.eval()
    with torch.no_grad():
        logits, _ = model(
            x_wsi=torch.randn(5, 6, 16),
            x_omics=torch.randn(5, 5, 20),
        )
    explanation = model.explain_last_batch()
    assert logits.shape == (5, 4)
    assert explanation["low_risk_counterfactual"].shape == (5,)
    assert explanation["high_risk_counterfactual"].shape == (5,)
    assert explanation["wsi_coordinate_assignment"].shape == (5, 3, 3)
    assert explanation["omic_coordinate_assignment"].shape == (5, 3, 3)
    for key in (
        "factual_coupling_marginal_error",
        "low_coupling_marginal_error",
        "high_coupling_marginal_error",
    ):
        assert torch.all(explanation[key] < 1e-3), key


def test_deterministic_slot_mode_repeats_exact_evaluation_logits():
    torch.manual_seed(19)
    args = make_args()
    args.dct_slot_init_mode = "deterministic"
    args.dct_slot_eval_seed = 991
    args.dct_coupling_projection_iters = 20
    model = DistributionalCounterfactualTransport(args, omic_input_dim=20)
    x_wsi = torch.randn(3, 6, 16)
    x_omics = torch.randn(3, 5, 20)

    model.eval()
    with torch.no_grad():
        first, _ = model(x_wsi=x_wsi, x_omics=x_omics)
        second, _ = model(x_wsi=x_wsi, x_omics=x_omics)
    assert torch.equal(first, second)

    model.train()
    train_first, _ = model(x_wsi=x_wsi, x_omics=x_omics)
    train_second, _ = model(x_wsi=x_wsi, x_omics=x_omics)
    assert not torch.equal(train_first, train_second)


def test_evidence_marginal_strength_zero_restores_uniform_transport_mass():
    args = make_args()
    args.dct_evidence_marginal_strength = 0.0
    model = DistributionalCounterfactualTransport(args, omic_input_dim=20)
    slots_wsi = torch.randn(2, 3, 16)
    slots_omic = torch.randn(2, 3, 16)
    _, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)

    assert torch.allclose(rows, torch.full_like(rows, 1.0 / 3.0))
    assert torch.allclose(cols, torch.full_like(cols, 1.0 / 3.0))


def test_geometry_reliability_is_higher_when_transport_geometries_agree():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    shared = torch.tensor([[[[0.0, 3.0], [3.0, 3.0]]]])
    agreed = shared.expand(1, 3, 2, 2).clone()
    conflicted = torch.tensor(
        [[
            [[0.0, 3.0], [3.0, 3.0]],
            [[3.0, 0.0], [3.0, 3.0]],
            [[3.0, 3.0], [0.0, 3.0]],
        ]]
    )

    agreed_reliability = model._geometry_reliability(agreed)
    conflicted_reliability = model._geometry_reliability(conflicted)

    assert torch.allclose(agreed_reliability, torch.ones_like(agreed_reliability))
    assert torch.all(conflicted_reliability < agreed_reliability)
    assert torch.all((conflicted_reliability >= 0.0) & (conflicted_reliability <= 1.0))


def test_rtem_is_opt_in_and_records_stage_reliability():
    torch.manual_seed(29)
    baseline_args = make_args()
    baseline = DistributionalCounterfactualTransport(baseline_args, omic_input_dim=20)
    rtem_args = make_args()
    rtem_args.dct_geometry_reliability_strength = 1.0
    rtem = DistributionalCounterfactualTransport(rtem_args, omic_input_dim=20)
    rtem.load_state_dict(baseline.state_dict())
    slots_wsi = torch.randn(2, 3, 16)
    slots_omic = torch.randn(2, 3, 16)

    baseline_costs, baseline_rows, baseline_cols, _ = baseline._cost_tensor(
        slots_wsi, slots_omic
    )
    rtem_costs, rtem_rows, rtem_cols, _ = rtem._cost_tensor(slots_wsi, slots_omic)

    assert torch.equal(baseline_costs, rtem_costs)
    assert baseline._last_transport_reliability is None
    assert rtem._last_transport_reliability.shape == (2, 4)
    assert torch.all(torch.isfinite(rtem._last_transport_reliability))
    uniform_rows = torch.full_like(rtem_rows, 1.0 / rtem_rows.size(-1))
    uniform_cols = torch.full_like(rtem_cols, 1.0 / rtem_cols.size(-1))
    assert torch.all(
        (rtem_rows - uniform_rows).abs().sum(dim=-1)
        <= (baseline_rows - uniform_rows).abs().sum(dim=-1) + 1e-7
    )
    assert torch.all(
        (rtem_cols - uniform_cols).abs().sum(dim=-1)
        <= (baseline_cols - uniform_cols).abs().sum(dim=-1) + 1e-7
    )


def test_competitive_semantic_slots_separate_opposite_token_evidence():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    tokens = torch.zeros(1, 2, 16)
    tokens[0, 0, 0] = 1.0
    tokens[0, 1, 0] = -1.0
    prototypes = torch.zeros(2, 16)
    prototypes[0, 0] = 1.0
    prototypes[1, 0] = -1.0
    _, weights = model._semantic_slots(tokens, prototypes)
    assert weights[0, 0, 0] > weights[0, 1, 0]
    assert weights[0, 1, 1] > weights[0, 0, 1]


def test_stage_risk_contrast_uses_observed_high_and_low_risk_sets():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    low = torch.zeros(2, 4)
    high = torch.zeros(2, 4)
    low[1, 0] = 1.0
    high[0, 0] = 1.0
    separated = model._stage_risk_contrast_loss(
        torch.tensor([[4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, -4.0]]), low, high
    )
    reversed_loss = model._stage_risk_contrast_loss(
        torch.tensor([[-4.0, -4.0, -4.0, -4.0], [4.0, -4.0, -4.0, -4.0]]), low, high
    )
    assert separated == 0
    assert reversed_loss > 0


def test_ipcw_pairwise_rank_matches_cindex_direction():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    event_time = torch.tensor([1.0, 3.0])
    censorship = torch.tensor([0.0, 1.0])
    correctly_ranked = model._ipcw_pairwise_ranking_loss(
        torch.tensor([[4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, -4.0]]),
        event_time,
        censorship,
    )
    reversed_rank = model._ipcw_pairwise_ranking_loss(
        torch.tensor([[-4.0, -4.0, -4.0, -4.0], [4.0, -4.0, -4.0, -4.0]]),
        event_time,
        censorship,
    )
    assert correctly_ranked < reversed_rank
    assert model.last_ipcw_pair_count.item() == 1


def test_etar_is_finite_and_records_transport_diagnostics():
    torch.manual_seed(7)
    args = make_args()
    args.dct_lambda_ipcw_rank = 0.0
    args.dct_lambda_etar = 0.10
    model = DistributionalCounterfactualTransport(args, omic_input_dim=20)
    model.configure_train_reference(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    model.train()
    logits, aux_loss = model(
        x_wsi=torch.randn(5, 6, 16),
        x_omics=torch.randn(5, 5, 20),
        event_time=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        c=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert logits.shape == (5, 4)
    assert torch.isfinite(aux_loss)
    assert model.last_training_losses["etar_pairs"] > 0
    assert model.last_training_losses["etar_evidence"] >= args.dct_etar_evidence_floor
    assert torch.allclose(
        aux_loss, args.dct_lambda_etar * model.last_training_losses["etar"]
    )
    aux_loss.backward()


def test_ipcw_rank_memory_adds_cross_batch_pairs_without_attaching_old_logits():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    model.dct_ipcw_rank_memory_size = 4
    model.configure_train_reference(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    model.train()
    model._remember_ipcw_batch(
        torch.tensor([0.0, 0.0]),
        torch.tensor([4.0, 5.0]),
        torch.tensor([1.0, 1.0]),
    )
    logits = torch.tensor(
        [[4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, -4.0]],
        requires_grad=True,
    )
    loss = model._ipcw_pairwise_ranking_loss(
        logits,
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert model.last_ipcw_pair_count.item() > 1
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert model._rank_memory_risk.requires_grad is False


def test_log_sinkhorn_projects_extreme_nonfinite_costs_to_finite_plan():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    cost = torch.tensor(
        [[[float("nan"), float("inf"), -float("inf")], [1e20, -1e20, 0.0], [0.0, 2.0, 3.0]]]
    )
    rows = torch.full((1, 3), 1.0 / 3.0)
    cols = torch.full((1, 3), 1.0 / 3.0)
    plan = model._log_sinkhorn(cost, rows, cols, eps=0.0, max_iter=20)
    assert torch.isfinite(plan).all()
    assert torch.all(plan >= 0)
