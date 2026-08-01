"""DCT v3.8.3 验证：中心化表征让 v3.8 的干预损失作用在真信号而非噪声上。

核心断言不是"分数会涨"（那需要真实训练），而是 v3.8.3 相对 v3.8 的唯一改动
（中心化运输几何）确实把干预响应从退化状态恢复到可用状态，且继承的三个损失
仍然可微、有限、可反向传播。
"""

from types import SimpleNamespace

import torch

from survot_rank.training.model_factory import get_model, list_methods


def make_args(**overrides):
    values = dict(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=6,
        slot_num_omics=6,
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
        dct_lambda_etar=0.0,
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_evidence_marginal_strength=1.0,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=1.0,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.03,
        dct_v38_lambda_reconfiguration=0.02,
        dct_v38_direction_margin=0.02,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.05,
        dct_v38_alpha_mid=0.50,
        dct_v38_alpha_full=1.0,
        dct_v38_warmup_epochs=0,
        dct_v38_ramp_epochs=0,
        dct_v38_dose_every=1,
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        dct_slot_eval_seed=91,
        dct_v383_center_slots=True,
        cur_epoch=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_batch(batch_size=16):
    torch.manual_seed(0)
    half = batch_size // 2
    return dict(
        x_wsi=torch.randn(batch_size, 64, 16),
        x_omics=torch.randn(batch_size, 24, 16),
        y=torch.randint(0, 4, (batch_size,)),
        c=torch.tensor([0.0] * half + [1.0] * (batch_size - half)),
        event_time=torch.arange(1.0, batch_size + 1.0),
        cur_epoch=3,
    )


def build(cls_or_method, **overrides):
    model = get_model(cls_or_method, make_args(**overrides), omic_input_dim=16)
    batch = make_batch()
    model.configure_train_reference(batch["event_time"], batch["c"])
    return model, batch


def _plan_deviation(model, batch):
    with torch.no_grad():
        slots_wsi, slots_omic, _, _ = model._encode_transport_slots(
            model.wsi_mlp(batch["x_wsi"]), model._encode_omics(batch), batch
        )
        costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
        plans, _ = model._plans_from_cost_tensor(costs, rows, cols, 3)
    plan = plans[0][0]
    uniform = 1.0 / (plan.size(1) * plan.size(2))
    return (plan - uniform).abs().sum(dim=(1, 2)).mean().item() / 2.0


def test_registered_in_factory():
    assert "dct_v383_intervention_consistency_centered" in list_methods()
    model = get_model("dct_v383", make_args(), omic_input_dim=16)
    assert type(model).__name__ == "DCTV383InterventionConsistencyCentered"


def test_collapsing_prototypes_removed():
    model, _ = build("dct_v383")
    assert not hasattr(model, "shared_wsi_prototypes")
    assert not hasattr(model, "shared_omic_prototypes")
    keys = set(model.state_dict())
    assert not any("shared_wsi_prototypes" in key for key in keys)
    assert not any("shared_omic_prototypes" in key for key in keys)


def test_inherits_v38_losses():
    """v3.8.3 必须原样继承 v3.8 的三个损失与超参，只改表征。"""
    model, _ = build("dct_v383")
    assert model.dct_v38_lambda_direction == 0.05
    assert model.dct_v38_lambda_dose == 0.03
    assert model.dct_v38_lambda_reconfiguration == 0.02
    # 干预目标 hook 来自 v3.8，不是 v3.3 的零 hook
    assert (
        type(model)._training_transport_objective
        is not __import__(
            "survot_rank.research.methods.distributional_counterfactual_transport.model",
            fromlist=["DistributionalCounterfactualTransport"],
        ).DistributionalCounterfactualTransport._training_transport_objective
    )


def test_centering_recovers_transport_information():
    """开启中心化后 plan 显著偏离均匀；关闭后退回塌缩，形成单变量对照。"""
    centered, batch = build("dct_v383", dct_v383_center_slots=True)
    collapsed, _ = build("dct_v383", dct_v383_center_slots=False)
    collapsed.load_state_dict(centered.state_dict())
    centered.eval()
    collapsed.eval()
    dev_centered = _plan_deviation(centered, batch)
    dev_collapsed = _plan_deviation(collapsed, batch)
    assert dev_centered > 0.20, dev_centered
    assert dev_centered > dev_collapsed


def test_intervention_losses_are_finite_and_differentiable():
    model, batch = build("dct_v383")
    model.train()
    for _ in range(3):  # 填满 anchor 并越过 warmup
        logits, aux = model(**batch)
    logits, aux = model(**batch)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux)
    (logits.square().mean() + aux).backward()
    named = dict(model.named_parameters())
    # 干预损失通路必须把梯度送回运输代价，否则损失又是装饰
    assert named["stage_pair_cost.3.weight"].grad is not None
    assert named["stage_pair_cost.3.weight"].grad.abs().mean() > 0


def test_v38_metrics_reported():
    model, batch = build("dct_v383")
    model.train()
    for _ in range(3):
        model(**batch)
    assert model.last_training_losses["v38_finite"].item() == 1.0
    assert "v38_direction" in model.last_training_losses
    assert "v38_high_plan_shift" in model.last_training_losses


def test_center_toggle_changes_output():
    centered, batch = build("dct_v383", dct_v383_center_slots=True)
    collapsed, _ = build("dct_v383", dct_v383_center_slots=False)
    collapsed.load_state_dict(centered.state_dict())
    centered.eval()
    collapsed.eval()
    with torch.no_grad():
        logits_centered, _ = centered(**batch)
        logits_collapsed, _ = collapsed(**batch)
    assert not torch.allclose(logits_centered, logits_collapsed)
