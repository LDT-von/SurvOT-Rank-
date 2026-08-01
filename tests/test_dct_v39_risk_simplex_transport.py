"""DCT v3.9 Risk-Simplex Transport 的结构性保证验证。

这些测试检查的是 v3.9 声称的三条结构保证是否为恒等式，而不是"某个损失是否
把模型推向该性质"。因此它们同时也是论文里那三条 proposition 的数值证据。
"""

from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.dct_v39_risk_simplex_transport.model import (
    DCTV39RiskSimplexTransport,
)
from survot_rank.training.model_factory import get_model, list_methods


def make_args(**overrides):
    values = dict(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=4,
        slot_num_omics=4,
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
        dct_anchor_momentum=0.0,
        dct_evidence_mass_floor=0.05,
        dct_evidence_marginal_strength=1.0,
        dct_coupling_projection_tol=1e-4,
        dct_mix_ratio=1.0,
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
        dct_v39_center_slots=True,
        dct_v39_residual_scale=0.0,
        dct_v39_tau_init=0.25,
        dct_v39_anchor_freeze_epoch=0,
        dct_v39_lambda_spread_target=0.0,
        dct_v39_projection_iters=3,
        cur_epoch=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_batch(batch_size=8, omic_dim=16, patches=32):
    torch.manual_seed(0)
    return dict(
        x_wsi=torch.randn(batch_size, patches, 16),
        x_omics=torch.randn(batch_size, omic_dim, 16),
        y=torch.randint(0, 4, (batch_size,)),
        c=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])[:batch_size],
        event_time=torch.tensor(
            [1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        )[:batch_size],
        cur_epoch=0,
    )


def build_model(**overrides):
    model = DCTV39RiskSimplexTransport(make_args(**overrides), omic_input_dim=16)
    times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    censorship = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    model.configure_train_reference(times, censorship)
    return model


def test_registered_in_factory():
    assert "dct_v39_risk_simplex_transport" in list_methods()
    # factory 用 spec_from_file_location 动态加载，返回的类对象与直接 import
    # 的不是同一个，因此按类名而不是 isinstance 断言。
    model = get_model("dct_v39", make_args(), omic_input_dim=16)
    assert type(model).__name__ == "DCTV39RiskSimplexTransport"
    assert not hasattr(model, "shared_wsi_prototypes")


def test_collapsing_prototype_pooling_is_removed():
    """v3.3 的塌缩来源必须真的从模块与 state_dict 中消失，而不是权重置零。"""
    model = build_model()
    assert not hasattr(model, "shared_wsi_prototypes")
    assert not hasattr(model, "shared_omic_prototypes")
    keys = set(model.state_dict())
    assert not any("shared_wsi_prototypes" in key for key in keys)
    assert not any("shared_omic_prototypes" in key for key in keys)
    # learned slot query 接手索引稳定性的职责
    assert model.slot_attention_wsi.slot_queries is not None


def test_centering_removes_common_mode_component():
    """保证 1/3：中心化后 sum(v_i)=0，故两两内积之和严格为 -sum||v_i||^2 < 0。

    严格成立的是内积形式；只有当各 slot 范数相等时，两两余弦均值才恰为
    -1/(K-1)，因此余弦只做宽松检查。这个区别很重要 —— 论文里的命题必须写
    内积形式，不能写成余弦恒等式。
    """
    torch.manual_seed(0)
    slots = torch.randn(6, 8, 32)
    centered = DCTV39RiskSimplexTransport._center_slots(slots)

    # 严格性质 1：跨 slot 均值为零
    assert torch.allclose(
        centered.mean(dim=1), torch.zeros_like(centered.mean(dim=1)), atol=1e-5
    )

    # 严格性质 2：两两内积之和 = -sum ||v_i||^2
    gram = torch.bmm(centered, centered.transpose(1, 2))
    k = gram.size(1)
    mask = ~torch.eye(k, dtype=torch.bool)
    offdiag_sum = gram[:, mask].sum(dim=-1)
    squared_norms = centered.pow(2).sum(dim=-1).sum(dim=-1)
    assert torch.allclose(offdiag_sum, -squared_norms, rtol=1e-4)

    # 宽松性质：余弦均值接近 -1/(K-1)
    normalized = torch.nn.functional.normalize(centered, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    observed = similarity[:, mask].mean().item()
    assert observed == pytest.approx(-1.0 / (k - 1), abs=0.02)


def test_anchor_hazard_ordering_is_an_identity():
    """保证 2/3：h_high 逐时间箱 >= h_low，因此 risk 关于 lambda 严格单调。"""
    model = build_model()
    # 用一个任意的、非默认的参数取值，确认排序不依赖初始化
    with torch.no_grad():
        model.anchor_hazard_low.normal_(0.0, 1.0)
        model.anchor_hazard_gap.normal_(0.0, 1.0)
    hazard_low, hazard_high = model._anchor_hazards()
    assert bool((hazard_high >= hazard_low).all())

    # risk 关于 lambda 单调递增：逐 lambda 扫描应严格递增
    risks = []
    for value in torch.linspace(0.0, 1.0, 11):
        logits = (1.0 - value) * hazard_low + value * hazard_high
        risks.append(model._risk(logits).mean().item())
    differences = torch.diff(torch.tensor(risks))
    assert bool((differences > 0).all()), risks


def test_prediction_stays_in_anchor_convex_hull():
    """保证 3/3：默认无残差旁路时，logits 严格落在锚定 hazard 的凸包内。"""
    model = build_model()
    model.eval()
    with torch.no_grad():
        logits, _ = model(**make_batch())
        hazard_low, hazard_high = model._anchor_hazards()
    # 每个阶段的凸组合再按 stage 权重（和为 1）加权，故仍在逐 bin 的上下界内
    lower = torch.minimum(hazard_low, hazard_high).min(dim=0).values
    upper = torch.maximum(hazard_low, hazard_high).max(dim=0).values
    assert bool((logits >= lower - 1e-5).all())
    assert bool((logits <= upper + 1e-5).all())


def test_lambda_is_bounded_and_reported():
    model = build_model()
    model.train()
    logits, aux = model(**make_batch())
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux)
    coordinate = model._last_lambda
    assert coordinate.shape == (8, model.spt_num_stages)
    assert bool(((coordinate >= 0.0) & (coordinate <= 1.0)).all())
    assert "v39_lambda_mean" in model.last_training_losses


def test_coordinate_endpoints_are_ordered():
    """独立忠实性检验：把锚点代价重新解出的 plan 送回坐标函数，端点应有序。

    这不是训练目标，所以它检验的是坐标定义本身，而不是某个 margin 项是否泛化。
    """
    model = build_model()
    batch = make_batch()
    model.train()
    for _ in range(3):
        model(**batch)  # 先让队列锚点被训练折统计填充
    audit = model.audit_coordinate_endpoints(**batch)
    assert bool((audit["low_anchor"] <= audit["high_anchor"]).all())


def test_objective_defaults_to_two_terms_only():
    """默认目标严格等于 v3.3 的两项：ETAR 与坐标铺开项都不参与。"""
    model = build_model()
    assert model.dct_lambda_etar == 0.0
    assert model.v39_lambda_spread_target == 0.0
    rank_loss = torch.tensor(2.0)
    total = model._combine_auxiliary_objectives(
        ipcw_rank_loss=rank_loss,
        etar_loss=torch.tensor(100.0),
        transport_objective=torch.tensor(0.0),
        transport_metrics={},
        epoch=0,
    )
    assert total.item() == pytest.approx(0.10 * 2.0)


def test_residual_bypass_is_opt_in():
    """残差旁路默认关闭；开启后预测可以离开凸包，用于容量消融。"""
    strict = build_model()
    relaxed = build_model(dct_v39_residual_scale=0.5)
    relaxed.load_state_dict(strict.state_dict())
    strict.eval()
    relaxed.eval()
    batch = make_batch()
    with torch.no_grad():
        strict_logits, _ = strict(**batch)
        relaxed_logits, _ = relaxed(**batch)
    assert not torch.allclose(strict_logits, relaxed_logits)


def test_anchor_freeze_stops_updates():
    model = build_model(dct_v39_anchor_freeze_epoch=1)
    batch = make_batch()
    model.train()
    model(**batch)
    assert bool(model.risk_anchor_seen.all()), "锚点应在第一个 batch 后被填充"
    model.args.cur_epoch = 5
    before = model.risk_anchor_costs.clone()
    model(**dict(batch, cur_epoch=5))
    assert torch.equal(before, model.risk_anchor_costs)


def test_backward_survives_inplace_anchor_update():
    """回归测试：锚点 buffer 被 lerp_ 原地更新，坐标通路必须先 clone。

    v3.9 把队列锚点拉进了训练前向路径（v3.3 只在评估分支读它）。若不 clone，
    第二次 forward 的 backward 会因张量版本被原地修改而直接抛 RuntimeError。
    """
    model = build_model()
    batch = make_batch()
    model.train()
    model(**batch)  # 第一次前向：填充并原地更新锚点
    logits, aux = model(**batch)
    (logits.square().mean() + aux).backward()  # 不 clone 时这里会崩
    named = dict(model.named_parameters())
    # 坐标通路必须真的把梯度送回运输代价与 slot，否则机制只是装饰
    assert named["stage_pair_cost.3.weight"].grad.abs().mean() > 0
    assert named["anchor_hazard_low"].grad.abs().mean() > 0
    assert named["v39_log_tau"].grad is not None


def test_tau_autoscale_prevents_degenerate_lambda():
    """温度必须匹配代价差尺度，否则 lambda 挤在 0.5 附近导致学不动。

    实测锚点代价差尺度约 0.01：tau=0.25 时 lambda 的标准差仅 0.035，
    自动标定后可达 0.1 以上。
    """
    calibrated = build_model(dct_v39_tau_autoscale=True, dct_v39_tau_init=0.02)
    mismatched = build_model(dct_v39_tau_autoscale=False, dct_v39_tau_init=5.0)
    batch = make_batch()
    for model in (calibrated, mismatched):
        model.train()
        # 第一个 batch 锚点还没就绪（坐标在 `_update_risk_anchors` 之前算），
        # 此时没有有意义的尺度可标定；标定发生在锚点就绪后的首个 batch。
        model(**batch)
        assert not bool(model.v39_tau_calibrated)
        model(**batch)
    assert bool(calibrated.v39_tau_calibrated)
    assert not bool(mismatched.v39_tau_calibrated)
    # 错配的大温度必然把坐标压向 0.5
    assert mismatched._last_lambda.std() < calibrated._last_lambda.std()
    assert abs(mismatched._last_lambda.mean().item() - 0.5) < 0.02


def test_centering_recovers_transport_information():
    """删掉塌缩池化 + 中心化后，OT plan 必须显著偏离均匀分布。

    这是 v3.9 相对 v3.3 的核心可测差异：v3.3 的 plan 偏离均匀度实测仅 0.01,
    运输机制在数值上不携带患者特异信息。
    """
    model = build_model()
    model.eval()
    batch = make_batch()
    with torch.no_grad():
        slots_wsi, slots_omic, _, _ = model._encode_transport_slots(
            model.wsi_mlp(batch["x_wsi"]), model._encode_omics(batch), batch
        )
        costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
        plans, _ = model._plans_from_cost_tensor(costs, rows, cols, 0)
    plan = plans[0][0]
    uniform = 1.0 / (plan.size(1) * plan.size(2))
    deviation = (plan - uniform).abs().sum(dim=(1, 2)).mean().item() / 2.0
    assert deviation > 0.20, f"运输计划过于接近均匀分布: {deviation}"
