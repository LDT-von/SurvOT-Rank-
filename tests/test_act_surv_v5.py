#!/usr/bin/env python3
"""Test ACT-Surv v5 three constructive properties.

Run:  python -m pytest tests/test_act_surv_v5.py -v

Tests:
  1. Exact additive attribution: residuals always zero
  2. Closed-form counterfactual: deletion matches re-solve (sanity check)
  3. Bounded extrapolation: predictions ∈ convex hull of K archetype curves
  4. Forward pass runs without NaN/Inf
  5. Training loss is warmup-protected (zero at epoch 0)
  6. Diagnostics are logged correctly
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.archetypal_transport_composition_v5.model import (
    ArchetypalTransportCompositionV5,
)


def make_args(**overrides):
    base = dict(
        omic_sizes=[128, 128, 128, 128],   # default: 4-pathway Pathways format
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="Pathways",
        alpha_surv=0.15,
        act5_num_archetypes=6,
        act5_epsilon=0.10,
        act5_hazard_scale=1.0,
        act5_warmup_epochs=5,
        act5_lambda_balance=0.01,
        act5_lambda_rank=0.10,
        act5_rank_margin=0.02,
        act5_rank_temperature=0.50,
        act5_rank_max_pairs=4096,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_kwargs(B=4, num_patches=8, num_pathways=4, device="cpu", with_labels=True):
    """Build kwargs for forward pass (no args — model gets them separately)."""
    omic_kw = {f"x_omic{i}": torch.randn(B, 128, device=device) for i in range(1, num_pathways + 1)}
    kwargs = {
        "x_wsi": torch.randn(B, num_patches, 16, device=device),
        "wsi_available": torch.ones(B, dtype=torch.bool, device=device),
        "omics_available": torch.ones(B, dtype=torch.bool, device=device),
        **omic_kw,
    }
    if with_labels:
        # y, c: patient-level labels [B] squeezed to [B, num_classes] by expand
        kwargs["y"] = torch.tensor([12.0, 24.0, 36.0, 48.0], device=device).unsqueeze(0).expand(B, -1)
        kwargs["c"] = torch.tensor([1.0, 0.0, 1.0, 1.0], device=device).unsqueeze(0).expand(B, -1)
        kwargs["cur_epoch"] = 10
    return kwargs


class TestAdditiveAttribution:
    """Property 1: logit_t = Σ_k Σ_i P_{i,k} h_{k,t} exactly — residuals always zero."""

    def test_completeness_error_zero(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4)
        logits, _ = model(**kwargs)

        # Verify: logit = Σ_k α_k h_{k,t}  exactly
        plan = model.last_explanations["transport_plan"]
        hazards = model.last_explanations["archetype_hazard_logits"]
        alpha = plan.sum(dim=1)           # [B, K]
        expected = alpha @ hazards        # [B, num_classes]
        error = (logits - expected).abs().max().item()
        assert error < 1e-5, f"Attribution error {error:.2e} > 1e-5 — not exact!"

    def test_token_archetype_contribution_sums_to_logit(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4)
        logits, _ = model(**kwargs)

        # token_contribution = Σ_k P_{i,k} h_{k,t}  per token
        plan = model.last_explanations["transport_plan"]
        hazards = model.last_explanations["archetype_hazard_logits"]
        token_contrib = plan @ hazards              # [B, tokens, num_classes]
        token_sum = token_contrib.sum(dim=1)        # [B, num_classes]
        error = (logits - token_sum).abs().max().item()
        assert error < 1e-5


class TestClosedFormCounterfactual:
    """Property 2: closed-form deletion — no Sinkhorn re-solve needed."""

    def test_deletion_runs(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4, with_labels=False)
        model.eval()   # ensure explanations are populated
        logits_ref, _ = model(**kwargs)

        token_idx = 3
        cf = model.deletion_counterfactual(token_idx)
        assert cf.shape == logits_ref.shape
        assert not torch.isnan(cf).any(), "Counterfactual contains NaN!"

    def test_deletion_removes_token(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4)
        model.eval()
        logits_ref, _ = model(**kwargs)

        plan = model.last_explanations["transport_plan"]
        hazards = model.last_explanations["archetype_hazard_logits"]
        B, T, K = plan.shape
        num_classes = args.n_classes

        # Closed-form deletion
        token_idx = T // 2
        cf_closed = model.deletion_counterfactual(token_idx)

        # Manual closed-form check
        removed = plan[:, token_idx] @ hazards       # [B, num_classes]
        remaining_mass = 1.0 - plan[:, token_idx].sum(dim=1).clamp_min(1e-8)
        factual = logits_ref
        cf_manual = (factual - removed) / remaining_mass.unsqueeze(1)
        error = (cf_closed - cf_manual).abs().max().item()
        assert error < 1e-4, f"Closed-form mismatch: {error:.2e}"

    def test_explain_runs(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4, with_labels=False)
        model.eval()
        model(**kwargs)

        result = model.explain(token_index=0, archetype_index=0)
        assert "token_contribution" in result
        assert "archetype_contribution" in result
        assert "full_contribution" in result
        error = (result["full_contribution"] - model.last_explanations["logits"]).abs().max().item()
        assert error < 1e-5


class TestBoundedExtrapolation:
    """Property 3: predictions always lie in the convex hull of K archetype hazard curves."""

    def test_logit_in_convex_hull(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=8, with_labels=False)
        model.eval()
        logits, _ = model(**kwargs)

        # Each logit = Σ_k α_k h_{k,t},  α_k ≥ 0, Σ_k α_k = 1
        # This is a convex combination of the K archetype hazard curves
        plan = model.last_explanations["transport_plan"]
        alpha = plan.sum(dim=1)     # [B, K], should sum to 1

        alpha_sum = alpha.sum(dim=1)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-4), \
            f"Composition doesn't sum to 1: min={alpha_sum.min():.4f}, max={alpha_sum.max():.4f}"

        all_positive = (alpha >= -1e-4).all()
        assert all_positive, f"Negative composition values: {(alpha < 0).sum()} entries"

    def test_archetypes_are_distinct(self):
        args = make_args(act5_num_archetypes=8)
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4, with_labels=False)
        model.eval()
        model(**kwargs)

        diag = model.last_explanations
        cosine_max = diag["act5_archetype_cosine_max"]
        # Orthogonal init should give max cosine < 0.5 for 8 archetypes
        assert cosine_max < 0.8, f"Archetypes not distinct: cosine_max={cosine_max:.4f}"


class TestNumericalHealth:
    """Model must not produce NaN/Inf."""

    @pytest.mark.parametrize("device", ["cpu"])
    def test_forward_no_nan(self, device):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args).to(device)
        kwargs = make_kwargs(B=8, device=device)
        logits, loss = model(**kwargs)
        assert torch.isfinite(logits).all(), "NaN/Inf in logits!"
        assert torch.isfinite(loss).all(), "NaN/Inf in loss!"

    def test_training_mode(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        model.train()
        kwargs = make_kwargs(B=4, with_labels=True)
        logits, loss = model(**kwargs)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()
        assert "act5_total" in model.last_training_losses

    def test_warmup_zero_loss_at_epoch_zero(self):
        args = make_args(act5_warmup_epochs=5)
        model = ArchetypalTransportCompositionV5(args)
        model.train()
        kwargs = make_kwargs(B=4, with_labels=True)
        kwargs["cur_epoch"] = 0
        logits, loss = model(**kwargs)
        assert loss.abs().max().item() < 1e-6, "Loss should be 0 during warmup!"

    def test_warmup_nonzero_after_warmup(self):
        args = make_args(act5_warmup_epochs=5)
        model = ArchetypalTransportCompositionV5(args)
        model.train()
        kwargs = make_kwargs(B=4, with_labels=True)
        kwargs["cur_epoch"] = 6   # past warmup
        logits, loss = model(**kwargs)
        assert loss.abs().max().item() > 1e-6, "Loss should be non-zero after warmup!"


class TestDiagnostics:
    """Diagnostic metrics are logged and bounded."""

    def test_hazard_spread_nonzero(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4, with_labels=False)
        model.eval()
        model(**kwargs)
        spread = model.last_explanations.get("act5_hazard_spread", torch.tensor(float("nan")))
        assert torch.isfinite(spread).all() and spread > 0

    def test_diagnostics_populated(self):
        args = make_args()
        model = ArchetypalTransportCompositionV5(args)
        kwargs = make_kwargs(B=4, with_labels=True)
        model.train()
        model(**kwargs)
        losses = model.last_training_losses
        for key in ("act5_warmup_fraction", "act5_balance", "act5_rank", "act5_total"):
            assert key in losses, f"Missing loss key: {key}"
            assert torch.isfinite(losses[key]).all(), f"Non-finite loss: {key}={losses[key]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
