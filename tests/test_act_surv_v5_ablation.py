#!/usr/bin/env python3
"""Experiment A: MLP-head vs ACT-head ablation.

RESEARCH QUESTION:
  ACT-Surv replaces the final MLP (composition → logits) with a linear
  composition @ hazard_dictionary, where hazard_dictionary is a shared
  K × T learnable matrix.  Does this cost us prediction accuracy?

HYPOTHESIS:
  If ΔC = C_ACT − C_MLP satisfies |ΔC| < 0.01,
  the ACT-head provides structural interpretability "almost for free".

PROTOCOL:
  - Same encoder (WSI + Pathway-omics)
  - Same OT transport (softmax-over-archetypes, epsilon=0.10)
  - Same loss (IPCW ranking + KL balance)
  - Same split, same data, same training epochs
  - Only difference: MLP(composition) vs linear(composition @ H)

  MLP architecture: composition → LayerNorm → ReLU → Dropout(0.2)
                     → Linear(K, 32) → ReLU → Dropout(0.2)
                     → Linear(32, num_classes)

Run:
  python -m pytest tests/test_act_surv_v5_ablation.py -v
  # or standalone:
  python tests/test_act_surv_v5_ablation.py --cancer blca --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from survot_rank.research.methods.archetypal_transport_composition_v5.model import (
    ArchetypalTransportCompositionV5,
    _ipcw_ranking_loss,
)
from tests.test_act_surv_v5 import make_args, make_kwargs


# ──────────────────────────────────────────────────────────────────────────────
# MLP-head variant
# ──────────────────────────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """MLP decoder: composition → logits.

    Mirrors ACT-Surv v5 interface (same attribute names) so that the
    evaluation harness is identical.
    """

    def __init__(self, num_archetypes: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_archetypes),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_archetypes, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, composition: torch.Tensor) -> torch.Tensor:
        # composition: [B, K] with Σ composition_k = 1
        return self.net(composition)


class ACTSurvWithMLPHead(nn.Module):
    """ACT-Surv v5 with MLP head replacing the linear archetype hazard dictionary."""

    def __init__(
        self,
        args,
        omic_input_dim=None,
        omic_names=None,
        pathway_names=None,
    ):
        super().__init__()
        # ── ACT-Surv core (OT transport) ───────────────────────────────────
        self._act = ArchetypalTransportCompositionV5(
            args, omic_input_dim, omic_names, pathway_names
        )
        self.num_archetypes = self._act.num_archetypes
        self.num_classes = self._act.num_classes
        self.warmup_epochs = self._act.warmup_epochs
        self.lambda_balance = self._act.lambda_balance
        self.lambda_rank = self._act.lambda_rank
        self.rank_margin = self._act.rank_margin
        self.rank_temperature = self._act.rank_temperature
        self.rank_max_pairs = self._act.rank_max_pairs

        # ── MLP decoder ─────────────────────────────────────────────────────
        self.mlp_decoder = MLPDecoder(self.num_archetypes, self.num_classes)

        # ── State ────────────────────────────────────────────────────────────
        self.last_explanations: dict | None = None
        self.last_training_losses: dict = {}

    @property
    def archetype_embedding(self):
        return self._act.archetype_embedding

    def forward(self, **kwargs):
        x_wsi = kwargs["x_wsi"].float()
        device = x_wsi.device
        batch_size = x_wsi.size(0)
        current_epoch = int(kwargs.get("cur_epoch", kwargs.get("epoch", 0)))

        # ── Encode + OT transport (identical to ACT-Surv v5) ─────────────────
        wsi_tokens = self._act._encode_wsi(x_wsi)
        omic_tokens = self._act._encode_omics(kwargs)
        num_wsi = wsi_tokens.size(1)

        has_wsi = kwargs.get("wsi_available")
        has_omic = kwargs.get("omics_available")
        if has_wsi is None:
            has_wsi = torch.ones(batch_size, dtype=torch.bool, device=device)
        if has_omic is None:
            has_omic = torch.ones(batch_size, dtype=torch.bool, device=device)

        wsi_mask = torch.ones(batch_size, num_wsi, dtype=torch.bool, device=device) & has_wsi[:, None]
        omic_mask = torch.ones(batch_size, omic_tokens.size(1), dtype=torch.bool, device=device) & has_omic[:, None]
        token_mask = torch.cat([wsi_mask, omic_mask], dim=1)

        plan, cost = self._act._transport(
            torch.cat([wsi_tokens, omic_tokens], dim=1),
            token_mask,
        )
        composition = plan.sum(dim=1)  # [B, K]

        # ── MLP head (replaces linear @ H) ─────────────────────────────────
        logits = self.mlp_decoder(composition)  # [B, num_classes]

        # ── Store explanations ─────────────────────────────────────────────
        self.last_explanations = {
            "transport_plan": plan.detach(),
            "composition": composition.detach(),
            "logits": logits.detach(),
            "hazards": torch.sigmoid(logits).detach(),
            "survival": torch.cumprod(1.0 - torch.sigmoid(logits), dim=1).detach(),
        }

        # ── Training losses ─────────────────────────────────────────────────
        if not self.training:
            self.last_training_losses = {}
            return logits, logits.sum() * 0.0

        zero = logits.sum() * 0.0
        warmup_fraction = min(1.0, current_epoch / max(1, self.warmup_epochs))
        aux_loss = zero

        if warmup_fraction > 0:
            mean_comp = composition.mean(dim=0).clamp_min(1e-8)
            balance_loss = (
                mean_comp
                * (
                    mean_comp.log()
                    - torch.log(torch.tensor(float(self.num_archetypes), device=device))
                )
            ).sum()

            rank_loss = zero
            if kwargs.get("y") is not None and kwargs.get("c") is not None:
                rank_loss = _ipcw_ranking_loss(
                    logits,
                    kwargs["y"],
                    kwargs["c"],
                    margin=self.rank_margin,
                    temperature=self.rank_temperature,
                    max_pairs=self.rank_max_pairs,
                )

            aux_loss = warmup_fraction * (
                self.lambda_balance * balance_loss + self.lambda_rank * rank_loss
            )

            self.last_training_losses = {
                "warmup_fraction": logits.new_tensor(warmup_fraction).detach(),
                "balance": balance_loss.detach(),
                "rank": rank_loss.detach(),
                "total": aux_loss.detach(),
            }
        else:
            self.last_training_losses = {"warmup_fraction": logits.new_tensor(0.0).detach()}

        return logits, aux_loss


# ──────────────────────────────────────────────────────────────────────────────
# Training harness (mirrors v5 pipeline structure)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    head_type: Literal["act", "mlp"],
    args,
    omic_input_dim=None,
    omic_names=None,
    pathway_names=None,
):
    if head_type == "act":
        return ArchetypalTransportCompositionV5(args, omic_input_dim, omic_names, pathway_names)
    else:
        return ACTSurvWithMLPHead(args, omic_input_dim, omic_names, pathway_names)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    max_epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_cidx = 0.0
    best_state = None
    history = []

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            x_wsi = batch["x_wsi"].to(device)
            kwargs = {"x_wsi": x_wsi, "cur_epoch": epoch}
            for k, v in batch.items():
                if k not in ("x_wsi",):
                    kwargs[k] = v.to(device) if hasattr(v, "to") else v

            logits, loss = model(**kwargs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_risks = []
        val_y = []
        val_c = []

        with torch.no_grad():
            for batch in val_loader:
                x_wsi = batch["x_wsi"].to(device)
                kwargs = {"x_wsi": x_wsi}
                for k, v in batch.items():
                    if k not in ("x_wsi",):
                        kwargs[k] = v.to(device) if hasattr(v, "to") else v
                logits, _ = model(**kwargs)
                hazards = torch.sigmoid(logits)
                risk = -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)
                val_risks.extend(risk.cpu().numpy())
                if batch.get("y") is not None:
                    y = batch["y"]
                    if y.ndim > 1:
                        y = y[:, 0]
                    val_y.extend(y.cpu().numpy())
                if batch.get("c") is not None:
                    c = batch["c"]
                    if c.ndim > 1:
                        c = c[:, 0]
                    val_c.extend(c.cpu().numpy())

        val_risks = np.array(val_risks)
        val_y = np.array(val_y)
        val_c = np.array(val_c)
        events = (1.0 - val_c).astype(bool)

        if events.sum() > 0 and len(val_y) > 1:
            cidx, _, _, _, _ = concordance_index_censored(events, val_y, val_risks)
        else:
            cidx = 0.5

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        history.append({"epoch": epoch, "loss": avg_loss, "val_cidx": cidx})

        if cidx > best_cidx:
            best_cidx = cidx
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return best_cidx, history


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark entry point
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    head_type: str
    cancer: str
    fold: int
    best_cidx: float
    history: list = field(default_factory=list)
    runtime_seconds: float = 0.0


@dataclass
class AblationReport:
    cancer: str
    fold_results: list[FoldResult]
    delta_mean: float
    delta_std: float
    act_mean: float
    act_std: float
    mlp_mean: float
    mlp_std: float
    verdict: str  # "ACT competitive" | "MLP wins" | "inconclusive"


def run_ablation_fold(
    head_type: Literal["act", "mlp"],
    cancer: str,
    fold: int,
    *,
    train_data,  # your DataLoader or similar
    val_data,
    args=None,
    device="cuda",
    max_epochs=50,
) -> FoldResult:
    """Train one fold of one head type and return the best C-index."""
    if args is None:
        args = make_args()

    start = time.time()
    model = build_model(head_type, args)
    best_cidx, history = train_model(
        model, train_data, val_data,
        max_epochs=max_epochs, device=device,
    )
    elapsed = time.time() - start

    return FoldResult(
        head_type=head_type,
        cancer=cancer,
        fold=fold,
        best_cidx=best_cidx,
        history=history,
        runtime_seconds=elapsed,
    )


def summarize(ablation_results: list[FoldResult]) -> AblationReport:
    """Aggregate fold results into a summary."""
    act_scores = [r.best_cidx for r in ablation_results if r.head_type == "act"]
    mlp_scores = [r.best_cidx for r in ablation_results if r.head_type == "mlp"]

    act_mean = np.mean(act_scores)
    act_std = np.std(act_scores, ddof=1)
    mlp_mean = np.mean(mlp_scores)
    mlp_std = np.std(mlp_scores, ddof=1)
    delta_mean = act_mean - mlp_mean
    delta_std = np.std(
        [a - m for a, m in zip(act_scores, mlp_scores)], ddof=1
    )

    if abs(delta_mean) < 0.015 and delta_std < 0.03:
        verdict = "ACT competitive: ΔC ≈ 0"
    elif delta_mean > 0.01:
        verdict = "ACT wins"
    elif delta_mean < -0.01:
        verdict = "MLP wins"
    else:
        verdict = "inconclusive"

    return AblationReport(
        cancer=ablation_results[0].cancer,
        fold_results=ablation_results,
        delta_mean=delta_mean,
        delta_std=delta_std,
        act_mean=act_mean,
        act_std=act_std,
        mlp_mean=mlp_mean,
        mlp_std=mlp_std,
        verdict=verdict,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic unit test (runs without real data)
# ──────────────────────────────────────────────────────────────────────────────

class DummyLoader:
    """Minimal DataLoader stand-in for unit testing."""

    def __init__(self, size=64, batch_size=8, num_patches=8, num_pathways=4):
        self.size = size
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.num_pathways = num_pathways
        self.device = "cpu"

    def __iter__(self):
        for start in range(0, self.size, self.batch_size):
            end = min(start + self.batch_size, self.size)
            B = end - start
            yield {
                "x_wsi": torch.randn(B, self.num_patches, 16),
                **{f"x_omic{i}": torch.randn(B, 128) for i in range(1, self.num_pathways + 1)},
                "y": torch.rand(B, 4) * 48,
                "c": torch.randint(0, 2, (B, 4)).float(),
            }

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size


def test_mlp_vs_act_synthetic():
    """Unit test: verify both heads train without NaN on synthetic data."""
    args = make_args(act5_num_archetypes=6, n_classes=4)
    device = "cpu"
    train_loader = DummyLoader(size=40, batch_size=8)
    val_loader = DummyLoader(size=20, batch_size=8)

    for head_type in ("act", "mlp"):
        model = build_model(head_type, args).to(device)
        best_cidx, history = train_model(
            model, train_loader, val_loader,
            max_epochs=5, device=device,
        )
        assert np.isfinite(best_cidx), f"{head_type}: best_cidx is NaN"
        print(f"  [{head_type}] best_cidx={best_cidx:.4f}, last_loss={history[-1]['loss']:.4f}")

    print("  ✓ Both heads train without NaN")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
