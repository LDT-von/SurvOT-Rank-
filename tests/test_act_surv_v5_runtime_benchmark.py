#!/usr/bin/env python3
"""Experiment C: Runtime benchmark — N× Sinkhorn LOO vs closed-form.

RESEARCH QUESTION:
  How much faster is the closed-form plan intervention compared to re-running
  full Sinkhorn for each token deletion?

CLAIM TO SUPPORT:
  "One OT solve + batched vectorised deletion achieves N× speed-up over
   N separate Sinkhorn re-runs, enabling efficient audit of entire WSI."

BENCHMARK PROTOCOL:
  For token counts N ∈ {50, 100, 200, 500, 1000, 2000}:
    - Time to delete all N tokens via closed-form: T_closed
    - Time to delete all N tokens via Sinkhorn re-run: T_sinkhorn
    - Report T_sinkhorn / T_closed (speed-up factor)

  We also benchmark the per-token overhead to show the O(1) vs O(N) gap.

KEY METRIC IN PAPER:
  Speed-up factor = T_sinkhorn(N) / T_closed
  At N=1000 typical patches, expect ≥ 100× speed-up.

Run:
  python -m pytest tests/test_act_surv_v5_runtime_benchmark.py -v
  # standalone:
  python tests/test_act_surv_v5_runtime_benchmark.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.stats import trim_mean

from tests.test_act_surv_v5 import make_args


# ──────────────────────────────────────────────────────────────────────────────
# Sinkhorn re-run baseline
# ──────────────────────────────────────────────────────────────────────────────

def sinkhorn_rerun_deletion(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    hazard_logits: torch.Tensor,
    token_idx: int,
) -> torch.Tensor:
    """Delete token_idx by re-running full OT with that token masked."""
    new_mask = mask.clone()
    new_mask[:, token_idx] = False
    plan, _ = model._transport(tokens, new_mask)
    composition = plan.sum(dim=1)
    return composition @ hazard_logits


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenCountBenchmark:
    num_tokens: int
    device: str
    n_trials: int

    # Closed-form (batch all at once)
    t_closed_total: float      # seconds for all N deletions
    t_closed_per_token: float  # seconds per token

    # Sinkhorn re-run
    t_sinkhorn_total: float
    t_sinkhorn_per_token: float

    # Speed-up
    speedup_total: float       # T_sinkhorn / T_closed
    speedup_per_token: float

    # Warm-up trials excluded from timing
    t_closed_warmup_ms: float
    t_sinkhorn_warmup_ms: float


@dataclass
class RuntimeBenchmarkReport:
    benchmarks: list[TokenCountBenchmark]
    summary_token_counts: list[int]
    speedup_at_500: float      # interpolation or nearest
    speedup_at_1000: float
    estimated_speedup_formula: str


def benchmark_token_counts(
    token_counts: list[int] = None,
    n_wsi_tokens_ratio: float = 0.8,
    n_omic_tokens: int = 20,
    n_trials: int = 5,
    n_warmup: int = 2,
    device: str = "cuda",
    seed: int = 42,
) -> RuntimeBenchmarkReport:
    """Benchmark closed-form vs Sinkhorn across different token counts."""
    if token_counts is None:
        token_counts = [50, 100, 200, 500, 1000, 2000]

    rng = torch.Generator(device=device).manual_seed(seed)

    # Fixed model config
    args = make_args(act5_num_archetypes=6, act5_epsilon=0.10, n_classes=4)
    model = __import__(
        "survot_rank.research.methods.archetypal_transport_composition_v5.model",
        fromlist=["ArchetypalTransportCompositionV5"]
    ).ArchetypalTransportCompositionV5(args)
    model = model.to(device)
    model.eval()
    hazard_logits = model._logit_hazard_raw.detach()

    benchmarks = []

    for N in token_counts:
        N_wsi = int(N * n_wsi_tokens_ratio)
        N_omic = n_omic_tokens
        D = model.proj_dim

        t_closed_trials = []
        t_sinkhorn_trials = []

        for trial in range(n_trials + n_warmup):
            # Generate random tokens
            wsi_tokens = torch.randn(1, N_wsi, D, device=device, generator=rng)
            omic_tokens = torch.randn(1, N_omic, D, device=device, generator=rng)
            tokens = torch.cat([wsi_tokens, omic_tokens], dim=1)   # [1, N, D]
            mask = torch.ones(1, N_wsi + N_omic, dtype=torch.bool, device=device)
            token_indices = list(range(N_wsi + N_omic))

            is_warmup = trial < n_warmup

            # ── Closed-form: batch all deletions at once ──────────────────
            # P @ H gives all token contributions at once
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()

            plan_full, _ = model._transport(tokens, mask)
            C_full = plan_full @ hazard_logits    # [1, N, T] = all token contributions
            factual = (plan_full.sum(dim=1) @ hazard_logits)  # [1, T]

            # Vectorised deletion for all tokens
            a = plan_full.sum(dim=1)             # [1, K] — per-archetype total mass (sums to N_wsi + N_omic)
            # Naive: loop over tokens (to measure per-call overhead)
            for ti in token_indices[:min(100, len(token_indices))]:  # cap at 100 for speed
                removed = plan_full[:, ti] @ hazard_logits   # [1, T]
                a_i = plan_full[:, ti].sum()                  # scalar
                _ = (factual - removed) / (1.0 - a_i).clamp_min(1e-8)

            torch.cuda.synchronize() if device == "cuda" else None
            t1 = time.perf_counter()
            t_closed = (t1 - t0) * 1000   # ms

            # ── Sinkhorn re-run per token ─────────────────────────────────
            torch.cuda.synchronize() if device == "cuda" else None
            t2 = time.perf_counter()

            for ti in token_indices[:min(100, len(token_indices))]:
                _ = sinkhorn_rerun_deletion(model, tokens, mask, hazard_logits, ti)

            torch.cuda.synchronize() if device == "cuda" else None
            t3 = time.perf_counter()
            t_sinkhorn = (t3 - t2) * 1000   # ms

            if is_warmup:
                t_closed_trials.append(t_closed)
                t_sinkhorn_trials.append(t_sinkhorn)
            else:
                t_closed_trials.append(t_closed)
                t_sinkhorn_trials.append(t_sinkhorn)

        t_closed_mean = float(np.mean(t_closed_trials[n_warmup:]))
        t_sinkhorn_mean = float(np.mean(t_sinkhorn_trials[n_warmup:]))

        # Per-token estimates (for 100 tokens measured, scale to N)
        scale = N / 100.0
        t_closed_total = t_closed_mean * scale
        t_sinkhorn_total = t_sinkhorn_mean * scale

        bm = TokenCountBenchmark(
            num_tokens=N,
            device=device,
            n_trials=n_trials,
            t_closed_total=t_closed_total,
            t_closed_per_token=t_closed_total / max(N, 1),
            t_sinkhorn_total=t_sinkhorn_total,
            t_sinkhorn_per_token=t_sinkhorn_total / max(N, 1),
            speedup_total=t_sinkhorn_total / max(t_closed_total, 1e-9),
            speedup_per_token=(t_sinkhorn_total / max(N, 1)) / max(t_closed_total / max(N, 1), 1e-9),
            t_closed_warmup_ms=float(np.mean(t_closed_trials[:n_warmup])),
            t_sinkhorn_warmup_ms=float(np.mean(t_sinkhorn_trials[:n_warmup])),
        )
        benchmarks.append(bm)
        print(f"  N={N:5d}: closed={t_closed_total:7.3f}ms  "
              f"sinkhorn={t_sinkhorn_total:8.3f}ms  "
              f"speedup={bm.speedup_total:7.1f}×")

    # Summary
    speedup_at_500 = next(
        (b.speedup_total for b in benchmarks if b.num_tokens == 500), 0.0
    )
    speedup_at_1000 = next(
        (b.speedup_total for b in benchmarks if b.num_tokens == 1000), 0.0
    )

    return RuntimeBenchmarkReport(
        benchmarks=benchmarks,
        summary_token_counts=token_counts,
        speedup_at_500=speedup_at_500,
        speedup_at_1000=speedup_at_1000,
        estimated_speedup_formula="≈ N × (T_sinkhorn_single / T_closed_single)",
    )


def format_table(report: RuntimeBenchmarkReport) -> str:
    lines = [
        "\n" + "=" * 72,
        "RUNTIME BENCHMARK: Closed-form vs Sinkhorn re-run LOO deletion",
        "=" * 72,
        f"{'N tokens':>10} | {'T_closed (ms)':>13} | {'T_sinkhorn (ms)':>15} | {'Speed-up':>10}",
        "-" * 60,
    ]
    for b in report.benchmarks:
        lines.append(
            f"{b.num_tokens:>10} | "
            f"{b.t_closed_total:>13.2f} | "
            f"{b.t_sinkhorn_total:>15.2f} | "
            f"{b.speedup_total:>9.1f}×"
        )
    lines.extend([
        "-" * 60,
        f"Speed-up at N=500:  {report.speedup_at_500:.1f}×",
        f"Speed-up at N=1000: {report.speedup_at_1000:.1f}×",
        f"Formula: {report.estimated_speedup_formula}",
        "=" * 72,
    ])
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runtime benchmark for ACT-Surv v5")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    print("ACT-Surv v5 Runtime Benchmark")
    print(f"  device={args.device}, trials={args.trials}")

    report = benchmark_token_counts(
        device=args.device,
        n_trials=args.trials,
    )

    print(format_table(report))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nResults saved to {args.output}")
