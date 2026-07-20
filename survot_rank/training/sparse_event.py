"""Training utilities for survival cohorts with sparse observed events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def event_sampling_weights(censorship, target_event_fraction: float) -> torch.Tensor:
    """Return per-patient weights with the requested event sampling mass.

    ``censorship == 0`` denotes an observed event in this repository.  Each
    patient within a group receives the same probability, while the total
    probability mass assigned to observed events is ``target_event_fraction``.
    """

    target = float(target_event_fraction)
    if not 0.0 < target < 1.0:
        raise ValueError("event_sampling_fraction must be strictly between 0 and 1")

    censorship = np.asarray(censorship, dtype=np.float64).reshape(-1)
    if censorship.size == 0:
        raise ValueError("Cannot build an event-aware sampler for an empty split")
    observed = censorship < 0.5
    num_events = int(observed.sum())
    num_censored = int((~observed).sum())
    if num_events == 0 or num_censored == 0:
        raise ValueError(
            "Event-aware sampling requires at least one observed event and one censored case"
        )

    weights = np.empty(censorship.size, dtype=np.float64)
    weights[observed] = target / num_events
    weights[~observed] = (1.0 - target) / num_censored
    return torch.as_tensor(weights, dtype=torch.double)


def make_event_aware_sampler(
    censorship,
    target_event_fraction: float,
    *,
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Create a deterministic weighted sampler for one training fold."""

    weights = event_sampling_weights(censorship, target_event_fraction)
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples if num_samples is not None else weights.numel()),
        replacement=True,
        generator=generator,
    )


@dataclass
class EarlyStoppingController:
    """Track early stopping without charging patience during LR warmup."""

    patience: int
    min_delta: float = 0.0
    warmup_epochs: int = 0
    best: float = float("-inf")
    bad_epochs: int = 0

    def update(self, epoch: int, score: float) -> bool:
        """Return ``True`` when training should stop after this epoch."""

        if self.patience <= 0 or int(epoch) < self.warmup_epochs:
            return False
        score = float(score)
        if score > self.best + self.min_delta:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
