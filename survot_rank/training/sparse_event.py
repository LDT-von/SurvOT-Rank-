"""Training utilities for survival cohorts with sparse observed events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import BatchSampler, WeightedRandomSampler


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


class StratifiedEventBatchSampler(BatchSampler):
    """Patient-complete batches with observed events spread across the epoch.

    Every patient index is emitted exactly once.  Events are assigned to
    different batches before censored patients fill the remaining capacity, so
    the number of event-containing batches is maximal without changing the
    cohort distribution or duplicating rare event cases.
    """

    def __init__(self, censorship, batch_size: int, *, seed: int):
        censorship = np.asarray(censorship, dtype=np.float64).reshape(-1)
        if censorship.size == 0:
            raise ValueError("Cannot build stratified batches for an empty split")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.event_indices = np.flatnonzero(censorship < 0.5).tolist()
        self.censored_indices = np.flatnonzero(censorship >= 0.5).tolist()
        if not self.event_indices or not self.censored_indices:
            raise ValueError(
                "Stratified event batches require observed events and censored cases"
            )
        self.num_samples = int(censorship.size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        events = [
            self.event_indices[i]
            for i in torch.randperm(len(self.event_indices), generator=generator).tolist()
        ]
        censored = [
            self.censored_indices[i]
            for i in torch.randperm(len(self.censored_indices), generator=generator).tolist()
        ]

        batches = [[] for _ in range(len(self))]
        for position, index in enumerate(events):
            batches[position % len(batches)].append(index)

        batch_cursor = 0
        for index in censored:
            while len(batches[batch_cursor]) >= self.batch_size:
                batch_cursor = (batch_cursor + 1) % len(batches)
            batches[batch_cursor].append(index)
            batch_cursor = (batch_cursor + 1) % len(batches)

        order = torch.randperm(len(batches), generator=generator).tolist()
        for batch_index in order:
            batch = batches[batch_index]
            within = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[i] for i in within]


def make_stratified_event_batch_sampler(
    censorship, batch_size: int, *, seed: int
) -> StratifiedEventBatchSampler:
    return StratifiedEventBatchSampler(censorship, batch_size, seed=seed)


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
