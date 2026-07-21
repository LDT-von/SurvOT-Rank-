import numpy as np

from survot_rank.training.sparse_event import (
    EarlyStoppingController,
    event_sampling_weights,
    make_event_aware_sampler,
    make_stratified_event_batch_sampler,
)


def test_event_sampling_weights_assign_requested_group_mass():
    censorship = np.array([0, 0] + [1] * 18, dtype=float)
    weights = event_sampling_weights(censorship, 0.25).numpy()

    assert np.isclose(weights[:2].sum(), 0.25)
    assert np.isclose(weights[2:].sum(), 0.75)
    assert np.isclose(weights.sum(), 1.0)


def test_event_aware_sampler_is_deterministic_and_targets_events():
    censorship = np.array([0] * 10 + [1] * 90, dtype=float)
    first = list(make_event_aware_sampler(censorship, 0.25, seed=7, num_samples=4000))
    second = list(make_event_aware_sampler(censorship, 0.25, seed=7, num_samples=4000))

    assert first == second
    sampled_event_fraction = np.mean(np.asarray(first) < 10)
    assert 0.22 < sampled_event_fraction < 0.28


def test_stratified_batches_use_every_patient_once_and_maximise_event_batches():
    censorship = np.array([0] * 3 + [1] * 17, dtype=float)
    first_sampler = make_stratified_event_batch_sampler(censorship, 4, seed=11)
    second_sampler = make_stratified_event_batch_sampler(censorship, 4, seed=11)
    first = list(first_sampler)
    second = list(second_sampler)

    assert first == second
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(20))
    assert len(flattened) == len(set(flattened))
    assert all(1 <= len(batch) <= 4 for batch in first)
    event_batches = sum(any(index < 3 for index in batch) for batch in first)
    assert event_batches == 3


def test_stratified_batches_reshuffle_each_epoch_without_changing_membership():
    censorship = np.array([0] * 8 + [1] * 24, dtype=float)
    sampler = make_stratified_event_batch_sampler(censorship, 8, seed=5)
    epoch_zero = list(sampler)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    assert sorted(index for batch in epoch_zero for index in batch) == list(range(32))
    assert sorted(index for batch in epoch_one for index in batch) == list(range(32))


def test_early_stopping_does_not_spend_patience_during_warmup():
    stopper = EarlyStoppingController(patience=2, min_delta=0.01, warmup_epochs=3)

    assert not stopper.update(0, 0.70)
    assert not stopper.update(1, 0.60)
    assert not stopper.update(2, 0.50)
    assert stopper.bad_epochs == 0
    assert stopper.best == float("-inf")

    assert not stopper.update(3, 0.55)
    assert not stopper.update(4, 0.54)
    assert stopper.update(5, 0.53)
