import numpy as np

from survot_rank.training.sparse_event import (
    EarlyStoppingController,
    event_sampling_weights,
    make_event_aware_sampler,
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
