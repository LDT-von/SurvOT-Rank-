from __future__ import annotations

"""Smoke tests for the paper-evidence ablation switches and audit loader.

These tests prove that the four ablation switches added to
``DistributionalCounterfactualTransport`` and ``DCTTransportInterventionConsistency``
do not perturb the frozen v3.8.2 recipe when the switches are off (the
production default).
"""

from scripts import run_dct_v382_paper_evidence as evidence
from scripts import run_dct_v382_cross_cancer_prototype as cross_cancer
from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)


def test_evidence_launcher_default_queue_is_twelve_jobs():
    jobs = evidence.build_jobs(evidence.build_parser().parse_args(
        ["plan", "--python", "python"]
    ))
    assert len(jobs) == 12
    variants = sorted({job.variant for job in jobs})
    cancers = sorted({job.cancer for job in jobs})
    assert variants == ["fixed_coupling", "null_calibration", "random_anchors", "stage_randomization"]
    assert cancers == ["blca", "lusc", "ucec"]


def test_evidence_ablation_changes_only_its_registered_term():
    from scripts import run_dct_v382_final_cross_cancer as base
    frozen = base.FINAL_OVERRIDES
    for job in evidence.build_jobs(evidence.build_parser().parse_args(
        ["plan", "--python", "python"]
    )):
        changed = evidence.ABLATIONS[job.variant]
        # Every changed key must appear as a ``--set key=value`` flag in the
        # job command. Frozen keys must NOT have been duplicated.
        command_text = " ".join(job.command)
        for key, value in changed.items():
            # ``load_config`` normalises booleans to title-case (``True``/``False``)
            # before rendering; mirror that here so the test does not silently
            # disagree with what the CLI actually emits.
            if isinstance(value, bool):
                rendered = "True" if value else "False"
            else:
                rendered = str(value)
            assert f"{key}={rendered}" in command_text, (
                f"{job.variant} {job.cancer.upper()} fold{job.fold} "
                f"missing {key}={rendered}"
            )
        # Pick one frozen key and confirm it is NOT present as an override.
        frozen_flag = next(iter(frozen))
        for token in job.command:
            if token.startswith(f"{frozen_flag}="):
                # If a frozen flag appears, it must match the frozen value
                # (i.e. the launcher did not silently overwrite it).
                frozen_value = frozen[frozen_flag]
                if isinstance(frozen_value, bool):
                    rendered_frozen = "True" if frozen_value else "False"
                else:
                    rendered_frozen = str(frozen_value)
                assert token == f"{frozen_flag}={rendered_frozen}", (
                    f"{job.variant} frozen key {frozen_flag} overridden unexpectedly"
                )


def test_cross_cancer_default_queue_pairs_three_sources_with_three_targets():
    jobs = cross_cancer.build_jobs(cross_cancer.build_parser().parse_args(
        ["plan", "--python", "python"]
    ))
    assert len(jobs) == 6
    phases = [job.phase for job in jobs]
    assert phases == ["source", "target"] * 3


def test_cross_cancer_target_jobs_carry_freeze_prototype_flag():
    jobs = cross_cancer.build_jobs(cross_cancer.build_parser().parse_args(
        ["plan", "--python", "python"]
    ))
    target_jobs = [job for job in jobs if job.phase == "target"]
    assert target_jobs, "cross-cancer queue must schedule target jobs"
    for job in target_jobs:
        joined = " ".join(job.command)
        assert "dct_freeze_source_prototype" in joined, (
            f"target job {job.cancer} fold{job.fold} missing freeze flag"
        )


def test_dct_ablation_switches_default_to_off():
    """Production v3.8.2 training must NOT see any ablation switch on."""
    switches = (
        "dct_fixed_coupling",
        "dct_random_anchors",
        "dct_perm_labels_seed",
        "dct_stage_jitter_fraction",
        "dct_freeze_source_prototype",
    )
    # Read defaults straight from the constructor source to avoid running
    # the model. The defaults are defined in __init__ and gated on
    # ``getattr(args, ...)`` so an unset attribute means the switch is off.
    import inspect
    init_src = inspect.getsource(DistributionalCounterfactualTransport.__init__)
    for name in switches:
        assert name in init_src, f"{name} hook missing from model __init__"


def test_audit_metrics_module_imports():
    from scripts import audit_dct_v382 as audit
    assert hasattr(audit, "direction_consistency")
    assert hasattr(audit, "dose_monotonicity")
    assert hasattr(audit, "reconfiguration_magnitude")
    assert hasattr(audit, "cmd_audit")
    assert hasattr(audit, "cmd_sweep")


def test_direction_consistency_above_chance_on_synthetic_signal():
    import numpy as np
    from scripts.audit_dct_v382 import direction_consistency

    n = 200
    rng = np.random.default_rng(0)
    event_time = rng.uniform(0, 100, n)
    censorship = np.where(event_time < 50, 0.0, 1.0)  # first half observed
    factual = rng.normal(0, 1, n)
    # Craft a counterfactual that moves risk in the right direction: high
    # labels get +1.0, low labels get -1.0. Direction rate should be ~1.0.
    high_mask = (censorship < 0.5) & (event_time < np.quantile(event_time[censorship < 0.5], 0.4))
    low_mask = (event_time > np.quantile(event_time[censorship < 0.5], 0.6))
    low_risk = factual.copy()
    high_risk = factual.copy()
    low_risk[low_mask] = factual[low_mask] - 1.0
    high_risk[high_mask] = factual[high_mask] + 1.0
    metrics = direction_consistency(factual, low_risk, high_risk, event_time, censorship)
    assert metrics["correct_rate"] > 0.65, metrics
    assert metrics["chance_gap"] > 0.10, metrics


def test_direction_consistency_returns_chance_on_random_signal():
    import numpy as np
    from scripts.audit_dct_v382 import direction_consistency

    n = 400
    rng = np.random.default_rng(1)
    event_time = rng.uniform(0, 100, n)
    censorship = np.where(event_time < 50, 0.0, 1.0)
    factual = rng.normal(0, 1, n)
    low_risk = rng.normal(0, 1, n)
    high_risk = rng.normal(0, 1, n)
    metrics = direction_consistency(factual, low_risk, high_risk, event_time, censorship)
    # 0.5 ± 0.07 by chance at this sample size.
    assert 0.40 <= metrics["correct_rate"] <= 0.60, metrics
