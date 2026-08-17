"""Single source of truth for public SurvOT-Rank method identifiers.

The catalog separates two questions that were previously mixed together:

* ``key`` / ``aliases`` / ``method_dir`` describe how executable code is loaded.
* ``status`` describes the current research role as of ``CATALOG_UPDATED``.

``primary`` does not mean that every experiment is complete, and ``reference``
does not mean that a method is deleted.  All entries remain runnable so old
experiments can be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


CATALOG_UPDATED = "2026-08-17"
PRIMARY_METHOD = "dct_v310_directional_regularized_transport"
METHOD_STATUSES = ("primary", "candidate", "repair", "research", "reference")
STATUS_LABELS = {
    "primary": "current paper mainline",
    "candidate": "formal evaluation candidate",
    "repair": "active repair gate",
    "research": "research branch",
    "reference": "baseline / historical reference",
}


def _method_dir(folder: str) -> str:
    return os.path.join("survot_rank", "research", "methods", folder)


@dataclass(frozen=True)
class MethodSpec:
    """Executable method registration plus a compact research-role label."""

    key: str
    display_name: str
    family: str
    status: str
    folder: str
    class_name: str
    aliases: tuple[str, ...] = ()
    model_file: str = "model.py"

    @property
    def method_dir(self) -> str:
        return _method_dir(self.folder)


# Keep this order stable for backward-compatible argparse help and error output.
METHOD_SPECS = (
    MethodSpec(
        "ot_event_hazard_v2",
        "V31 OT Event Hazard",
        "ot-event",
        "reference",
        "ot_event_hazard_v2",
        "OTEventHazardV2Survival",
        aliases=("31",),
        model_file="model_v2.py",
    ),
    MethodSpec(
        "otehv2_rankevent",
        "V45 Prognostic Event Transport",
        "ot-event",
        "reference",
        "prognostic_event_transport",
        "OTEHV2RankEvent",
        aliases=("45", "pet", "prognostic_event_transport"),
    ),
    MethodSpec(
        "otehv2_rankevent_v2",
        "V45v2 Prognostic Event Transport",
        "ot-event",
        "reference",
        "prognostic_event_transport",
        "OTEHV2RankEventV2",
        aliases=("45v2",),
    ),
    MethodSpec(
        "otehv2_timelocal_competing",
        "V50 Time-Local Competing Risk",
        "ot-event",
        "reference",
        "prognostic_event_transport",
        "OTEHTimeLocalCompeting",
        aliases=("50",),
    ),
    MethodSpec(
        "rank_guided_event_transport",
        "Rank-Guided Event Transport",
        "evidence-transport",
        "research",
        "rank_guided_event_transport",
        "RankGuidedEventTransport",
    ),
    MethodSpec(
        "stagewise_prognostic_transport",
        "Stagewise Prognostic Transport",
        "evidence-transport",
        "research",
        "stagewise_prognostic_transport",
        "StagewisePrognosticTransport",
    ),
    MethodSpec(
        "faithful_evidence_transport",
        "Faithful Evidence Transport",
        "evidence-transport",
        "research",
        "faithful_evidence_transport",
        "FaithfulEvidenceTransport",
    ),
    MethodSpec(
        "distributional_counterfactual_transport",
        "DCT Score-First",
        "dct",
        "research",
        "distributional_counterfactual_transport",
        "DistributionalCounterfactualTransport",
    ),
    MethodSpec(
        "dct_v41_survival_evidence_ledger",
        "DCT v4.1 Survival Evidence Ledger",
        "dct",
        "research",
        "dct_v41_survival_evidence_ledger",
        "DCTV41SurvivalEvidenceLedger",
        aliases=("dct_v41", "dct_v4_1"),
    ),
    MethodSpec(
        "dct_listwise_transport",
        "DCT v3.6 Listwise Transport",
        "dct",
        "research",
        "dct_listwise_transport",
        "DCTListwiseTransport",
    ),
    MethodSpec(
        "dct_transport_intervention_consistency",
        "DCT v3.8 Transport Intervention Consistency",
        "dct",
        "research",
        "dct_transport_intervention_consistency",
        "DCTTransportInterventionConsistency",
        aliases=("dct_v38",),
    ),
    MethodSpec(
        "dct_v382_prognostic_transport_reconstruction",
        "DCT v3.8.2 Prognostic Transport Reconstruction",
        "dct",
        "reference",
        "dct_v382_prognostic_transport_reconstruction",
        "DCTV382PrognosticTransportReconstruction",
        aliases=("dct_v382", "dct_v3_8_2"),
    ),
    MethodSpec(
        "dct_v382_minimal_transport",
        "DCT v3.8.2 Monotone Dose-Response (minimal)",
        "dct",
        "reference",
        "dct_v382_minimal_transport",
        "DCTV382MonotoneDoseResponse",
        aliases=("dct_v382_minimal", "dct_minimal", "dct_v3_8_2_minimal", "dct_monotone"),
    ),
    MethodSpec(
        "dct_v310_directional_regularized_transport",
        "DCT v3.10 Directionally Regularized Transport (DCT-Reg)",
        "dct",
        "primary",
        "dct_v310_directional_regularized_transport",
        "DCTV310DirectionalRegularizedTransport",
        aliases=("dct_v310", "dct_v3_10", "dct_reg", "dct_directional_reg"),
    ),
    MethodSpec(
        "dct_v383_intervention_consistency_centered",
        "DCT v3.8.3 Centered Intervention Consistency",
        "dct",
        "research",
        "dct_v383_intervention_consistency_centered",
        "DCTV383InterventionConsistencyCentered",
        aliases=("dct_v383", "dct_v3_8_3"),
    ),
    MethodSpec(
        "dct_v39_risk_simplex_transport",
        "DCT v3.9 Risk-Simplex Transport",
        "dct",
        "research",
        "dct_v39_risk_simplex_transport",
        "DCTV39RiskSimplexTransport",
        aliases=("dct_v39", "dct_v3_9", "rst", "risk_simplex_transport"),
    ),
    MethodSpec(
        "intervention_stable_survival_transport",
        "IST-Surv v4.0",
        "intervention",
        "repair",
        "intervention_stable_survival_transport",
        "InterventionStableSurvivalTransport",
        aliases=("v40", "ist_surv"),
    ),
    MethodSpec(
        "censoring_aware_temporal_evidence_transport",
        "CATET",
        "evidence-transport",
        "candidate",
        "censoring_aware_temporal_evidence_transport",
        "CensoringAwareTemporalEvidenceTransport",
    ),
    MethodSpec(
        "v60_ot_event_rank",
        "V60 OT Event Rank",
        "ot-event",
        "reference",
        "v60_ot_event_rank",
        "V60OTEventRank",
        aliases=("60",),
    ),
    MethodSpec(
        "archetypal_transport_composition",
        "ACT-Surv v4.2",
        "archetype",
        "research",
        "archetypal_transport_composition",
        "ArchetypalTransportComposition",
        aliases=("act_surv", "actsurv", "v42", "dct_v42"),
    ),
    MethodSpec(
        "archetypal_transport_composition_v5",
        "ACT-Surv v5",
        "archetype",
        "candidate",
        "archetypal_transport_composition_v5",
        "ArchetypalTransportCompositionV5",
        aliases=("act_surv_v5", "actv5", "v5"),
    ),
    MethodSpec(
        "archetypal_risk_composition",
        "ArcSurv",
        "archetype",
        "candidate",
        "archetypal_risk_composition",
        "ArchetypalRiskComposition",
        aliases=("arcsurv", "arc_surv"),
    ),
    MethodSpec(
        "cohort_anchored_adaptive_prognostic_slot_attention",
        "CA-PSA",
        "archetype",
        "candidate",
        "cohort_anchored_adaptive_prognostic_slot_attention",
        "CohortAnchoredAdaptivePrognosticSlotAttention",
        aliases=("ca_psa", "capsa"),
    ),
    MethodSpec(
        "v70_patient_specific_prognostic_circuits",
        "V70 PSPC-Surv",
        "circuit",
        "research",
        "v70_patient_specific_prognostic_circuits",
        "V70PatientSpecificPrognosticCircuits",
        aliases=("70", "pspc_surv", "pspc"),
    ),
)


METHOD_CATALOG = {spec.key: spec for spec in METHOD_SPECS}
if len(METHOD_CATALOG) != len(METHOD_SPECS):
    raise RuntimeError("Duplicate canonical method key in METHOD_SPECS")
_invalid_statuses = sorted({spec.status for spec in METHOD_SPECS} - set(METHOD_STATUSES))
if _invalid_statuses:
    raise RuntimeError(f"Unknown method statuses in METHOD_SPECS: {_invalid_statuses}")
_primary_keys = [spec.key for spec in METHOD_SPECS if spec.status == "primary"]
if _primary_keys != [PRIMARY_METHOD]:
    raise RuntimeError(
        f"Expected one primary method ({PRIMARY_METHOD}), found {_primary_keys}"
    )

METHOD_REGISTRY = {
    spec.key: (spec.method_dir, spec.class_name)
    for spec in METHOD_SPECS
}

METHOD_ALIASES: dict[str, str] = {}
for _spec in METHOD_SPECS:
    for _alias in _spec.aliases:
        if _alias in METHOD_CATALOG or _alias in METHOD_ALIASES:
            raise RuntimeError(f"Duplicate or ambiguous method alias: {_alias}")
        METHOD_ALIASES[_alias] = _spec.key

METHOD_CHOICES = tuple(METHOD_CATALOG) + tuple(METHOD_ALIASES)


def iter_method_specs(status: str | None = None) -> Iterable[MethodSpec]:
    """Yield catalog entries, optionally filtered by current research status."""

    if status is not None and status not in METHOD_STATUSES:
        raise ValueError(f"Unknown method status: {status}")
    return (spec for spec in METHOD_SPECS if status is None or spec.status == status)


def catalog_errors(project_root: str | Path) -> list[str]:
    """Return local catalog/code mismatches without importing model modules."""

    root = Path(project_root)
    errors: list[str] = []
    for spec in METHOD_SPECS:
        model_path = root / spec.method_dir / spec.model_file
        if not model_path.is_file():
            errors.append(f"{spec.key}: missing {model_path.relative_to(root)}")
    return errors
