"""DCT v3.10: the frozen paper-facing DCT-Reg objective.

The survival NLL is supplied by the shared trainer.  This model returns exactly
two differentiable auxiliary terms:

    0.10 * IPCW pairwise ranking + 0.05 * directional transport response.

All historical DCT auxiliary objectives remain available through their original
registered versions, but cannot be re-enabled in this final class by a config or
CLI override.
"""

from __future__ import annotations

from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)


class DCTV310DirectionalRegularizedTransport(
    DCTTransportInterventionConsistency
):
    """Frozen DCT-Reg recipe: NLL + 0.10 IPCW-rank + 0.05 direction.

    DCT v3.10 actively shapes the risk response to prognostic ground-cost
    interventions.  It does not claim that an unconstrained OT plan naturally
    carries prognostic semantics, and it does not make a causal-treatment claim.
    """

    NLL_WEIGHT = 1.0
    IPCW_RANK_WEIGHT = 0.10
    DIRECTION_WEIGHT = 0.05
    FROZEN_ARGUMENTS = {
        "dct_lambda_ipcw_rank": IPCW_RANK_WEIGHT,
        "dct_ipcw_rank_margin": 0.02,
        "dct_ipcw_rank_temperature": 0.50,
        "dct_ipcw_max_weight": 10.0,
        "dct_ipcw_rank_memory_size": 64,
        "dct_lambda_etar": 0.0,
        "dct_lambda_listwise": 0.0,
        "dct_v38_lambda_direction": DIRECTION_WEIGHT,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
        "dct_v38_direction_margin": 0.02,
        "dct_v38_temperature": 0.05,
        "dct_v38_alpha_mid": 0.50,
        "dct_v38_alpha_full": 1.00,
        "dct_v38_warmup_epochs": 0,
        "dct_v38_ramp_epochs": 0,
        "dct_anchor_momentum": 0.90,
        "dct_evidence_cost_weight": 0.0,
        "dct_evidence_mass_floor": 0.05,
        "dct_evidence_marginal_strength": 1.0,
        "dct_geometry_reliability_strength": 0.0,
        "dct_mix_ratio": 1.0,
        "dct_slot_init_mode": "deterministic",
        "dct_fixed_coupling": False,
        "dct_random_anchors": False,
        "dct_perm_labels_seed": 0,
        "dct_stage_jitter_fraction": 0.0,
        "dct_freeze_source_prototype": "",
        "dct_v382_lambda_mgptr": 0.0,
        "dct_v382_adaptive_aux_weights": False,
    }

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        bag_loss = str(getattr(args, "bag_loss", "nll_surv"))
        if bag_loss != "nll_surv":
            raise ValueError(
                "DCT v3.10 is frozen to bag_loss='nll_surv'; "
                f"received {bag_loss!r}"
            )

        for name, value in self.FROZEN_ARGUMENTS.items():
            setattr(args, name, value)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        # Paper-facing objective invariants.  These assignments happen after
        # parent construction so no YAML/CLI value can silently change the
        # submitted method.
        self.dct_lambda_ipcw_rank = self.IPCW_RANK_WEIGHT
        self.dct_lambda_etar = 0.0
        self.dct_lambda_listwise = 0.0
        self.dct_v38_lambda_direction = self.DIRECTION_WEIGHT
        self.dct_v38_lambda_dose = 0.0
        self.dct_v38_lambda_reconfiguration = 0.0

        # The final coefficient is active without the historical 5+10 epoch
        # warmup/ramp that confounded the early LUSC screen.
        self.dct_v38_warmup_epochs = 0
        self.dct_v38_ramp_epochs = 0

        # Compatibility attributes are explicit zeros even though this class
        # does not inherit the v3.8.2 MGPTR implementation.
        self.dct_v382_lambda_mgptr = 0.0
        self.dct_v382_adaptive_aux_weights = False

    @classmethod
    def objective_weights(cls) -> dict[str, float]:
        """Return the immutable paper objective for manifests and tests."""

        return {
            "nll": cls.NLL_WEIGHT,
            "ipcw_rank": cls.IPCW_RANK_WEIGHT,
            "direction": cls.DIRECTION_WEIGHT,
        }

    def _combine_auxiliary_objectives(
        self,
        *,
        ipcw_rank_loss,
        etar_loss,
        transport_objective,
        transport_metrics,
        epoch,
    ):
        """Combine only the two frozen auxiliary terms.

        ``transport_objective`` is already exactly ``0.05 * direction`` because
        dose and reconfiguration are class invariants at zero.  Keeping this
        combiner local prevents a future parent-class ETAR change from altering
        the DCT v3.10 objective.
        """

        del etar_loss, transport_metrics, epoch
        return self.IPCW_RANK_WEIGHT * ipcw_rank_loss + transport_objective
