"""DCT v3.10: the frozen paper-facing DCT objective.

The survival NLL is supplied by the shared trainer.  The paper objective
contributes one differentiable auxiliary term:

    0.10 * IPCW pairwise ranking.

An opt-in environment variable ``SURVOT_V310_DIR_WEIGHT`` can re-enable a
non-zero direction coefficient for sensitivity sweeps.  When that variable is
unset (or zero), the model behaves exactly as the frozen paper recipe.

All historical DCT auxiliary objectives remain available through their original
registered versions, but cannot be re-enabled in this final class by a config
or CLI override.
"""

from __future__ import annotations

import os

from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)


class DCTV310DirectionalRegularizedTransport(
    DCTTransportInterventionConsistency
):
    """Frozen DCT recipe: NLL + 0.10 IPCW-rank.

    DCT v3.10 uses IPCW pairwise ranking as its only auxiliary objective.
    A non-zero direction coefficient can be opted into for sensitivity sweeps
    via the SURVOT_V310_DIR_WEIGHT environment variable; see the module
    docstring for the contract.
    """

    NLL_WEIGHT = 1.0
    IPCW_RANK_WEIGHT = 0.10
    # Optional escape hatch for sensitivity tests. The launcher can opt in by
    # exporting SURVOT_V310_DIR_WEIGHT before spawning the trainer; otherwise
    # the paper objective stays at zero and this attribute is unused.
    DIRECTION_WEIGHT = float(os.environ.get("SURVOT_V310_DIR_WEIGHT", "0.0"))
    FROZEN_ARGUMENTS = {
        "dct_lambda_ipcw_rank": IPCW_RANK_WEIGHT,
        "dct_ipcw_rank_margin": 0.02,
        "dct_ipcw_rank_temperature": 0.50,
        "dct_ipcw_max_weight": 10.0,
        "dct_ipcw_rank_memory_size": 64,
        "dct_lambda_etar": 0.0,
        "dct_lambda_listwise": 0.0,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
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
        """Combine the frozen IPCW term with the optional direction term.

        The direction term is included only when ``SURVOT_V310_DIR_WEIGHT`` is
        set to a non-zero value (sensitivity sweeps).  When the variable is
        zero the parent already produces ``transport_objective == 0`` and the
        return value collapses to ``IPCW_RANK_WEIGHT * ipcw_rank_loss``.
        Keeping this combiner local prevents a future parent-class ETAR change
        from altering the DCT v3.10 objective.
        """

        del etar_loss, transport_metrics, epoch
        return self.IPCW_RANK_WEIGHT * ipcw_rank_loss + transport_objective
