"""DCT v3.8.2 Monotone Dose-Response: smallest recipe that answers the core scientific claim.

This method keeps the v3.8 factual path and IPCW ranking, but removes every
component that has been independently ablated as neutral or harmful:

* MGPTR (λ=0): the v3.8.2 single-variable ablation showed Δ=−0.0031 (noise).
* adaptive auxiliary weighting: full adaptive recipe loses 0.0087 to fixed.
* dose / reconfiguration losses: v3.8 2³ factorial ablation showed they add
  nothing on top of the direction loss and can regress the recipe.
* ETAR, geometry reliability, evidence-cost: already disabled in v3.8.2.

The only structural objective retained is the direction loss. Its purpose is
to prove that re-optimised Sinkhorn under high / low risk anchors moves the
predicted survival risk in the requested direction, which is the monotone
dose-response claim that other survival-transport methods (MCAT, MOTCat,
MMP) do not state.

No new parameters, no new Sinkhorn solve and no new training head are added
beyond what the v3.8 base class already provides.
"""

from __future__ import annotations

from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)


class DCTV382MonotoneDoseResponse(DCTTransportInterventionConsistency):
    """Frozen monotone dose-response recipe: IPCW rank + direction only.

    All optional structural losses are forced to zero in ``__init__`` so that
    the class cannot accidentally re-enable a harmful component.  Direction
    loss remains a CLI-tunable coefficient so future ablation launches can
    confirm it is the load-bearing term.

    Scientific claim: re-optimised Sinkhorn under high / low risk anchors moves
    the predicted survival risk in the requested monotone direction.  This
    property holds for the factual OT coupling; it is not a causal claim
    about treatment effects.
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_v38_lambda_dose = 0.0
        self.dct_v38_lambda_reconfiguration = 0.0
        self.dct_v382_lambda_mgptr = 0.0
        self.dct_v382_adaptive_aux_weights = False
        if getattr(self, "dct_v38_lambda_direction", 0.0) < 0.0:
            raise ValueError(
                "DCT monotone dose-response recipe requires dct_v38_lambda_direction >= 0"
            )
