# Risk-Anchored Counterfactual Evidence Transport

`distributional_counterfactual_transport` is an experimental multimodal survival
line for **model-based counterfactual sensitivity analysis**.  It is not a causal
treatment-effect estimator.

## Mechanism

1. Patient-local WSI/pathway representations are first learned by Slot Attention,
   preserving within-patient competition and capacity. Those local slots are
   then competitively aligned to separate global prototype dictionaries.
   Prototype index therefore defines a stable coordinate across patients without
   discarding the factual backbone that produced the earlier DCT scores.
2. At the start of every fold, only that fold's training labels fit time-stage
   upper boundaries and a censoring Kaplan--Meier curve.
3. Each stage builds two IPCW-weighted transport-cost anchors during training:
   observed events within the stage form the high-risk anchor; patients known to
   survive past the stage boundary form the low-risk anchor.  The latter includes
   late-censored patients but excludes patients censored before that boundary.
4. Evidence gates modify the OT marginals by default, so weak semantic slots may
   carry less mass instead of being forced to use a uniform balanced marginal.
   Their additional transport-energy contribution is a separately tested option.
5. To query a patient, factual stage costs are moved toward either anchor and
   Sinkhorn is solved again under the intervened cost.  A numerical projection
   verifies the resulting coupling's evidence-conditioned marginals.
6. Factual, low-anchor, and high-anchor couplings use the same survival decoder.
   Counterfactual risk direction is measured after training; it is never imposed
   as a training target.

```text
WSI patches / pathway tokens
        -> patient-local Slot Attention
        -> competitive global semantic prototype coordinates
        -> stage-specific evidence-conditioned OT cost
        -> IPCW high-event / low-risk-set cost anchors
        -> cost-space intervention
        -> re-optimised Sinkhorn coupling
        -> shared survival decoder and post-hoc sensitivity metrics
```

## Objective

The score-first default has exactly two terms:

\[
\mathcal L = \mathcal L_{NLL} + \lambda_{rank}\mathcal L_{IPCW-rank}.
\]

Let \(r_i=-\sum_k S_i(k)\), exactly the scalar risk used by validation
C-index, and let
\(\mathcal P=\{(i,j):\delta_i=1, T_i<T_j\}\) be the comparable pairs in a
training batch.  The ranking term is

\[
\mathcal L_{IPCW-rank}=
\frac{\sum_{(i,j)\in\mathcal P} w_i\,\tau\,
\operatorname{softplus}((m-(r_i-r_j))/\tau)}
{\sum_{(i,j)\in\mathcal P}w_i},\qquad
w_i=\min(w_{max},\widehat G(T_i)^{-2}).
\]

The censoring Kaplan--Meier estimate \(\widehat G\) is fitted only on the
current fold's training cases.  Squared IPCW matches censoring-adjusted
concordance weighting; clipping prevents one late event from controlling a
small batch.  With the default \(\lambda=0.10\) and \(\tau=0.50\), a tied-risk
pair contributes \(0.10\times0.50\log 2\), the same initial scale as the old
unweighted \(0.05\times\log 2\) term; this changes the statistical target
without introducing an arbitrary jump in auxiliary-loss magnitude.

OT cost, unweighted ranking, anchor contrast, stage-mean hinge, and coordinate
orthogonality remain available only as zero-by-default ablations.  In
particular, minimizing a fully learnable OT cost is not part of the default:
the transport plan is a representation mechanism, not an energy target that
should be driven toward zero.  Risk anchors are updated as detached moving
statistics, and the two counterfactual Sinkhorn solves run only for post-hoc
evaluation rather than wasting training compute.

There is deliberately no `low_risk_counterfactual < factual_risk <
high_risk_counterfactual` loss.

## 2026 follow-up diagnostics: U and M

Two single-variable diagnostics are available without changing the default DCT:

- **U / RTEM** sets `dct_geometry_reliability_strength=1.0`.  Each stage's
  cosine/euclidean/dot costs are mapped to edge distributions; normalized
  Jensen--Shannon agreement tempers how strongly evidence gates may move OT
  marginals away from uniform.  It adds no parameters or auxiliary loss.
- **M / risk-set memory** sets `dct_ipcw_rank_memory_size=64`.  Detached
  risk/time/censoring values from earlier batches in the same epoch enlarge the
  comparable-pair context for batch 8; the memory is cleared every epoch.

They are diagnostic variants, not separate paper methods, and must be tested
individually after the R/Q/G/L baseline screen:

```bash
python3 scripts/run_dct_v35_screen.py plan --variants u,m --cancers brca,blca,luad,ucec --folds 0,2
```

The literature boundary, slimming audit, mathematical definition and adoption
thresholds are in [`../DCT_2026_RESEARCH_AND_SLIMMING.md`](../DCT_2026_RESEARCH_AND_SLIMMING.md).

## Sparse-event BRCA high-score candidate

`configs/distributional_counterfactual_transport_brca_highscore.yaml` is an
experimental DCT v3.4 training recipe.  It does not replace the recorded v3.3
results until a complete five-fold run is available.  The model and score-first
objective are unchanged; only the sparse-event optimisation protocol changes:

- train-only survival bins retain the leakage-safe target construction;
- weighted sampling raises the expected observed-event fraction to 0.25, or
  two events in a batch of eight;
- `alpha_surv = 2/3` approximately balances aggregate event and censoring NLL
  under that sampled mixture because
  `0.25 = 0.75 * (1 - alpha_surv)`;
- IPCW rank memory remains enabled for cross-batch comparable pairs; and
- all folds run the complete 50-epoch horizon because earlier BRCA peaks
  occurred as late as epoch 42.

Historical configs keep `event_sampling_fraction = 0`, so their data order and
sampling distribution are unchanged.  Early stopping now starts accumulating
patience only after its configured warmup rather than spending patience while
the learning rate is still warming up.

## Formal three-cancer rerun

`scripts/run_dct_multicancer_formal.sh` is the single Linux entry point for
the next reproducible BRCA, LUAD, and LUSC runs.  It selects cancer-specific
configs rather than applying one global recipe:

- BRCA uses `distributional_counterfactual_transport_brca_highscore.yaml` for
  sparse-event sampling and IPCW rank memory;
- LUAD uses `distributional_counterfactual_transport_luad_formal.yaml`; and
- LUSC uses `distributional_counterfactual_transport_lusc_formal.yaml`.

All three use train-fold discrete-time bins.  The unified runner also builds
the IPCW/Brier/iAUC reference from that same training fold.  Evaluation times
outside a fold's observable follow-up are excluded with their matching
discrete-survival columns; unavailable metrics are recorded as `NaN` and the
reason is appended to `metric_diagnostics_fold*.log`, never silently replaced
by zero.

## Required validation

Report, per fold and across seeds:

- train-fold stage edges and anchor coverage;
- coupling marginal residuals;
- risk delta and survival-curve difference across `alpha` values;
- directional-consistency and monotonicity-violation rates;
- stability of prototype-coordinate assignments; and
- ablations of NLL-only, unweighted ranking, the former six-loss recipe, and
  re-optimised cost-space intervention.

## Design references

- [MOTCat (ICCV 2023) official implementation](https://github.com/Innse/MOTCat):
  OT is used to construct multimodal co-attention, while its trainer optimises
  the survival loss (plus optional parameter regularisation) rather than adding
  the returned OT distance to the prediction objective.
- [DeepHit (AAAI 2018)](https://doi.org/10.1609/aaai.v32i1.11842): likelihood
  and differentiable risk ranking are complementary prediction targets.
- [Uno et al. (2011)](https://pubmed.ncbi.nlm.nih.gov/21484848/): censoring-
  adjusted concordance motivates the inverse-squared censoring-survival weight.
