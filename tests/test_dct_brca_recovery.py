from pathlib import Path

from scripts.run_dct_brca_recovery import BASE, VARIANTS, build_command, parse_folds, parse_variants


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_brca_recovery_causal_ladder_is_fold_and_result_isolated():
    assert parse_variants("ref,det,bin,strat") == ["ref", "det", "bin", "strat"]
    assert parse_folds("0,2") == [0, 2]
    assert BASE["event_stratified_batches"] is False
    assert VARIANTS["ref"]["dct_slot_init_mode"] == "gaussian"
    assert VARIANTS["det"]["dct_slot_init_mode"] == "deterministic"
    assert VARIANTS["bin"]["fit_bins_on_train"] is True
    assert VARIANTS["strat"]["event_stratified_batches"] is True

    command, result_dir = build_command("python3", "strat", 2, "0", "4", smoke=False)
    joined = " ".join(command)
    assert (REPO_ROOT / "configs/distributional_counterfactual_transport_brca.yaml").exists()
    assert "distributional_counterfactual_transport_brca.yaml" in joined
    assert "k_start=2" in joined and "k_end=3" in joined
    assert "fit_bins_on_train=true" in joined
    assert "event_stratified_batches=true" in joined
    assert result_dir.as_posix() == "results/dct_brca_recovery/strat"
    assert "blca" not in joined.lower()
    assert "luad" not in joined.lower()


def test_brca_recovery_candidates_do_not_mutate_the_shared_base():
    assert BASE["alpha_surv"] == 0.15
    assert BASE["dct_lambda_ipcw_rank"] == 0.10
    assert BASE["lr"] == 0.0005
    assert VARIANTS["a30"]["alpha_surv"] == 0.30
    assert VARIANTS["norank"]["dct_lambda_ipcw_rank"] == 0.0
    assert VARIANTS["reg"]["lr"] == 0.0002
