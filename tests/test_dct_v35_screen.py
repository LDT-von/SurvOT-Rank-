from pathlib import Path

from scripts.run_dct_v35_screen import (
    CANCERS,
    COMMON_OVERRIDES,
    VARIANTS,
    build_train_command,
    parse_cancers,
    parse_folds,
    parse_variants,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_v35_screen_covers_all_requested_cancers_and_four_single_change_variants():
    assert len(CANCERS) == 10
    assert parse_cancers("all") == list(CANCERS)
    assert parse_variants("r,q,g,l") == ["r", "q", "g", "l"]
    assert parse_folds("0,2") == [0, 2]
    assert set(VARIANTS) == {"r", "q", "g", "l"}
    assert COMMON_OVERRIDES["batch_size"] == 8
    assert COMMON_OVERRIDES["fit_bins_on_train"] is True
    assert COMMON_OVERRIDES["event_sampling_fraction"] == 0.0
    assert COMMON_OVERRIDES["event_stratified_batches"] is True
    assert COMMON_OVERRIDES["dct_ipcw_rank_memory_size"] == 0


def test_v35_command_is_fold_isolated_and_uses_existing_cancer_config():
    command, result_dir = build_train_command(
        "python3", "brca", "q", 2, "0", "4"
    )
    joined = " ".join(command)
    assert (REPO_ROOT / "configs/distributional_counterfactual_transport_brca.yaml").exists()
    assert "k_start=2" in joined
    assert "k_end=3" in joined
    assert "dct_slot_init_mode=learned" in joined
    assert "event_stratified_batches=true" in joined
    assert result_dir.as_posix() == "results/dct_v3.5_screen/q/brca"


def test_v35_variants_change_only_the_declared_architecture_controls():
    assert VARIANTS["r"]["dct_slot_init_mode"] == "deterministic"
    assert VARIANTS["q"]["dct_slot_init_mode"] == "learned"
    assert VARIANTS["g"]["dct_evidence_marginal_strength"] == 0.25
    assert VARIANTS["l"]["wsi_projection_dim"] == 128
    assert VARIANTS["l"]["otehv2_layers"] == 1
