import pandas as pd

from scripts.run_dct_v36_listwise_screen import (
    VARIANTS,
    build_train_command,
    parse_cancers,
    parse_folds,
)
from scripts.summarize_dct_v36_listwise import _promotion_gate


def test_v36_runner_defaults_and_variant_isolation():
    assert parse_cancers("blca,brca") == ["blca", "brca"]
    assert parse_folds("0,2") == [0, 2]
    assert set(VARIANTS) == {
        "nll",
        "ipcw",
        "etar",
        "ipcw_etar",
        "gpl",
        "tcl",
    }
    command, result_dir = build_train_command(
        "python",
        "blca",
        "tcl",
        2,
        "0",
        "4",
    )
    rendered = " ".join(command)
    assert "survot_method=dct_listwise_transport" in rendered
    assert "dct_listwise_mode=stage_transport" in rendered
    assert "dct_lambda_ipcw_rank=0.0" in rendered
    assert "dct_lambda_etar=0.0" in rendered
    assert result_dir.as_posix() == "results/dct_v3.6_listwise/tcl/blca"


def test_v36_smoke_is_one_epoch_and_uses_separate_result_root():
    command, result_dir = build_train_command(
        "python",
        "brca",
        "gpl",
        0,
        "0",
        "0",
        smoke=True,
    )
    rendered = " ".join(command)
    assert "max_epochs=1" in rendered
    assert "max_smoke_batches=1" in rendered
    assert result_dir.as_posix() == "results/dct_v3.6_listwise_smoke/gpl/brca"


def test_v36_promotion_gate_requires_transport_specific_and_stable_gain():
    records = []
    for cancer in ("blca", "brca"):
        for fold in (0, 2):
            records.extend(
                [
                    {
                        "variant": "ipcw",
                        "cancer": cancer,
                        "fold": fold,
                        "status": "ok",
                        "best_cindex": 0.70,
                        "last5_cindex": 0.64,
                        "best_last_gap": 0.06,
                    },
                    {
                        "variant": "gpl",
                        "cancer": cancer,
                        "fold": fold,
                        "status": "ok",
                        "best_cindex": 0.695,
                        "last5_cindex": 0.645,
                        "best_last_gap": 0.05,
                    },
                    {
                        "variant": "tcl",
                        "cancer": cancer,
                        "fold": fold,
                        "status": "ok",
                        "best_cindex": 0.71,
                        "last5_cindex": 0.67,
                        "best_last_gap": 0.04,
                    },
                ]
            )
    decision = _promotion_gate(pd.DataFrame(records), ["blca", "brca"], [0, 2])
    assert decision["promote"] is True
    assert decision["criteria"]["tcl_fold_wins_over_gpl"] == 4
