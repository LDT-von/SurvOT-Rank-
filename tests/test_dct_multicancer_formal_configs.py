from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_dct_multicancer_formal import CONFIGS, parse_cancers


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_formal_launcher_covers_the_three_reviewed_cancers():
    assert parse_cancers("brca,luad,lusc") == ["brca", "luad", "lusc"]
    assert set(CONFIGS) == {"brca", "luad", "lusc"}


def test_formal_configs_are_train_only_and_have_distinct_result_dirs():
    result_dirs = set()
    for cancer, relative_path in CONFIGS.items():
        with open(REPO_ROOT / relative_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["data"]["study"] == cancer
        assert config["train"]["fit_bins_on_train"] is True
        assert config["train"]["results_dir"] not in result_dirs
        result_dirs.add(config["train"]["results_dir"])
