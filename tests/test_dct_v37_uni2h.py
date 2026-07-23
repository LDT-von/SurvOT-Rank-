from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from scripts.run_dct_v37_uni2h_screen import (
    CANCERS,
    COMMON_OVERRIDES,
    UNI2H_DIM,
    VARIANTS,
    build_parser,
    build_train_command,
    inspect_feature_directory,
    parse_cancers,
    parse_folds,
    parse_variants,
)
from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
    SurvivalDataset,
)


def test_v37_uni2h_highscore_changes_only_the_wsi_input_protocol():
    parsed_defaults = build_parser().parse_args([])
    assert parsed_defaults.variants == ["highscore"]
    assert parsed_defaults.mode == "plan"
    command, result_dir = build_train_command(
        "python3",
        "coadread",
        "highscore",
        2,
        "0",
        "4",
        "/data1/TCGA-UNI2-h-features",
    )
    rendered = " ".join(command)
    assert len(CANCERS) == 10
    assert parse_cancers("all") == list(CANCERS)
    assert parse_variants("highscore,clean") == ["highscore", "clean"]
    assert parse_folds("0,2") == [0, 2]
    assert "wsi_encoder=uni2-h" in rendered
    assert "encoding_dim=1536" in rendered
    assert "fit_bins_on_train=false" in rendered
    assert "event_stratified_batches=false" in rendered
    assert "dct_slot_init_mode=gaussian" in rendered
    assert "dct_lambda_etar=0.0" in rendered
    assert result_dir.as_posix() == "results/dct_v3.7_uni2h/highscore/coadread"
    assert COMMON_OVERRIDES["dct_lambda_ipcw_rank"] == 0.10


def test_v37_uni2h_clean_is_an_explicit_separate_control():
    command, result_dir = build_train_command(
        "python3",
        "blca",
        "clean",
        0,
        "0",
        "4",
        "/data1/TCGA-UNI2-h-features",
    )
    rendered = " ".join(command)
    assert set(VARIANTS) == {"highscore", "clean"}
    assert "fit_bins_on_train=true" in rendered
    assert "dct_slot_init_mode=deterministic" in rendered
    assert result_dir.as_posix() == "results/dct_v3.7_uni2h/clean/blca"


def test_uni2h_hdf5_loader_accepts_leading_batch_dimension(tmp_path):
    feature_path = tmp_path / "TCGA-TEST-01Z-00-DX1.h5"
    values = np.arange(5 * UNI2H_DIM, dtype=np.float32).reshape(1, 5, UNI2H_DIM)
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=values)

    dataset = object.__new__(SurvivalDataset)
    dataset.wsi_path = str(tmp_path)
    dataset.encoding_dim = UNI2H_DIM
    dataset.dataset_factory = SimpleNamespace(num_patches=4)
    dataset._wsi_feature_index = None

    loaded = dataset.load_wsi("TCGA-TEST-01Z-00-DX1.svs")
    assert loaded.shape == (5, UNI2H_DIM)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, torch.from_numpy(values).squeeze(0))


def test_uni2h_doctor_checks_real_shape(tmp_path):
    directory = tmp_path / "blca" / "uni2-h" / "pt_files"
    directory.mkdir(parents=True)
    with h5py.File(directory / "sample.h5", "w") as handle:
        handle.create_dataset(
            "features", data=np.zeros((1, 3, UNI2H_DIM), dtype=np.float32)
        )
    report = inspect_feature_directory(tmp_path, "blca")
    assert report["ok"] is True
    assert report["count"] == 1
    assert report["shape"] == (1, 3, UNI2H_DIM)


def test_uni2h_loader_rejects_wrong_dimension(tmp_path):
    feature_path = tmp_path / "bad.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((1, 3, 1024), dtype=np.float32))

    dataset = object.__new__(SurvivalDataset)
    dataset.wsi_path = str(tmp_path)
    dataset.encoding_dim = UNI2H_DIM
    dataset.dataset_factory = SimpleNamespace(num_patches=4)
    dataset._wsi_feature_index = None

    try:
        dataset.load_wsi("bad.svs")
    except ValueError as error:
        assert "expected 1536" in str(error)
    else:
        raise AssertionError("wrong UNI2-h dimension was accepted")
