import json

import pytest

from histogen.config import Config


def test_reads_2023_key_names(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "dataframe_root": "/data/",
        "trail_sites": ["center_0", "center_1"],
        "holdout_trial_site": "center_1",
        "font_path": "/unused.pfb",
        "train_val_portions": [70, 10],
    }))
    cfg = Config.from_json(path)
    assert cfg.sites == ["center_0", "center_1"]
    assert cfg.holdout_site == "center_1"
    assert cfg.training_sites == ["center_0"]
    assert cfg.extra == {}


def test_overrides_win_over_file(config_file):
    cfg = Config.from_json(config_file, {"batch_size": 2, "holdout_site": "a"})
    assert cfg.batch_size == 2
    assert cfg.training_sites == ["b", "c"]


def test_rejects_unknown_holdout(config_file):
    with pytest.raises(ValueError, match="holdout_site"):
        Config.from_json(config_file, {"holdout_site": "z"})


def test_rejects_portions_over_100(config_file):
    with pytest.raises(ValueError, match="train_val_portions"):
        Config.from_json(config_file, {"train_val_portions": [90, 20]})
