import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def h_and_e_patch() -> Image.Image:
    """A 32x32 patch with pink (eosin-like) and purple (haematoxylin-like) halves."""
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :16] = (230, 150, 190)
    rgb[:, 16:] = (110, 60, 160)
    return Image.fromarray(rgb)


@pytest.fixture
def site_dir(tmp_path):
    """Three sites of 12 synthetic patches each, with per-site CSVs like the real data."""
    rng = np.random.default_rng(0)
    for site in ("a", "b", "c"):
        rows = []
        for i in range(12):
            label = i % 2
            pixels = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            pixels[..., 0] = 200 if label else 60  # make the classes separable
            path = tmp_path / f"{site}_{i}.png"
            Image.fromarray(pixels).save(path)
            rows.append({"file_paths": str(path), "class_label": label})
        pd.DataFrame(rows).to_csv(tmp_path / f"{site}.csv", index=False)
    return tmp_path


@pytest.fixture
def config_file(site_dir):
    path = site_dir / "config.json"
    path.write_text(json.dumps({
        "dataframe_root": str(site_dir),
        "sites": ["a", "b", "c"],
        "holdout_site": "c",
        "model": "resnet18",
        "pretrained": False,
        "epochs": 2,
        "batch_size": 4,
        "num_workers": 0,
        "augmentation_in_training": True,
        "output_dir": str(site_dir / "out"),
    }))
    return path
