"""Leave-one-site-out data loading for patch-level histopathology datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .config import Config
from .transforms import eval_transform, train_transform

PATH_COLUMN = "file_paths"
LABEL_COLUMN = "class_label"


class PatchDataset(Dataset):
    """Image patches listed in a DataFrame with `file_paths` and `class_label` columns."""

    def __init__(self, frame: pd.DataFrame, transform: Callable | None = None) -> None:
        missing = {PATH_COLUMN, LABEL_COLUMN} - set(frame.columns)
        if missing:
            raise ValueError(f"dataframe is missing columns: {sorted(missing)}")
        self.paths = frame[PATH_COLUMN].tolist()
        self.labels = frame[LABEL_COLUMN].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def read_sites(root: str | Path, sites: list[str]) -> pd.DataFrame:
    """Concatenate `<root>/<site>.csv` for each site."""
    frames = [pd.read_csv(Path(root) / f"{site}.csv") for site in sites]
    return pd.concat(frames, ignore_index=True)


def split_indices(n: int, portions: tuple[int, int], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Seeded train/val/test index split; `portions` are train and val percentages."""
    train_pct, val_pct = portions
    n_train = n * train_pct // 100
    n_val = n * val_pct // 100
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def make_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """Loaders for training-site train/val splits and the held-out site.

    Only the training split is augmented; validation and the held-out site use
    the plain normalising transform.
    """
    in_dist = read_sites(cfg.dataframe_root, cfg.training_sites)
    out_dist = read_sites(cfg.dataframe_root, [cfg.holdout_site])
    train_idx, val_idx, _ = split_indices(len(in_dist), cfg.train_val_portions, cfg.seed)

    train_set = Subset(PatchDataset(in_dist, train_transform(cfg.augmentation_in_training)), train_idx)
    val_set = Subset(PatchDataset(in_dist, eval_transform()), val_idx)
    test_set = PatchDataset(out_dist, eval_transform())

    def loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers)

    return {
        "train": loader(train_set, shuffle=True),
        "val_in": loader(val_set, shuffle=False),
        "test_out": loader(test_set, shuffle=False),
    }
