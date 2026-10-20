"""Experiment configuration, loaded from JSON with optional CLI overrides."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # Data: one CSV per site, named <site>.csv, with columns `file_paths` and `class_label`.
    dataframe_root: str
    sites: list[str]
    holdout_site: str
    train_val_portions: tuple[int, int] = (70, 10)

    # Model
    model: str = "resnet18"
    pretrained: bool = True
    kimianet_weight_path: str | None = None
    num_classes: int = 2
    frozen_children: int = 1  # leading child modules to freeze; 1 = ResNet conv1

    # Optimisation
    learning_rate: float = 1e-4
    momentum: float = 0.9
    epochs: int = 50
    batch_size: int = 64
    lr_step_size: int = 7
    lr_gamma: float = 0.1
    augmentation_in_training: bool = False

    # Runtime
    seed: int = 0
    num_workers: int = 8
    output_dir: str = "."
    wandb_project: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.holdout_site not in self.sites:
            raise ValueError(f"holdout_site {self.holdout_site!r} is not one of sites {self.sites}")
        train, val = self.train_val_portions
        if train <= 0 or val < 0 or train + val > 100:
            raise ValueError(f"train_val_portions {self.train_val_portions} must be percentages summing to <= 100")
        self.train_val_portions = (train, val)

    @property
    def training_sites(self) -> list[str]:
        return [s for s in self.sites if s != self.holdout_site]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = dict(raw)
        # Keys used by the 2023 config.json.
        renames = {"trail_sites": "sites", "holdout_trial_site": "holdout_site"}
        for old, new in renames.items():
            if old in raw:
                raw.setdefault(new, raw.pop(old))
        raw.pop("font_path", None)  # only used by removed debugging code

        known = {f.name for f in fields(cls)}
        extra = {k: raw.pop(k) for k in list(raw) if k not in known}
        return cls(**raw, extra=extra)

    @classmethod
    def from_json(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
        raw = json.loads(Path(path).read_text())
        raw.update(overrides or {})
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
