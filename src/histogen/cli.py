"""Command-line entry point: `histogen-train --config configs/camelyon17.json`."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import make_dataloaders
from .models import build_model, freeze_leading_children
from .train import fit


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/camelyon17.json", help="JSON experiment config")
    p.add_argument("--holdout-site", help="override the held-out site")
    p.add_argument("--model", help="override the backbone")
    p.add_argument("--epochs", type=int, help="override the number of epochs")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                   help='override any config key, e.g. --set batch_size=32 --set pretrained=false')
    return p.parse_args(argv)


def load_config(args: argparse.Namespace) -> Config:
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    for key in ("holdout_site", "model", "epochs"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    return Config.from_json(args.config, overrides)


def main(argv: list[str] | None = None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(parse_args(argv))
    seed_everything(cfg.seed)
    device = pick_device()

    log_metrics = None
    if cfg.wandb_project:
        import wandb  # optional; the entity comes from WANDB_ENTITY or the logged-in default

        run = wandb.init(project=cfg.wandb_project, name=cfg.holdout_site, config=cfg.to_dict())
        log_metrics = lambda phase, metrics, epoch: run.log(  # noqa: E731
            {f"{phase}/{k}": v for k, v in metrics.items()}, step=epoch)

    model = build_model(cfg.model, cfg.num_classes, cfg.pretrained, cfg.kimianet_weight_path)
    model = freeze_leading_children(model, cfg.frozen_children).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    model, _ = fit(model, make_dataloaders(cfg), optimizer, scheduler, device, cfg.epochs, log_metrics)

    out = Path(cfg.output_dir) / (
        f"holdout-{cfg.holdout_site}_{cfg.model}_lr{cfg.learning_rate}_ep{cfg.epochs}"
        f"_bs{cfg.batch_size}_aug{int(cfg.augmentation_in_training)}.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, out)
    logging.info("saved %s", out)
    return out


if __name__ == "__main__":
    main()
