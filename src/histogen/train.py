"""Training loop: train on the training sites, select on their validation split,
report on the held-out site every epoch."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import fbeta_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

MetricsLogger = Callable[[str, dict[str, float], int], None]


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    auc: float
    f_beta5: float


def compute_metrics(targets: np.ndarray, probs: np.ndarray, loss_sum: float) -> EpochResult:
    """Dataset-level metrics for binary classification; `probs` is (N, 2)."""
    preds = probs.argmax(axis=1)
    n = len(targets)
    # AUC is undefined when only one class is present (e.g. a tiny validation split).
    auc = roc_auc_score(targets, probs[:, 1]) if len(np.unique(targets)) > 1 else float("nan")
    return EpochResult(
        loss=loss_sum / n,
        accuracy=float((preds == targets).mean()),
        auc=float(auc),
        # beta=5 weights recall 5x over precision, as reported in the paper.
        f_beta5=float(fbeta_score(targets, preds, beta=5, average="weighted", zero_division=0)),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    """One pass over `loader`; trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    loss_sum, all_targets, all_probs = 0.0, [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)  # CrossEntropyLoss takes raw logits
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * inputs.size(0)
        all_targets.append(labels.cpu().numpy())
        all_probs.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

    return compute_metrics(np.concatenate(all_targets), np.concatenate(all_probs), loss_sum)


def fit(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    log_metrics: MetricsLogger | None = None,
) -> tuple[nn.Module, list[dict[str, EpochResult]]]:
    """Train for `epochs` and return the model with the best `val_in` accuracy."""
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, copy.deepcopy(model.state_dict())
    history: list[dict[str, EpochResult]] = []

    for epoch in range(epochs):
        results = {"train": run_epoch(model, loaders["train"], criterion, device, optimizer)}
        scheduler.step()
        for phase in ("val_in", "test_out"):
            results[phase] = run_epoch(model, loaders[phase], criterion, device)
        history.append(results)

        for phase, r in results.items():
            log.info("epoch %d %-8s loss %.4f acc %.4f auc %.4f", epoch, phase, r.loss, r.accuracy, r.auc)
            if log_metrics:
                log_metrics(phase, vars(r), epoch)

        if results["val_in"].accuracy > best_acc:
            best_acc = results["val_in"].accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    log.info("best val_in accuracy %.4f", best_acc)
    return model, history
