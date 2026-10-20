"""Backbones compared in the paper, each with a fresh classification head."""

from __future__ import annotations

import logging
from collections import OrderedDict
from enum import Enum
from pathlib import Path

import torch
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)


class ModelName(str, Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    RESNET152 = "resnet152"
    RESNEXT50_32X4D = "resnext50_32x4d"
    RESNEXT101_32X8D = "resnext101_32x8d"
    RESNET18_SSL = "resnet18_ssl"
    RESNET18_SWSL = "resnet18_swsl"
    KIMIANET = "kimianet"


_TORCHVISION = {
    ModelName.RESNET18: (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    ModelName.RESNET50: (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    ModelName.RESNET152: (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
    ModelName.RESNEXT50_32X4D: (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    ModelName.RESNEXT101_32X8D: (models.resnext101_32x8d, models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1),
}

# Semi- and semi-weakly supervised ResNets (Yalniz et al. 2019) from torch.hub.
_HUB_REPO = "facebookresearch/semi-supervised-ImageNet1K-models"


def build_model(
    name: str | ModelName,
    num_classes: int,
    pretrained: bool = True,
    kimianet_weight_path: str | Path | None = None,
) -> nn.Module:
    """Build a backbone and replace its final layer with a `num_classes` head.

    KimiaNet is a DenseNet-121 whose features are loaded from
    `kimianet_weight_path`; `pretrained` does not apply to it. The SSL/SWSL
    models always come with their torch.hub weights.
    """
    name = ModelName(name)

    if name is ModelName.KIMIANET:
        if kimianet_weight_path is None:
            raise ValueError("kimianet requires kimianet_weight_path")
        model = models.densenet121(weights=None)
        # KimiaNet pools inside `features`, matching how its weights were trained.
        model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        load_kimianet_features(model, kimianet_weight_path)
        return model

    if name in (ModelName.RESNET18_SSL, ModelName.RESNET18_SWSL):
        model = torch.hub.load(_HUB_REPO, name.value)
    else:
        builder, weights = _TORCHVISION[name]
        model = builder(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def kimianet_to_densenet_keys(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Map KimiaNet checkpoint keys onto `build_model("kimianet")`'s DenseNet-121.

    The published checkpoint wraps the features as `fully_connected.model`
    inside nn.DataParallel, so its keys look like `module.model.0.conv0.weight`.
    The 30-class `fc_4` head is dropped.
    """
    mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        if key.startswith("model."):
            mapped["features." + key.removeprefix("model.")] = value
        elif key.startswith("features."):
            mapped[key] = value
    return mapped


def load_kimianet_features(model: nn.Module, weight_path: str | Path) -> None:
    """Load KimiaNet feature weights, failing loudly if any feature tensor is missing."""
    state = torch.load(weight_path, map_location="cpu")
    features = kimianet_to_densenet_keys(state)
    expected = {k for k in model.state_dict() if k.startswith("features.")}
    missing = expected - features.keys()
    if missing:
        raise RuntimeError(
            f"{weight_path} lacks {len(missing)} of {len(expected)} KimiaNet feature tensors "
            f"(e.g. {sorted(missing)[:2]}); first checkpoint keys: {list(state)[:3]}"
        )
    model.load_state_dict(features, strict=False)
    log.info("Loaded %d KimiaNet feature tensors from %s", len(expected), weight_path)


def freeze_leading_children(model: nn.Module, n: int) -> nn.Module:
    """Freeze the first `n` direct children (for ResNets, n=1 freezes conv1)."""
    for child in list(model.children())[:n]:
        for param in child.parameters():
            param.requires_grad = False
    return model
