from collections import OrderedDict

import pytest
import torch
from torch import nn
from torchvision import models

from histogen.models import (
    ModelName,
    build_model,
    freeze_leading_children,
    kimianet_to_densenet_keys,
)


@pytest.mark.parametrize("name", ["resnet18", "resnext50_32x4d"])
def test_every_torchvision_name_builds_with_new_head(name):
    model = build_model(name, num_classes=2, pretrained=False)
    assert model.fc.out_features == 2
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2)


def test_config_strings_round_trip():
    assert {m.value for m in ModelName} >= {"resnext50_32x4d", "resnext101_32x8d", "resnet18_ssl"}


def _kimianet_checkpoint(tmp_path):
    """A checkpoint laid out like the published KimiaNet weights:
    nn.DataParallel(fully_connected(model=densenet.features+pool, fc_4=Linear(1024, 30)))."""
    features = nn.Sequential(models.densenet121(weights=None).features, nn.AdaptiveAvgPool2d((1, 1)))
    with torch.no_grad():
        for p in features.parameters():
            p.fill_(0.5)
    state = OrderedDict(("module.model." + k, v) for k, v in features.state_dict().items())
    state["module.fc_4.weight"] = torch.zeros(30, 1024)
    state["module.fc_4.bias"] = torch.zeros(30)
    path = tmp_path / "KimiaNetPyTorchWeights.pth"
    torch.save(state, path)
    return path


def test_kimianet_checkpoint_keys_are_remapped():
    mapped = kimianet_to_densenet_keys({"module.model.0.conv0.weight": 1, "module.fc_4.weight": 2})
    assert mapped == OrderedDict({"features.0.conv0.weight": 1})


def test_kimianet_weights_actually_load(tmp_path):
    model = build_model("kimianet", num_classes=2, kimianet_weight_path=_kimianet_checkpoint(tmp_path))
    assert torch.all(model.features[0].conv0.weight == 0.5)
    assert model.classifier.out_features == 2


def test_kimianet_rejects_checkpoint_that_does_not_match(tmp_path):
    path = tmp_path / "wrong.pth"
    torch.save({"something.else": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="KimiaNet feature tensors"):
        build_model("kimianet", num_classes=2, kimianet_weight_path=path)


def test_freeze_leading_children():
    model = freeze_leading_children(build_model("resnet18", 2, pretrained=False), 1)
    assert not model.conv1.weight.requires_grad
    assert model.bn1.weight.requires_grad
