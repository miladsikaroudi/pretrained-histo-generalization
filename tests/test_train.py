import math

import numpy as np
import pytest
import torch

from histogen.cli import main
from histogen.train import compute_metrics


def test_metrics_on_known_predictions():
    targets = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    r = compute_metrics(targets, probs, loss_sum=2.0)
    assert r.loss == 0.5
    assert r.accuracy == 0.75
    assert r.auc == 1.0


def test_auc_is_nan_with_one_class():
    r = compute_metrics(np.array([1, 1]), np.array([[0.2, 0.8], [0.6, 0.4]]), 0.0)
    assert math.isnan(r.auc)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cli_trains_end_to_end_on_cpu(config_file):
    out = main(["--config", str(config_file)])
    saved = torch.load(out, weights_only=False)
    assert saved["config"]["holdout_site"] == "c"
    assert "fc.weight" in saved["state_dict"]
