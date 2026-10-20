import numpy as np
import torch

from histogen.transforms import HEDJitter, RandomGaussBlur, eval_transform, train_transform


def test_hed_jitter_draws_new_parameters_per_image(h_and_e_patch):
    jitter = HEDJitter(theta=0.1)
    np.random.seed(0)
    first = np.asarray(jitter(h_and_e_patch))
    second = np.asarray(jitter(h_and_e_patch))
    assert first.shape == (32, 32, 3) and first.dtype == np.uint8
    assert not np.array_equal(first, second)


def test_hed_jitter_theta_zero_keeps_stain_structure(h_and_e_patch):
    out = np.asarray(HEDJitter(theta=0.0)(h_and_e_patch)).astype(int)
    left, right = out[:, :16].mean(axis=(0, 1)), out[:, 16:].mean(axis=(0, 1))
    assert left.sum() > right.sum()  # eosin half stays lighter than haematoxylin half


def test_gauss_blur_radius_varies(h_and_e_patch):
    blur = RandomGaussBlur(radius=(0.5, 1.5))
    import random

    random.seed(1)
    outs = {np.asarray(blur(h_and_e_patch)).tobytes() for _ in range(5)}
    assert len(outs) > 1


def test_eval_transform_normalises_to_tensor(h_and_e_patch):
    x = eval_transform()(h_and_e_patch)
    assert isinstance(x, torch.Tensor) and x.shape == (3, 32, 32)
    assert x.min() < 0 < x.max()


def test_train_transform_without_augmentation_is_eval(h_and_e_patch):
    assert torch.equal(train_transform(False)(h_and_e_patch), eval_transform()(h_and_e_patch))
