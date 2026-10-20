"""Histopathology augmentations.

HEDJitter follows Tellez et al., "Whole-slide mitosis detection in H&E breast
histology using PHH3 as a reference to train distilled stain-invariant
convolutional networks" (IEEE TMI 2018), as packaged by Chaoyang Yan (Apache-2.0) in
https://github.com/gatsby2016/Augmentation-PyTorch-Transforms. Earlier
versions of this repo vendored that whole module; only the two classes used
here are kept, rewritten so their random parameters are drawn per image.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter
from skimage import color
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HEDJitter:
    """Perturb an RGB image in haematoxylin-eosin-DAB stain space.

    Each call draws alpha ~ U[1-theta, 1+theta] and beta ~ U[-theta, theta]
    per stain channel and maps s -> alpha * s + beta, then converts back to RGB
    and rescales to [0, 255].
    """

    def __init__(self, theta: float = 0.0) -> None:
        if theta < 0:
            raise ValueError("theta must be non-negative")
        self.theta = theta

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"))
        alpha = np.random.uniform(1 - self.theta, 1 + self.theta, (1, 3))
        beta = np.random.uniform(-self.theta, self.theta, (1, 3))

        hed = color.rgb2hed(rgb).reshape(-1, 3)
        out = color.hed2rgb((alpha * hed + beta).reshape(rgb.shape))

        lo, hi = out.min(), out.max()
        if hi == lo:
            return Image.fromarray(np.zeros_like(rgb))
        return Image.fromarray((255 * (out - lo) / (hi - lo)).astype(np.uint8))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(theta={self.theta})"


class RandomGaussBlur:
    """Gaussian blur with a radius drawn uniformly from ``radius`` on each call."""

    def __init__(self, radius: tuple[float, float] = (0.5, 1.5)) -> None:
        lo, hi = radius
        if not 0 <= lo <= hi:
            raise ValueError("radius must be (min, max) with 0 <= min <= max")
        self.radius = (lo, hi)

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.radius)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(radius={self.radius})"


def eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(augment: bool) -> T.Compose:
    """One randomly chosen stain/blur augmentation per image, as in the paper."""
    if not augment:
        return eval_transform()
    return T.Compose([
        T.RandomChoice([
            HEDJitter(theta=0.1),
            T.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
            HEDJitter(theta=0.0),
            RandomGaussBlur(radius=(0.5, 1.5)),
        ]),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
