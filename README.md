# Generalization of vision pre-trained models for histopathology

Code for Sikaroudi et al., *Generalization of vision pre-trained models for histopathology*, Scientific Reports 13, 6065 (2023).

The experiment is leave-one-site-out on Camelyon17: train a backbone on four hospitals, select the checkpoint on a validation split of those four, and measure accuracy, AUC and F-beta on the fifth hospital, which the model never sees.

## Corrections since publication (October 2026)

A review of the original code found four bugs. All four are fixed here. Each one changes what a retrained model learns, so new runs will not reproduce the numbers in the paper exactly.

| Bug in the 2023 code | Effect | Fix |
|---|---|---|
| `CrossEntropyLoss` received softmax outputs | Softmax applied twice; flattened gradients | The loss takes raw logits |
| `HEDJitter` and `RandomGaussBlur` drew their random parameters once, in `__init__` | Every training image got the same stain shift and blur, so augmentation was a fixed colour transform | Parameters are drawn per image |
| `random_split` sliced one dataset that carried the training transform | Validation images were augmented | Separate train and validation datasets; only training is augmented |
| The KimiaNet loader stripped `module.` and loaded with `strict=False` | With a checkpoint in KimiaNet's published layout (`module.model.*`), no weights load and nothing reports it; `tests/test_models.py` reproduces this | Keys are remapped, and loading fails if any feature tensor is missing |

Check the last row against your own copy of `KimiaNetPyTorchWeights.pth`: with the published layout, the KimiaNet results in the paper came from a randomly initialised DenseNet-121.

## Install

```bash
pip install -e .            # add [wandb] to log runs, [dev] for tests
```

Python 3.10+, PyTorch 2.1+.

## Data

One CSV per site in `dataframe_root`, named `<site>.csv`, with two columns:

| column | meaning |
|---|---|
| `file_paths` | path to an RGB patch |
| `class_label` | 0 = normal, 1 = tumour |

## Run

```bash
histogen-train --config configs/camelyon17.json
histogen-train --config configs/camelyon17.json --holdout-site center_3 --model resnet50
histogen-train --config configs/camelyon17.json --set batch_size=32 --set augmentation_in_training=true
```

`python main.py` still works. Each run saves `holdout-<site>_<model>_…pt` to `output_dir` with the best `val_in` weights and the config used.

### Models

`resnet18`, `resnet50`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d` (torchvision, ImageNet weights if `pretrained`); `resnet18_ssl`, `resnet18_swsl` (Yalniz et al. 2019, from torch.hub); `kimianet` (DenseNet-121 with weights from `kimianet_weight_path`).

### Config keys

| key | default | |
|---|---|---|
| `sites`, `holdout_site` | required | Site names; the hold-out is excluded from training |
| `train_val_portions` | `[70, 10]` | Percent of training-site patches for train and validation; the rest is unused |
| `augmentation_in_training` | `false` | One of HED jitter (θ = 0.1 or 0), brightness/contrast jitter, or Gaussian blur per image |
| `frozen_children` | `1` | Leading child modules to freeze (ResNet `conv1`) |
| `seed` | `0` | Seeds the split, the weights and the augmentations |
| `wandb_project` | unset | Log to Weights & Biases; the entity comes from `WANDB_ENTITY` |

Configs from 2023 (`trail_sites`, `holdout_trial_site`) still load.

## Layout

```
src/histogen/
  config.py      Config dataclass, JSON + overrides
  data.py        PatchDataset, seeded site split, loaders
  transforms.py  HEDJitter, RandomGaussBlur, train/eval pipelines
  models.py      backbones, KimiaNet loading, layer freezing
  train.py       epoch loop, dataset-level metrics, best-checkpoint selection
  cli.py         histogen-train
tests/           runs on CPU with synthetic patches
```

```bash
pip install -e ".[dev]" && pytest
```

## Citing

```bibtex
@article{sikaroudi2023generalization,
  title   = {Generalization of vision pre-trained models for histopathology},
  author  = {Sikaroudi, Milad and Hosseini, Maryam and Gonzalez, Ricardo and Rahnamayan, Shahryar and Tizhoosh, HR},
  journal = {Scientific Reports},
  volume  = {13},
  number  = {1},
  pages   = {6065},
  year    = {2023}
}
```

## Credits

`HEDJitter` implements the HED augmentation of Tellez et al. (IEEE TMI 2018), after [gatsby2016/Augmentation-PyTorch-Transforms](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms) (Apache-2.0). KimiaNet is by Riasatian et al. (Medical Image Analysis 2021).

## License

MIT. Contact: msikaroudi@uwaterloo.ca
