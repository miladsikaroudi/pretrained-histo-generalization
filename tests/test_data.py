import torch

from histogen.config import Config
from histogen.data import make_dataloaders, read_sites, split_indices
from histogen.transforms import eval_transform


def test_split_is_seeded_disjoint_and_complete():
    a = split_indices(100, (70, 10), seed=3)
    assert a == split_indices(100, (70, 10), seed=3)
    assert a != split_indices(100, (70, 10), seed=4)
    train, val, test = a
    assert (len(train), len(val), len(test)) == (70, 10, 20)
    assert sorted(train + val + test) == list(range(100))


def test_read_sites_concatenates(site_dir):
    assert len(read_sites(site_dir, ["a", "b"])) == 24


def test_holdout_site_is_never_trained_on(config_file):
    cfg = Config.from_json(config_file)
    loaders = make_dataloaders(cfg)
    train_paths = {loaders["train"].dataset.dataset.paths[i] for i in loaders["train"].dataset.indices}
    assert train_paths and not any("/c_" in p for p in train_paths)
    assert all("/c_" in p for p in loaders["test_out"].dataset.paths)


def test_only_training_split_is_augmented(config_file):
    loaders = make_dataloaders(Config.from_json(config_file))
    val = loaders["val_in"].dataset
    x, _ = val[0]
    expected = eval_transform()(__import__("PIL.Image").Image.open(val.dataset.paths[val.indices[0]]).convert("RGB"))
    assert torch.equal(x, expected)
    assert loaders["train"].dataset.dataset.transform is not val.dataset.transform
