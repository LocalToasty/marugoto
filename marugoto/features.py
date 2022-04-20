"""Train a network on MIL h5 bag features."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence, Optional, TypeVar

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from fastai.vision.all import (
    create_head, Learner, RocAuc,
    SaveModelCallback, EarlyStoppingCallback, CSVLogger,
    DataLoader, DataLoaders)

from .data import ZipDataset, EncodedDataset


__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'mvantreeck@ukaachen.de'


__all__ = ['train', 'make_dataset', 'H5TileDataset']


T = TypeVar('T')


def train(
    *,
    target_enc,
    train_bags: Sequence[Path],
    train_targets: Sequence[T],
    valid_bags: Sequence[Path],
    valid_targets: Sequence[T],
    n_epoch: int = 32,
    patience: int = 4,
) -> Learner:
    """Train a MLP on image features.

    Args:
        target_enc:  A scikit learn encoder mapping the targets to arrays
            (e.g. `OneHotEncoder`).
        train_bags:  H5s containing the bags to train on (cf.
            `marugoto.mil`).
        train_targets:  The ground truths of the training bags.
        valid_bags:  H5s containing the bags to train on.
        train_targets:  The ground thruths of the validation bags.
    """
    train_ds = make_dataset(target_enc, train_bags, train_targets)
    valid_ds = make_dataset(target_enc, valid_bags, valid_targets, seed=0)

    # build dataloaders
    # `or 4` to appease mypy (os.cpu_count() can be `None`)
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=min(16, os.cpu_count() or 4))
    valid_dl = DataLoader(
        valid_ds, batch_size=512, shuffle=False, num_workers=min(16, os.cpu_count() or 4))

    model = nn.Sequential(
        create_head(512, 2)[1:],
        nn.Softmax(dim=1))

    # weigh inversely to class occurances
    counts = pd.value_counts(train_targets)
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl)
    learn = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()])

    cbs = [
        SaveModelCallback(monitor='roc_auc_score', fname=f'best_valid'),
        EarlyStoppingCallback(monitor='roc_auc_score',
                              min_delta=0.001, patience=patience),
        CSVLogger()]

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-3, cbs=cbs)

    return learn


def make_dataset(
    target_enc,
    bags: Sequence[Path], targets: Sequence[Any],
    seed: Optional[int] = 0
) -> ZipDataset:
    """Creates a instance-wise dataset from MIL bag H5DFs."""
    assert len(bags) == len(targets), \
        'number of bags and ground truths does not match!'
    tile_ds: ConcatDataset = ConcatDataset(
        H5TileDataset(h5, seed=seed) for h5 in bags)
    lens = np.array([len(ds) for ds in tile_ds.datasets])
    ys = np.repeat(targets, lens)
    ds = ZipDataset(
        tile_ds,
        EncodedDataset(target_enc, ys, dtype=torch.float32))    # type: ignore
    return ds


@dataclass
class H5TileDataset(Dataset):
    """A dataset containing the instances of a MIL bag."""
    h5path: Path
    """H5DF file to take the bag tile features from.

    The file has to contain a dataset 'feats' of dimension NxF, where N is
    the number of tiles and F the dimension of the tile's feature vector.
    """
    tile_no: Optional[int] = 256
    """Number of tiles to sample (with replacement) from the bag.
    
    If `tile_no` is `None`, _all_ the bag's tiles will be taken.
    """
    seed: Optional[int] = None
    """Seed to initialize the RNG for sampling.

    If `tile_no` is `None`, this option has no effect and all the bag's
    tiles will be given in the same order as in the h5.
    """

    def __post_init__(self):
        assert not self.seed or self.tile_no, \
            '`seed` must not be set if `tile_no` is `None`.'

    def __getitem__(self, index) -> torch.Tensor:
        with h5py.File(self.h5path, mode='r') as f:
            if self.tile_no:
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                index = torch.randint(
                    len(f['feats']), (self.tile_no or len(f['feats']),))[index]
            return torch.tensor(f['feats'][index]).unsqueeze(0)

    def __len__(self):
        if self.tile_no:
            return self.tile_no

        with h5py.File(self.h5path, mode='r') as f:
            len(f['feats'])
