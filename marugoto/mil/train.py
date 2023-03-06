#!/usr/bin/env python3
import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from fastai.vision.all import (
    CSVLogger,
    DataLoader,
    DataLoaders,
    Learner,
    RocAuc,
    SaveModelCallback,
)
from fastai.vision.learner import Learner
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from marugoto.data import SKLearnEncoder

from .data import get_cohort_df, make_dataset
from .deploy import deploy
from .model import MILModel

__all__ = ["train_from_clini_slide", "train"]


def main() -> None:
    parser = ArgumentParser("Train a categorical model on a cohort's tile's features.")
    add_train_args(parser)
    args = parser.parse_args()
    print(args)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    info: Dict[str, Any] = {
        "description": "MIL training",
        "clini": str(args.clini_table.absolute()),
        "slide": str(args.slide_table.absolute()),
        "feature_dir": str(args.feature_dir.absolute()),
        "target_label": str(args.target_label),
        "cat_labels": [str(c) for c in (args.cat_labels or [])],
        "cont_labels": [str(c) for c in (args.cont_labels or [])],
        "output_dir": str(args.output_dir.absolute()),
        "datetime": datetime.now().astimezone().isoformat(),
    }
    model_path = args.output_dir / "export.pkl"
    if model_path.exists():
        print(f"{model_path} already exists. Skipping...")
        exit(0)

    train_result = train_from_clini_slide(
        clini_table=args.clini_table,
        slide_table=args.slide_table,
        feature_dir=args.feature_dir,
        target_label=args.target_label,
        cat_labels=args.cat_labels,
        cont_labels=args.cont_labels,
        output_dir=args.output_dir,
        info=info,
    )

    train_result.train_df.drop(columns="slide_path").to_csv(
        args.output_dir / "train.csv", index=False
    )
    train_result.valid_df.drop(columns="slide_path").to_csv(
        args.output_dir / "valid.csv", index=False
    )

    info["class distribution"]["training"] = {
        k: int(v)
        for k, v in train_result.train_df[args.target_label].value_counts().items()
    }
    info["class distribution"]["validation"] = {
        k: int(v)
        for k, v in train_result.valid_df[train_result.target_label]
        .value_counts()
        .items()
    }

    with open(args.output_dir / "info.json", "w") as f:
        json.dump(info, f)

    train_result.learn.export()
    train_result.patient_preds_df.to_csv(
        args.output_dir / "patient-preds-validset.csv", index=False
    )


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds arguments required for model training to an ArgumentParser."""
    parser.add_argument(
        "--clini-table",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path of the clini table.",
    )
    parser.add_argument(
        "--slide-table",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path of the slide table.",
    )
    parser.add_argument(
        "--feature-dir",
        metavar="DIR",
        type=Path,
        required=True,
        help="Path the h5 features are saved in.",
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        type=str,
        required=True,
        help="Label to train for.",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        required=True,
        help="Path to write the outputs to.",
    )
    parser.add_argument(
        "--category",
        metavar="CAT",
        dest="categories",
        type=str,
        required=False,
        action="append",
        help=(
            "Category to train for. "
            "Has to be an entry specified in the clini table's column "
            "specified by `--clini-table`. "
            "Can be given multiple times to specify multiple categories to train for. "
        ),
    )

    multimodal_group = parser.add_argument_group("Multimodal training")
    multimodal_group.add_argument(
        "--additional-training-category",
        metavar="LABEL",
        dest="cat_labels",
        type=str,
        required=False,
        action="append",
        help="Categorical column in clini table to additionally use in training.",
    )
    multimodal_group.add_argument(
        "--additional-training-continuous",
        metavar="LABEL",
        dest="cont_labels",
        type=str,
        required=False,
        action="append",
        help="Continuous column in clini table to additionally use in training.",
    )

    return parser


@dataclass
class TrainResult:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    learn: Learner
    patient_preds_df: pd.DataFrame
    info: Dict[str, Any]


def train_from_clini_slide(
    *,
    clini_table: Union[Path, pd.DataFrame],
    slide_table: Union[Path, pd.DataFrame],
    feature_dir: Path,
    target_label: str,
    categories: Optional[npt.NDArray[np.str_]] = None,
    cat_labels: Optional[Iterable[str]] = None,
    cont_labels: Optional[Iterable[str]] = None,
    output_dir: Optional[Path] = None,
    n_epoch: int = 32,
    info: Optional[Dict[str, Any]] = None,
):
    df, categories = get_cohort_df(
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,
        target_label=target_label,
        categories=categories,
    )

    logging.info("Overall distribution")
    logging.info(df[target_label].value_counts())
    assert not df[
        target_label
    ].empty, "no input dataset. Do the tables / feature dir belong to the same cohorts?"

    info = info if info is not None else {}
    info["class distribution"] = {
        "overall": {k: int(v) for k, v in df[target_label].value_counts().items()}
    }

    # Split off validation set
    train_patients, valid_patients = train_test_split(
        df.PATIENT, stratify=df[target_label]
    )
    train_df = df[df.PATIENT.isin(train_patients)]
    valid_df = df[df.PATIENT.isin(valid_patients)]

    target_enc = OneHotEncoder(sparse_output=False).fit(categories.reshape(-1, 1))

    add_features = []
    if cat_labels:
        add_features.append(
            (_make_cat_enc(train_df, cat_labels), df[cat_labels].values)
        )
    if cont_labels:
        add_features.append(
            (_make_cont_enc(train_df, cont_labels), df[cont_labels].values)
        )

    learn = train(
        bags=df.slide_path.values,
        targets=(target_enc, df[target_label].values),
        add_features=add_features,
        valid_idxs=df.PATIENT.isin(valid_patients).values,
        path=output_dir,
        n_epoch=n_epoch,
    )

    # save some additional information to the learner to make deployment easier
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    # deploy on validation set
    patient_preds_df = deploy(
        learn=learn,
        test_df=valid_df,
        target_label=target_label,
    )

    return TrainResult(
        train_df=train_df,
        valid_df=valid_df,
        learn=learn,
        patient_preds_df=patient_preds_df,
        info=info,
    )


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, npt.NDArray],
    add_features: Iterable[Tuple[SKLearnEncoder, npt.NDArray]] = [],
    valid_idxs: npt.NDArray[np.int_],
    n_epoch: int = 32,
    path: Optional[Path] = None,
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    target_enc, targs = targets
    train_ds = make_dataset(
        bags=bags[~valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[~valid_idxs]),
        add_features=[(enc, vals[~valid_idxs]) for enc, vals in add_features],
        bag_size=512,
    )

    valid_ds = make_dataset(
        bags=bags[valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[valid_idxs]),
        add_features=[(enc, vals[valid_idxs]) for enc, vals in add_features],
        bag_size=None,
    )

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=1, drop_last=True
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )
    batch = train_dl.one_batch()

    model = MILModel(batch[0].shape[-1], batch[-1].shape[-1])

    # weigh inversely to class occurances
    counts = pd.value_counts(targs[~valid_idxs])
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32
    )
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl)
    learn = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=path)

    cbs = [
        SaveModelCallback(fname=f"best_valid"),
        CSVLogger(),
    ]

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs)

    return learn


def read_table(path_or_df: Union[Path, pd.DataFrame], dtype=str):
    if isinstance((df := path_or_df), pd.DataFrame):
        return df
    elif isinstance((path := path_or_df), Path):
        if path.suffix == ".xlsx":
            return pd.read_excel(path, dtype=dtype)
        else:
            return pd.read_csv(path, dtype=dtype)
    else:
        raise ValueError(
            "path_or_df has to be either a Path or a Dataframe", f"{type(path_or_df)=}"
        )


def _make_cat_enc(df: pd.DataFrame, cats: Iterable[str]) -> SKLearnEncoder:
    # create a scaled one-hot encoder for the categorical values
    #
    # due to weirdeties in sklearn's OneHotEncoder.fit we fill NAs with other values
    # randomly sampled with the same probability as their distribution in the
    # dataset.  This is necessary for correctly determining StandardScaler's weigth
    fitting_cats = []
    for cat in cats:
        weights = df[cat].value_counts(normalize=True)
        non_na_samples = df[cat].fillna(
            pd.Series(np.random.choice(weights.index, len(df), p=weights))
        )
        fitting_cats.append(non_na_samples)
    cat_samples = np.stack(fitting_cats, axis=1)
    cat_enc = make_pipeline(
        OneHotEncoder(sparse=False, handle_unknown="ignore"),
        StandardScaler(),
    ).fit(cat_samples)
    return cat_enc


def _make_cont_enc(df, conts) -> SKLearnEncoder:
    cont_enc = make_pipeline(StandardScaler(), SimpleImputer(fill_value=0)).fit(
        df[conts].values
    )
    return cont_enc


if __name__ == "__main__":
    main()
