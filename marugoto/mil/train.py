#!/usr/bin/env python3

import json
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from fastai.vision.learner import Learner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn

from marugoto.mil._mil import train
from marugoto.mil.data import get_cohort_df
from marugoto.mil.helpers import _make_cat_enc, _make_cont_enc

__all__ = ["train_categorical"]


def main() -> None:
    parser = ArgumentParser("Train a categorical model on a cohort's tile's features.")
    add_train_args(parser)
    args = parser.parse_args()
    print(args)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    info = {
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

    train_result = train_categorical(
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


def train_categorical(
    *,
    clini_table: pd.DataFrame,
    slide_table: pd.DataFrame,
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

    if info is not None:
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

    patient_preds, patient_targs = learn.get_preds(act=nn.Softmax(dim=1))

    # TODO: The entire following section overlaps with deployment.
    # TODO: Deduplicate it.
    patient_preds_df = pd.DataFrame.from_dict(
        {
            "PATIENT": valid_df.PATIENT.values,
            target_label: valid_df[target_label].values,
            **{
                f"{target_label}_{cat}": patient_preds[:, i]
                for i, cat in enumerate(categories)
            },
        }
    )

    # calculate loss
    patient_preds = patient_preds_df[
        [f"{target_label}_{cat}" for cat in categories]
    ].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1)
    )
    patient_preds_df["loss"] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs), reduction="none"
    )

    patient_preds_df["pred"] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[
        [
            "PATIENT",
            target_label,
            "pred",
            *(f"{target_label}_{cat}" for cat in categories),
            "loss",
        ]
    ]
    patient_preds_df = patient_preds_df.sort_values(by="loss")

    return TrainResult(
        train_df=train_df,
        valid_df=valid_df,
        learn=learn,
        patient_preds_df=patient_preds_df,
    )


def read_table(path_or_df: Union[Path, pd.DataFrame], dtype=str):
    if df := isinstance(path_or_df, pd.DataFrame):
        return df
    elif path := isinstance(path_or_df, Path):
        if path.suffix(".xlsx"):
            return pd.read_excel(path, dtype=dtype)
        else:
            return pd.read_csv(path, dtype=dtype)
    else:
        raise ValueError(
            "path_or_df has to be either a Path or a Dataframe", f"{type(path_or_df)=}"
        )


if __name__ == "__main__":
    main()
