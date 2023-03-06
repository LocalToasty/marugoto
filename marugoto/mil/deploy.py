import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import torch
import torch.nn.functional as F
from fastai.vision.all import DataLoader, Learner, load_learner
from torch import nn

from .data import get_cohort_df, get_target_enc, make_dataset

__all__ = ["deploy_from_clini_slide", "deploy"]


def add_deploy_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds arguments required for model deployment to an ArgumentParser."""
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
        "--model",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path of the model to deploy.",
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        type=str,
        required=False,
        help="Label to train for. Inferred from model, if none given.",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        required=True,
        help="Path to write the outputs to.",
    )

    return parser


def main():
    parser = ArgumentParser("Deploy a categorical model.")
    parser = add_deploy_args(parser)
    args = parser.parse_args()

    if (preds_csv := args.output_dir / "patient-preds.csv").exists():
        print(f"{preds_csv} already exists!  Skipping...")
        exit(0)

    learn = load_learner(args.model_path)
    args.output_path.mkdir(parents=True, exist_ok=True)
    patient_preds_df = deploy_from_clini_slide(
        learn=learn,
        clini_table=args.clini_table,
        slide_table=args.slide_table,
        feature_dir=args.feature_dir,
        target_label=args.target_label,
    )
    patient_preds_df.to_csv(preds_csv, index=False)


def deploy_from_clini_slide(
    learn: Learner,
    clini_table: Union[Path, pd.DataFrame],
    slide_table: Union[Path, pd.DataFrame],
    feature_dir: Path,
    target_label: str,
) -> pd.DataFrame:
    target_enc = get_target_enc(learn)

    categories = target_enc.categories_[0]

    target_label = target_label or learn.target_label

    test_df, _ = get_cohort_df(
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,
        target_label=target_label,
        categories=categories,
    )
    return deploy(test_df=test_df, learn=learn, target_label=target_label)


def deploy(
    test_df: pd.DataFrame,
    learn: Learner,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None,
    cont_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), "duplicate patients!"
    if target_label is None:
        target_label = learn.target_label
    if cat_labels is None:
        cat_labels = learn.cat_labels
    if cont_labels is None:
        cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None,
    )

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )

    # removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict(
        {
            "PATIENT": test_df.PATIENT.values,
            target_label: test_df[target_label].values,
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

    return patient_preds_df


if __name__ == "__main__":
    main()
