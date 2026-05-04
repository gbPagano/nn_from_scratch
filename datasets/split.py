"""Split the Kaggle MNIST training set into train/validation files.

Reads `datasets/kaggle_mnist/train.csv` and writes two new files in the same
directory: `train_split.csv` and `val_split.csv`. The split is deterministic
(seeded), so it can be regenerated identically. Run this BEFORE
`data_augmentation.py` so augmented variants of validation images never leak
into the training set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

INPUT_PATH = Path("datasets/kaggle_mnist/train.csv")
TRAIN_OUT = Path("datasets/kaggle_mnist/train_split.csv")
VAL_OUT = Path("datasets/kaggle_mnist/val_split.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples assigned to validation (default 0.1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--train-out", type=Path, default=TRAIN_OUT)
    parser.add_argument("--val-out", type=Path, default=VAL_OUT)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pl.read_csv(args.input)
    n = df.height
    print(f"Loaded {n} samples.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(round(n * args.val_ratio))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    train_df = df[train_idx.tolist()]
    val_df = df[val_idx.tolist()]

    print(f"Writing {args.train_out} ({train_df.height} samples)...")
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.write_csv(args.train_out)

    print(f"Writing {args.val_out} ({val_df.height} samples)...")
    val_df.write_csv(args.val_out)
    print("Done.")


if __name__ == "__main__":
    main()
