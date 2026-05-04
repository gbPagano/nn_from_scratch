"""MNIST data augmentation using only numpy.

Reads `datasets/kaggle_mnist/train_split.csv` (produced by `split.py`),
generates N augmented copies per sample (random rotation, translation,
zoom and gaussian noise), and writes the result to
`datasets/kaggle_mnist/train_augmented.csv` in the same schema
(label,pixel0..pixel783).

Run `split.py` first so validation images are not augmented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

IMG_SIZE = 28
INPUT_PATH = Path("datasets/kaggle_mnist/train_split.csv")
OUTPUT_PATH = Path("datasets/kaggle_mnist/train_augmented.csv")


def affine_warp(
    img: np.ndarray,
    angle_deg: float,
    tx: float,
    ty: float,
    scale: float,
) -> np.ndarray:
    """Apply rotation + scale + translation around the image center using
    inverse mapping with bilinear interpolation. Out-of-bounds pixels are 0."""
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    inv_s = 1.0 / scale

    ys, xs = np.indices((h, w), dtype=np.float32)
    xs_c = xs - cx - tx
    ys_c = ys - cy - ty

    src_x = inv_s * (cos_t * xs_c + sin_t * ys_c) + cx
    src_y = inv_s * (-sin_t * xs_c + cos_t * ys_c) + cy

    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = src_x - x0
    wy = src_y - y0

    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bot * wy

    return np.where(valid, out, 0.0).astype(img.dtype)


def augment_image(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    max_angle: float = 12.0,
    max_shift: float = 2.5,
    scale_range: tuple[float, float] = (0.9, 1.1),
    noise_std: float = 0.03,
) -> np.ndarray:
    angle = rng.uniform(-max_angle, max_angle)
    tx = rng.uniform(-max_shift, max_shift)
    ty = rng.uniform(-max_shift, max_shift)
    scale = rng.uniform(*scale_range)

    out = affine_warp(img, angle, tx, ty, scale)

    if noise_std > 0:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(out.dtype)

    return np.clip(out, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copies",
        type=int,
        default=2,
        help="Number of augmented copies generated per original sample.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Also include the original (un-augmented) samples in the output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.input}...")
    df = pl.read_csv(args.input)
    labels = df["label"].to_numpy()
    pixels = df.drop("label").to_numpy().astype(np.float32) / 255.0
    n = pixels.shape[0]
    images = pixels.reshape(n, IMG_SIZE, IMG_SIZE)
    print(f"Loaded {n} samples.")

    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []

    if args.keep_original:
        chunks_x.append(pixels)
        chunks_y.append(labels)

    for k in range(args.copies):
        print(f"Generating augmented copy {k + 1}/{args.copies}...")
        aug = np.empty_like(images)
        for i in range(n):
            aug[i] = augment_image(images[i], rng)
        chunks_x.append(aug.reshape(n, IMG_SIZE * IMG_SIZE))
        chunks_y.append(labels)

    x_out = np.concatenate(chunks_x, axis=0)
    y_out = np.concatenate(chunks_y, axis=0)

    x_out = np.round(x_out * 255.0).clip(0, 255).astype(np.uint8)

    print(f"Building output dataframe with {x_out.shape[0]} rows...")
    pixel_cols = {f"pixel{i}": x_out[:, i] for i in range(IMG_SIZE * IMG_SIZE)}
    out_df = pl.DataFrame({"label": y_out, **pixel_cols})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.output}...")
    out_df.write_csv(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
