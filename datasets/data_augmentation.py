"""MNIST data augmentation using only numpy.

Reads `datasets/kaggle_mnist/train_split.csv` (produced by `split.py`),
generates N augmented copies per sample (random rotation, translation,
zoom, elastic deformation and gaussian noise), and writes the result to
`datasets/kaggle_mnist/train_augmented.csv` in the same schema
(label,pixel0..pixel783).

Originals are included by default so the model still sees clean digits;
pass --no-original to write only augmented samples.

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


def _gauss_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(t ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _conv1d_axis(x: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve `x` along `axis` with 1D `kernel`, reflect-padded."""
    r = (len(kernel) - 1) // 2
    pad = [(0, 0)] * x.ndim
    pad[axis] = (r, r)
    xp = np.pad(x, pad, mode="reflect")
    xp = np.moveaxis(xp, axis, -1)
    L = x.shape[axis]
    out = np.zeros(xp.shape[:-1] + (L,), dtype=x.dtype)
    for i, w in enumerate(kernel):
        out += w * xp[..., i:i + L]
    return np.moveaxis(out, -1, axis)


def gaussian_blur_2d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2D Gaussian blur on the last two axes."""
    k = _gauss_kernel_1d(sigma)
    x = _conv1d_axis(x, k, axis=-1)
    x = _conv1d_axis(x, k, axis=-2)
    return x


def affine_src_coords(
    h: int,
    w: int,
    angles_deg: np.ndarray,
    txs: np.ndarray,
    tys: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-mapping source coords for batched rotation+scale+translation
    around the image center. All param arrays are shape (B,)."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.deg2rad(angles_deg).astype(np.float32)
    cos_t = np.cos(theta)[:, None, None]
    sin_t = np.sin(theta)[:, None, None]
    inv_s = (1.0 / scales).astype(np.float32)[:, None, None]
    txs = txs.astype(np.float32)[:, None, None]
    tys = tys.astype(np.float32)[:, None, None]

    ys, xs = np.indices((h, w), dtype=np.float32)
    xs_c = xs[None] - cx - txs
    ys_c = ys[None] - cy - tys
    src_x = inv_s * (cos_t * xs_c + sin_t * ys_c) + cx
    src_y = inv_s * (-sin_t * xs_c + cos_t * ys_c) + cy
    return src_x, src_y


def elastic_displacement(
    b: int,
    h: int,
    w: int,
    rng: np.random.Generator,
    alpha: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simard 2003 elastic displacement fields. Returns (dx, dy) of shape (B, H, W)."""
    raw_x = rng.uniform(-1.0, 1.0, size=(b, h, w)).astype(np.float32)
    raw_y = rng.uniform(-1.0, 1.0, size=(b, h, w)).astype(np.float32)
    dx = gaussian_blur_2d(raw_x, sigma) * alpha
    dy = gaussian_blur_2d(raw_y, sigma) * alpha
    return dx.astype(np.float32), dy.astype(np.float32)


def warp_batch(imgs: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    """Bilinear gather. `imgs` (B,H,W); `src_x`, `src_y` (B,H,W). OOB -> 0."""
    b, h, w = imgs.shape
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

    b_idx = np.arange(b, dtype=np.int32)[:, None, None]
    Ia = imgs[b_idx, y0c, x0c]
    Ib = imgs[b_idx, y0c, x1c]
    Ic = imgs[b_idx, y1c, x0c]
    Id = imgs[b_idx, y1c, x1c]

    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bot * wy

    return np.where(valid, out, 0.0).astype(imgs.dtype)


def augment_batch(
    imgs: np.ndarray,
    rng: np.random.Generator,
    *,
    max_angle: float = 12.0,
    max_shift: float = 3.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    elastic_alpha: float = 8.0,
    elastic_sigma: float = 4.0,
    noise_std: float = 0.05,
) -> np.ndarray:
    n, h, w = imgs.shape
    angles = rng.uniform(-max_angle, max_angle, size=n)
    txs = rng.uniform(-max_shift, max_shift, size=n)
    tys = rng.uniform(-max_shift, max_shift, size=n)
    scales = rng.uniform(scale_range[0], scale_range[1], size=n)

    src_x, src_y = affine_src_coords(h, w, angles, txs, tys, scales)

    if elastic_alpha > 0 and elastic_sigma > 0:
        dx, dy = elastic_displacement(n, h, w, rng, elastic_alpha, elastic_sigma)
        src_x = src_x + dx
        src_y = src_y + dy

    out = warp_batch(imgs, src_x, src_y)

    if noise_std > 0:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(out.dtype)

    return np.clip(out, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copies",
        type=int,
        default=4,
        help="Number of augmented copies generated per original sample.",
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Skip the original (un-augmented) samples in the output.",
    )
    parser.add_argument("--max-angle", type=float, default=12.0)
    parser.add_argument("--max-shift", type=float, default=3.0)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument(
        "--elastic-alpha",
        type=float,
        default=8.0,
        help="Elastic deformation magnitude (0 disables).",
    )
    parser.add_argument(
        "--elastic-sigma",
        type=float,
        default=4.0,
        help="Gaussian smoothing sigma for the elastic field.",
    )
    parser.add_argument("--noise-std", type=float, default=0.05)
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

    if not args.no_original:
        chunks_x.append(pixels)
        chunks_y.append(labels)

    for k in range(args.copies):
        print(f"Generating augmented copy {k + 1}/{args.copies}...")
        aug = augment_batch(
            images,
            rng,
            max_angle=args.max_angle,
            max_shift=args.max_shift,
            scale_range=(args.scale_min, args.scale_max),
            elastic_alpha=args.elastic_alpha,
            elastic_sigma=args.elastic_sigma,
            noise_std=args.noise_std,
        )
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
