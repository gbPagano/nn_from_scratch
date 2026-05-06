# Neural Networks from scratch

A neural network library written from scratch. First as a Python prototype, then
rewritten in Rust with `ndarray` + OpenBLAS for performance.

This is a personal project, built to learn the internals of neural networks more
deeply (forward/backward passes, convolutions, optimizers, etc.) by implementing
everything by hand instead of relying on a framework. The concrete goal was to
reach **99% accuracy on MNIST** using only this hand-rolled code.

That goal has been reached, so the library is **no longer under active
development**. It is kept here as a finished learning project.

The main entry point is the `mnist` binary, which trains a CNN on the
[Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
dataset and writes a submission CSV.

> `model.bin` (in the repo root) is the trained checkpoint that scored
> **99.3%** on the Kaggle Digit Recognizer leaderboard. Use `--predict model.bin`
> to reproduce the submission without retraining.

## Features

- Layers: `Dense`, `Conv2D`, `MaxPooling`, `Flatten`
- Activations: `ReLU`, `ELU`, `Sigmoid`, `Tanh`, `Softmax`
- Losses: `MSE`, `CrossEntropy` (with fused `SoftmaxCE` for stability)
- Optimizers: SGD and Adam
- Training: mini-batch SGD, validation, early stopping (loss or accuracy) with
  best-weights restore, deterministic seeding, graceful Ctrl-C
- Save / load checkpoints via `bincode`
- Parallelism through `rayon` and BLAS-backed `ndarray`


## Network architecture (mnist binary)

```
Input (1, 28, 28)
  → Conv 1→32, 3x3   → ELU
  → MaxPool 2x2
  → Conv 32→64, 3x3  → ELU
  → MaxPool 2x2
  → Flatten
  → Dense 64·6·6 → 128 → ELU
  → Dense 128 → 10
  → SoftmaxCE
```

Optimizer: Adam with `lr = 1e-4`, `batch_size = 32`.

## Setup

Rust (stable) and OpenBLAS are required. On Arch:

```sh
sudo pacman -S openblas
```

## Usage

### 1. Prepare the dataset

Place `train.csv` and `test.csv` from the Kaggle competition under
`datasets/kaggle_mnist/`, then:

```sh
# train / validation split
python datasets/split.py

# (optional) generate augmented copies for the training set
python datasets/data_augmentation.py --copies 4
```

`data_augmentation.py` applies random rotation, translation, zoom, elastic
deformation (Simard 2003) and Gaussian noise, and writes
`train_augmented.csv`.

### 2. Train

```sh
cargo run --release --bin mnist -- \
    --train-path datasets/kaggle_mnist/train_augmented.csv \
    --epochs 30 \
    --early-stopping-patience 5 \
    --save my_model.bin
```

Useful flags:

- `--seed <u64>`: deterministic init
- `--early-stopping-metric loss|accuracy`
- `--restore-best-weights true|false`
- `Ctrl-C` once during training stops cleanly and keeps the best checkpoint;
  twice exits immediately

A Kaggle submission CSV is written at the end (default name:
`kaggle-submission-YYYYMMDDHHMM.csv`).

### 3. Predict from an existing checkpoint

```sh
cargo run --release --bin mnist -- --predict model.bin
```

This reproduces the 99.3% submission. Override the output path with
`--prediction-path my-submission.csv` if needed.
