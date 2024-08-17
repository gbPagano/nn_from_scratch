extern crate blas_src;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::*;
use nn_from_scratch::layers::activation::*;
use nn_from_scratch::layers::*;
use nn_from_scratch::loss::*;
use nn_from_scratch::*;

mod utils;
use utils::*;

fn bench_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Layers");
    let (mut layer_a, mut layer_b, mut activate, _, inputs, _) = simple_layers_a();

    group.sample_size(1000);
    group.bench_function("layers forward", |b| {
        b.iter(|| {
            let out_1 = layer_a.forward(inputs.clone());
            let out_2 = activate.forward(out_1);
            let out_3 = layer_b.forward(out_2);
            activate.forward(out_3);
        })
    });
    group.finish();
}

fn bench_nn(c: &mut Criterion) {
    let mut group = c.benchmark_group("Neural Networks");
    let (mut nn, inputs, desired) = simple_nn();

    group.sample_size(200);
    group.bench_function("simple nn fit", |b| {
        b.iter_with_setup(
            || {
                let x_train = inputs
                    .clone()
                    .insert_axis(ndarray::Axis(0))
                    .into_dimensionality::<Ix3>()
                    .unwrap();
                let y_train = desired
                    .clone()
                    .insert_axis(ndarray::Axis(0))
                    .into_dimensionality::<Ix3>()
                    .unwrap();
                (x_train, y_train)
            },
            |(x_train, y_train)| {
                nn.fit(
                    black_box(&x_train),
                    black_box(&y_train),
                    NNConfig {
                        epochs: 100,
                        evaluate_step: 100,
                        loss_function: HalfMSE::new().into(),
                        batch_size: 1,
                        learning_rate: 0.5,
                    },
                );
            },
        );
    });

    group.sample_size(10);
    let (x_train, y_train) = mnist_f32();
    group.bench_function("mnist nn f32 fit", |b| {
        b.iter_with_setup(
            || {
                type F = f32;
                let mut nn: NeuralNetwork<F> = NeuralNetwork::new(box_layers![
                    Dense::new(784, 28),
                    ELU::new(1.0 as f32),
                    Dense::new(28, 19),
                    ELU::new(1.0 as f32),
                    Dense::new(19, 10),
                    TanH::new()
                ]);
                nn.terminal_output = false;

                (nn, x_train.clone(), y_train.clone())
            },
            |(mut nn, x_train, y_train)| {
                nn.fit(
                    black_box(&x_train),
                    black_box(&y_train),
                    NNConfig {
                        epochs: 1,
                        evaluate_step: 100,
                        loss_function: HalfMSE::new().into(),
                        batch_size: 8,
                        learning_rate: 0.02,
                    },
                );
            },
        );
    });
    group.finish();
}

criterion_group!(benches, bench_layers, bench_nn);
criterion_main!(benches);
