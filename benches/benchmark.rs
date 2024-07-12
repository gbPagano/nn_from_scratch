extern crate blas_src;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nn_from_scratch::layer::*;
use nn_from_scratch::functions::*;
use nn_from_scratch::*;

mod utils;
use utils::*;

fn bench_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Layers");
    let (mut layer_a, mut layer_b, inputs, desired) = simple_layers_a();

    group.sample_size(1000);
    group.bench_function("layers forward", |b| {
        b.iter(|| {
            layer_a.forward(black_box(&inputs));
            layer_b.forward(black_box(&layer_a.output.as_ref().unwrap()));
        })
    });

    let output_err = &desired - layer_b.output.as_ref().unwrap();
    group.bench_function("layers calc gradient descent", |b| {
        b.iter(|| {
            layer_b.gradient_descent(black_box(Some(&output_err)), black_box(None));
            layer_a.gradient_descent(black_box(None), black_box(Some(&layer_b)));
        })
    });
    let grad_b = layer_b.gradient_descent(Some(&output_err), None);
    let grad_a = layer_a.gradient_descent(None, Some(&layer_b));
    group.bench_function("layers update weights", |b| {
        b.iter(|| {
            layer_b.update_weights(black_box(0.5), black_box(&grad_b));
            layer_a.update_weights(black_box(0.5), black_box(&grad_a));
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
                let x_train = inputs.clone().insert_axis(ndarray::Axis(0));
                let y_train = desired.clone().insert_axis(ndarray::Axis(0));
                (x_train, y_train)
            },
            |(x_train, y_train)| {
                nn.fit(black_box(x_train), black_box(y_train), 100, 0.05, 1, 1000);
            },
        );
    });

    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    let (x_train, y_train) = mnist_f32();
    group.bench_function("mnist nn f32 fit", |b| {
        b.iter_with_setup(
            || {
                let mut nn: NeuralNetwork<f32> = NeuralNetwork::new(vec![
                    Dense::new(784, 28, ELU::new(1.0)),
                    Dense::new(28, 19, ELU::new(1.0)),
                    Dense::new(19, 10, TanH),
                ]);
                nn.terminal_output = false;

                (nn, x_train.clone(), y_train.clone())
            },
            |(mut nn, x_train, y_train)| {
                nn.fit(black_box(x_train), black_box(y_train), 24, 0.02, 8, 1000);
            },
        );
    });

    let (x_train, y_train) = mnist_f64();
    group.bench_function("mnist nn f64 fit", |b| {
        b.iter_with_setup(
            || {
                let mut nn: NeuralNetwork<f64> = NeuralNetwork::new(vec![
                    Dense::new(784, 28, ELU::new(1.0)),
                    Dense::new(28, 19, ELU::new(1.0)),
                    Dense::new(19, 10, TanH),
                ]);
                nn.terminal_output = false;

                (nn, x_train.clone(), y_train.clone())
            },
            |(mut nn, x_train, y_train)| {
                nn.fit(black_box(x_train), black_box(y_train), 24, 0.02, 8, 1000);
            },
        );
    });

    group.finish();
}

criterion_group!(benches, bench_layers, bench_nn);
criterion_main!(benches);
