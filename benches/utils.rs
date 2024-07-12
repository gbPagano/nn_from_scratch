extern crate blas_src;

use num_traits::Float;
use csv::ReaderBuilder;
use ndarray::prelude::*;
use ndarray::{array, Array1, Array2};
use ndarray_csv::Array2Reader;
use nn_from_scratch::functions::*;
use nn_from_scratch::layer::*;
use nn_from_scratch::*;
use std::fs::File;

pub fn number_to_neurons<F: Float>(n: usize, negative_output: F, positive_output: F) -> Vec<F> {
    let mut res = vec![negative_output; 10];
    res[n] = positive_output;
    res
}

pub fn simple_layers_a() -> (Dense<f64>, Dense<f64>, Array1<f64>, Array1<f64>) {
    let mut layer_a = Dense::new(2, 2, Sigmoid);
    layer_a.weights = array![[0.15, 0.2], [0.25, 0.3]];
    layer_a.bias = array![0.35, 0.35];

    let mut layer_b = Dense::new(2, 2, Sigmoid);
    layer_b.weights = array![[0.4, 0.45], [0.5, 0.55]];
    layer_b.bias = array![0.6, 0.6];

    let inputs = array![0.05, 0.1];
    let desired = array![0.01, 0.99];

    (layer_a, layer_b, inputs, desired)
}

pub fn simple_nn() -> (NeuralNetwork<f64>, Array1<f64>, Array1<f64>) {
    let mut layer_a = Dense::new(2, 2, Sigmoid);
    layer_a.weights = array![[0.15, 0.2], [0.25, 0.3]];
    layer_a.bias = array![0.35, 0.35];

    let mut layer_b = Dense::new(2, 2, Sigmoid);
    layer_b.weights = array![[0.4, 0.45], [0.5, 0.55]];
    layer_b.bias = array![0.6, 0.6];

    let inputs = array![0.05, 0.1];
    let desired = array![0.01, 0.99];
    let nn = NeuralNetwork {
        layers: vec![Box::new(layer_a), Box::new(layer_b)],
        terminal_output: false,
    };

    (nn, inputs, desired)
}

pub fn mnist_f32() -> (Array2<f32>, Array2<f32>) {
    type F = f32;

    let (x_train, y_train) = {
        let file = File::open("datasets/kaggle_mnist/train.csv").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        let data_train: Array2<F> = reader.deserialize_array2_dynamic().unwrap();

        let mut x_train: Array2<F> = data_train.slice(s![.., 1..]).to_owned();
        x_train.map_inplace(|x| *x /= 255.0);

        let y_train = data_train.column(0).to_owned();
        let y_train = y_train
            .iter()
            .map(|&n| number_to_neurons::<F>(n as usize, -1.0, 1.0))
            .collect::<Vec<_>>();
        let y_train: Array2<F> = Array::from_shape_vec(
            (y_train.len(), y_train[0].len()),
            y_train.into_iter().flatten().collect(),
        )
        .unwrap();

        (x_train, y_train)
    };

    (x_train, y_train)
}

pub fn mnist_f64() -> (Array2<f64>, Array2<f64>) {
    type F = f64;

    let (x_train, y_train) = {
        let file = File::open("datasets/kaggle_mnist/train.csv").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        let data_train: Array2<F> = reader.deserialize_array2_dynamic().unwrap();

        let mut x_train: Array2<F> = data_train.slice(s![.., 1..]).to_owned();
        x_train.map_inplace(|x| *x /= 255.0);

        let y_train = data_train.column(0).to_owned();
        let y_train = y_train
            .iter()
            .map(|&n| number_to_neurons::<F>(n as usize, -1.0, 1.0))
            .collect::<Vec<_>>();
        let y_train: Array2<F> = Array::from_shape_vec(
            (y_train.len(), y_train[0].len()),
            y_train.into_iter().flatten().collect(),
        )
        .unwrap();

        (x_train, y_train)
    };

    (x_train, y_train)
}
