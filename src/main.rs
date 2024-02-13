extern crate blas_src;

pub mod functions;
pub mod layer;
pub mod neural_network;
pub mod utils;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fs::File;

use crate::functions::*;
use crate::layer::Layer;
use crate::neural_network::NeuralNetwork;

#[derive(Serialize, Deserialize)]
struct ArrayWrapper {
    data: Vec<f64>,
    shape: Vec<usize>,
}

fn main() {
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

    let mut nn: NeuralNetwork<F> = NeuralNetwork::new(vec![
        Layer::new(784, 28, ELU::new(1.0)),
        Layer::new(28, 19, ELU::new(1.0)),
        Layer::new(19, 10, TanH),
    ]);
    nn.fit(x_train, y_train, 10, 0.02, 8, 5);

    let x_test = {
        let file = File::open("datasets/kaggle_mnist/test.csv").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        let mut data_train: Array2<F> = reader.deserialize_array2_dynamic().unwrap();

        data_train.map_inplace(|x| *x /= 255.0);
        data_train
    };

    let mut predictions: Vec<Row> = Vec::new();
    for (idx, x) in x_test.axis_iter(Axis(0)).enumerate() {
        let out = nn.forward(&x.to_owned());
        let (res, _) = out
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
            .unwrap();

        predictions.push(Row {
            image_id: idx + 1,
            label: res,
        })
    }

    {
        let file = File::create("kaggle-submission.csv").unwrap();
        let mut writer = WriterBuilder::new().from_writer(file);

        for row in predictions {
            writer.serialize(row).unwrap();
        }
    }
}

#[derive(serde::Serialize)]
struct Row {
    #[serde(rename = "ImageId")]
    image_id: usize,
    #[serde(rename = "Label")]
    label: usize,
}

fn number_to_neurons<F: Float>(n: usize, negative_output: F, positive_output: F) -> Vec<F> {
    let mut res = vec![negative_output; 10];
    res[n] = positive_output;
    res
}
