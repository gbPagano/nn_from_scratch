extern crate blas_src;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use num_traits::Float;
use std::fs::File;

use nn_from_scratch::layers::activation::*;
use nn_from_scratch::layers::*;
use nn_from_scratch::loss::*;
use nn_from_scratch::*;

type F = f32;

fn main() {
    let (x_train, y_train) = load_mnist_dataset();

    let mut nn: NeuralNetwork<F> = NeuralNetwork::new(box_layers![
        Dense::new(784, 28),
        ELU::new(1.0 as F),
        Dense::new(28, 19),
        ELU::new(1.0 as F),
        Dense::new(19, 10),
        SoftmaxCE::new()
    ]);
    nn.fit(
        &x_train,
        &y_train,
        NNConfig {
            epochs: 100,
            learning_rate: 0.01,
            batch_size: 8,
            evaluate_step: 5,
            loss_function: CrossEntropySoftmax::new().into(),
        },
    );
    kaggle_predictions(&mut nn);
}

fn number_to_neurons<F: Float>(n: usize, negative_output: F, positive_output: F) -> Vec<F> {
    let mut res = vec![negative_output; 10];
    res[n] = positive_output;
    res
}

fn load_mnist_dataset() -> (Vec<ArrayD<F>>, Vec<ArrayD<F>>) {
    let file = File::open("datasets/kaggle_mnist/train.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data_train: Array2<F> = reader.deserialize_array2_dynamic().unwrap();

    let mut x_train: Array2<F> = data_train.slice(s![.., 1..]).to_owned();
    x_train.map_inplace(|x| *x /= 255.0);

    let y_train = data_train.column(0).to_owned();
    let y_train = y_train
        .iter()
        .map(|&n| number_to_neurons::<F>(n as usize, 0.0, 1.0))
        .collect::<Vec<_>>();
    let y_train: Array2<F> = Array::from_shape_vec(
        (y_train.len(), y_train[0].len()),
        y_train.into_iter().flatten().collect(),
    )
    .unwrap();

    let x_train = x_train
        .insert_axis(ndarray::Axis(2))
        .into_dimensionality::<Ix3>()
        .unwrap()
        .axis_iter(Axis(0))
        .map(|item| item.into_owned().into_dyn())
        .collect();
    let y_train = y_train
        .insert_axis(ndarray::Axis(2))
        .into_dimensionality::<Ix3>()
        .unwrap()
        .axis_iter(Axis(0))
        .map(|item| item.into_owned().into_dyn())
        .collect();

    (x_train, y_train)
}

#[derive(serde::Serialize)]
struct Row {
    #[serde(rename = "ImageId")]
    image_id: usize,
    #[serde(rename = "Label")]
    label: usize,
}

fn kaggle_predictions(nn: &mut NeuralNetwork<F>) {
    let x_test = {
        let file = File::open("datasets/kaggle_mnist/test.csv").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        let mut data_train: Array2<F> = reader.deserialize_array2_dynamic().unwrap();

        data_train.map_inplace(|x| *x /= 255.0);
        data_train
    };
    let x_test = x_test
        .insert_axis(ndarray::Axis(2))
        .into_dimensionality::<Ix3>()
        .unwrap();

    let mut predictions: Vec<Row> = Vec::new();
    for (idx, x) in x_test.axis_iter(Axis(0)).enumerate() {
        let out = nn.forward(x.to_owned().into_dyn());
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
