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
    let (x_train, y_train) = load_mnist_dataset("datasets/kaggle_mnist/train_split.csv");
    let (x_val, y_val) = load_mnist_dataset("datasets/kaggle_mnist/val_split.csv");
    println!(
        "Loaded {} train samples and {} validation samples.",
        x_train.len(),
        x_val.len()
    );

    let mut nn: NeuralNetwork<F> = NeuralNetwork::new(box_layers![
        Conv::new((1, 28, 28), 8, 3),
        ELU::new(1.0 as F),
        MaxPooling::new((8, 26, 26), 2, 2),
        Conv::new((8, 13, 13), 16, 3),
        ELU::new(1.0 as F),
        MaxPooling::new((16, 11, 11), 2, 2),
        Flatten::new((16, 6, 6)),
        Dense::new(16 * 6 * 6, 64),
        ELU::new(1.0 as F),
        Dense::new(64, 10),
        SoftmaxCE::new()
    ]);
    nn.fit_with_validation(
        &x_train,
        &y_train,
        Some((&x_val, &y_val)),
        NNConfig {
            epochs: 20,
            learning_rate: 0.001,
            batch_size: 32,
            evaluate_step: 1,
            loss_function: CrossEntropySoftmax::new().into(),
            optimizer: OptimizerConfig::adam_default(),
        },
    );
    kaggle_predictions(&mut nn);
}

fn number_to_neurons<F: Float>(n: usize, negative_output: F, positive_output: F) -> Vec<F> {
    let mut res = vec![negative_output; 10];
    res[n] = positive_output;
    res
}

fn load_mnist_dataset(path: &str) -> (Vec<ArrayD<F>>, Vec<ArrayD<F>>) {
    let file = File::open(path).unwrap();
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

    let n_samples = x_train.shape()[0];
    let x_train = x_train
        .into_shape((n_samples, 1, 28, 28))
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
    let n_test = x_test.shape()[0];
    let x_test = x_test.into_shape((n_test, 1, 28, 28)).unwrap();

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
