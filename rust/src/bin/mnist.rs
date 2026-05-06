extern crate blas_src;

use clap::{ArgAction, Parser, ValueEnum};
use csv::{ReaderBuilder, WriterBuilder};
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use num_traits::Float;
use std::fs::File;

use chrono::Local;
use ndarray_rand::rand::SeedableRng;
use rand::rngs::StdRng;

use nn_from_scratch::layers::activation::*;
use nn_from_scratch::layers::*;
use nn_from_scratch::loss::*;
use nn_from_scratch::*;

type F = f32;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, conflicts_with = "predict")]
    save: Option<String>,
    #[arg(long, conflicts_with = "save")]
    predict: Option<String>,
    #[arg(long, default_value = "datasets/kaggle_mnist/train_split.csv")]
    train_path: String,
    #[arg(long, default_value = "datasets/kaggle_mnist/val_split.csv")]
    val_path: String,
    #[arg(long, default_value = "datasets/kaggle_mnist/test.csv")]
    test_path: String,
    #[arg(long, default_value_t = default_prediction_path())]
    prediction_path: String,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 5)]
    early_stopping_patience: usize,
    #[arg(long, value_enum, default_value_t = CliEarlyStoppingMetric::Loss)]
    early_stopping_metric: CliEarlyStoppingMetric,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    restore_best_weights: bool,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliEarlyStoppingMetric {
    Loss,
    Accuracy,
}

impl From<CliEarlyStoppingMetric> for EarlyStoppingMetric {
    fn from(metric: CliEarlyStoppingMetric) -> Self {
        match metric {
            CliEarlyStoppingMetric::Loss => EarlyStoppingMetric::ValidationLoss,
            CliEarlyStoppingMetric::Accuracy => EarlyStoppingMetric::ValidationAccuracy,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let mut nn = build_network(cli.seed);

    if let Some(checkpoint_path) = cli.predict {
        nn.load(&checkpoint_path);
    } else {
        let (x_train, y_train) = load_mnist_dataset(&cli.train_path);
        let (x_val, y_val) = load_mnist_dataset(&cli.val_path);
        println!(
            "Loaded {} train samples and {} validation samples.",
            x_train.len(),
            x_val.len()
        );

        let training_summary = nn.fit_with_validation(
            &x_train,
            &y_train,
            Some((&x_val, &y_val)),
            NNConfig {
                epochs: cli.epochs,
                learning_rate: 0.0001,
                batch_size: 32,
                evaluate_step: 1,
                loss_function: CrossEntropySoftmax::new().into(),
                optimizer: OptimizerConfig::adam_default(),
                seed: cli.seed,
                early_stopping: (cli.early_stopping_patience > 0).then(|| {
                    EarlyStoppingConfig::new(
                        cli.early_stopping_metric.into(),
                        cli.early_stopping_patience,
                        cli.restore_best_weights,
                    )
                }),
            },
        );
        log_training_summary(&training_summary);

        if let Some(save_path) = cli.save {
            nn.save(&save_path);
        }
    }

    kaggle_predictions(&mut nn, &cli.test_path, &cli.prediction_path);
}

fn log_training_summary(summary: &TrainingSummary<F>) {
    println!(
        "Training finished after {} epoch(s).",
        summary.epochs_trained
    );

    if summary.stopped_early {
        println!("Early stopping triggered.");
    }

    if let (Some(best_epoch), Some(best_val_accuracy), Some(best_val_loss)) = (
        summary.best_epoch,
        summary.best_validation_accuracy,
        summary.best_validation_loss,
    ) {
        println!(
            "Best validation checkpoint: epoch {} | loss {:.8} | accuracy {:.4}",
            best_epoch, best_val_loss, best_val_accuracy
        );
    }
}

fn default_prediction_path() -> String {
    format!(
        "kaggle-submission-{}.csv",
        Local::now().format("%Y%m%d%H%M")
    )
}

fn build_network(seed: Option<u64>) -> NeuralNetwork<'static, F> {
    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        NeuralNetwork::new(box_layers![
            Conv::new_with_rng((1, 28, 28), 32, 3, &mut rng),
            ELU::new(1.0 as F),
            MaxPooling::new((32, 26, 26), 2, 2),
            Conv::new_with_rng((32, 13, 13), 64, 3, &mut rng),
            ELU::new(1.0 as F),
            MaxPooling::new((64, 11, 11), 2, 2),
            Flatten::new((64, 6, 6)),
            Dense::new_with_rng(64 * 6 * 6, 128, &mut rng),
            ELU::new(1.0 as F),
            Dense::new_with_rng(128, 10, &mut rng),
            SoftmaxCE::new()
        ])
    } else {
        NeuralNetwork::new(box_layers![
            Conv::new((1, 28, 28), 32, 3),
            ELU::new(1.0 as F),
            MaxPooling::new((32, 26, 26), 2, 2),
            Conv::new((32, 13, 13), 64, 3),
            ELU::new(1.0 as F),
            MaxPooling::new((64, 11, 11), 2, 2),
            Flatten::new((64, 6, 6)),
            Dense::new(64 * 6 * 6, 128),
            ELU::new(1.0 as F),
            Dense::new(128, 10),
            SoftmaxCE::new()
        ])
    }
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

fn kaggle_predictions(nn: &mut NeuralNetwork<F>, test_path: &str, prediction_path: &str) {
    let x_test = {
        let file = File::open(test_path).unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        let mut data_train: Array2<F> = reader.deserialize_array2_dynamic().unwrap();

        data_train.map_inplace(|x| *x /= 255.0);
        data_train
    };
    let n_test = x_test.shape()[0];
    let x_test = x_test.into_shape((n_test, 1, 28, 28)).unwrap();

    let batch_size = 64;
    let mut predictions: Vec<Row> = Vec::with_capacity(n_test);
    for chunk_start in (0..n_test).step_by(batch_size) {
        let end = (chunk_start + batch_size).min(n_test);
        let batch = x_test
            .slice(s![chunk_start..end, .., .., ..])
            .to_owned()
            .into_dyn();
        let out = nn.forward(batch);
        let out2 = out.view().into_dimensionality::<Ix2>().unwrap();
        for row in out2.outer_iter() {
            let label = row
                .iter()
                .enumerate()
                .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            predictions.push(Row {
                image_id: predictions.len() + 1,
                label,
            });
        }
    }

    {
        let file = File::create(prediction_path).unwrap();
        let mut writer = WriterBuilder::new().from_writer(file);

        for row in predictions {
            writer.serialize(row).unwrap();
        }
    }
}
