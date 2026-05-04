extern crate blas_src;

use kdam::{term, term::Colorizer, tqdm, BarExt, Column, RichProgress};
use ndarray::{ArrayD, ArrayView1, Axis, Ix2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{stderr, IsTerminal};

use super::layers::{Layer, LayerParameters};
use super::loss::{HalfMSE, Loss};
use super::optimizer::OptimizerConfig;
use super::Float;

#[derive(Serialize, Deserialize)]
struct NetworkCheckpoint<F> {
    layers: Vec<LayerParameters<F>>,
}

pub struct NeuralNetwork<'a, F: Float> {
    pub layers: Vec<Box<dyn Layer<F> + 'a>>,
    pub terminal_output: bool,
}

impl<'a, F: Float> NeuralNetwork<'a, F> {
    pub fn new(layers: Vec<Box<dyn Layer<F>>>) -> Self {
        NeuralNetwork {
            layers,
            terminal_output: true,
        }
    }

    pub fn forward(&mut self, x_input: ArrayD<F>) -> ArrayD<F> {
        let mut out = x_input;
        for layer in self.layers.iter_mut() {
            out = layer.forward(out);
        }
        out
    }

    fn backward(&mut self, mut grad: ArrayD<F>, learning_rate: F) {
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(grad, learning_rate);
        }
    }

    pub fn fit(&mut self, x_train: &[ArrayD<F>], y_train: &[ArrayD<F>], config: NNConfig<F>) {
        self.fit_with_validation(x_train, y_train, None, config)
    }

    pub fn save(&self, path: &str)
    where
        F: Serialize,
    {
        let checkpoint = NetworkCheckpoint {
            layers: self
                .layers
                .iter()
                .map(|layer| LayerParameters {
                    weights: layer.get_weights(),
                    bias: layer.get_bias(),
                })
                .collect(),
        };
        let file = File::create(path).unwrap();
        bincode::serialize_into(file, &checkpoint).unwrap();
    }

    pub fn load(&mut self, path: &str)
    where
        F: for<'de> Deserialize<'de>,
    {
        let file = File::open(path).unwrap();
        let checkpoint: NetworkCheckpoint<F> = bincode::deserialize_from(file).unwrap();
        assert_eq!(self.layers.len(), checkpoint.layers.len());

        for (layer, parameters) in self.layers.iter_mut().zip(checkpoint.layers.into_iter()) {
            if let Some(weights) = parameters.weights {
                layer.set_weights(weights);
            }
            if let Some(bias) = parameters.bias {
                layer.set_bias(bias);
            }
        }
    }

    pub fn fit_with_validation(
        &mut self,
        x_train: &[ArrayD<F>],
        y_train: &[ArrayD<F>],
        validation: Option<(&[ArrayD<F>], &[ArrayD<F>])>,
        config: NNConfig<F>,
    ) {
        term::init(stderr().is_terminal());
        let mut pb = self.get_bar(config.epochs);
        if self.terminal_output {
            pb.update(0).unwrap();
        }

        for layer in self.layers.iter_mut() {
            layer.set_optimizer(&config.optimizer);
        }

        let mut permutation: Vec<usize> = (0..x_train.len()).collect();
        let mut seeded_rng;
        let mut thread_rng;
        let rng: &mut dyn RngCore = if let Some(seed) = config.seed {
            seeded_rng = StdRng::seed_from_u64(seed);
            &mut seeded_rng
        } else {
            thread_rng = rand::thread_rng();
            &mut thread_rng
        };

        for epoch in 1..=config.epochs {
            permutation.shuffle(&mut *rng);
            for batch_idx in permutation.chunks(config.batch_size) {
                let x_batch = stack_views(x_train, batch_idx);
                let y_batch = stack_views(y_train, batch_idx);

                let out = self.forward(x_batch);
                let grad = config.loss_function.gradient(&y_batch, &out);
                self.backward(grad, config.learning_rate);
            }
            if self.terminal_output && epoch % config.evaluate_step == 0 {
                let (train_accuracy, train_loss) = self.evaluate(
                    x_train,
                    y_train,
                    config.loss_function.as_ref(),
                    config.batch_size,
                );
                let val_str = match validation {
                    Some((x_val, y_val)) => {
                        let (val_accuracy, val_loss) = self.evaluate(
                            x_val,
                            y_val,
                            config.loss_function.as_ref(),
                            config.batch_size,
                        );
                        format!(
                            " | Val Loss: {} | Val Accuracy: {}",
                            format!("{:.8}", val_loss).to_string().colorize("bold blue"),
                            format!("{:.4}", val_accuracy)
                                .to_string()
                                .colorize("bold blue"),
                        )
                    }
                    None => String::new(),
                };
                pb.write(format!(
                    "Epoch: {} | Train Loss: {} | Train Accuracy: {}{}",
                    format!(
                        "{: >width$}",
                        epoch,
                        width = config.epochs.to_string().len()
                    )
                    .colorize("bold cyan"),
                    format!("{:.8}", train_loss)
                        .to_string()
                        .colorize("bold cyan"),
                    format!("{:.4}", train_accuracy)
                        .to_string()
                        .colorize("bold cyan"),
                    val_str,
                ))
                .unwrap();
            }
            if self.terminal_output {
                pb.update(1).unwrap();
            }
        }
    }

    pub fn evaluate(
        &mut self,
        x_vec: &[ArrayD<F>],
        y_vec: &[ArrayD<F>],
        loss_fn: &dyn Loss<F>,
        batch_size: usize,
    ) -> (f64, F) {
        let total = y_vec.len();
        let mut correct = 0;
        let mut loss_acc = F::from_f32(0.0).unwrap();
        let indices: Vec<usize> = (0..total).collect();
        for batch_idx in indices.chunks(batch_size) {
            let x_batch = stack_views(x_vec, batch_idx);
            let y_batch = stack_views(y_vec, batch_idx);
            let out = self.forward(x_batch);

            let batch_loss = loss_fn.loss(&y_batch, &out);
            loss_acc += batch_loss * F::from_usize(batch_idx.len()).unwrap();

            let out2 = out.view().into_dimensionality::<Ix2>().unwrap();
            let y2 = y_batch.view().into_dimensionality::<Ix2>().unwrap();
            for (out_row, y_row) in out2.outer_iter().zip(y2.outer_iter()) {
                if argmax(&out_row) == argmax(&y_row) {
                    correct += 1;
                }
            }
        }
        let accuracy = correct as f64 / total as f64;
        let avg_loss = loss_acc / F::from_usize(total).unwrap();
        (accuracy, avg_loss)
    }

    fn get_bar(&self, total: usize) -> RichProgress {
        RichProgress::new(
            tqdm!(total = total, ncols = 40, force_refresh = true),
            vec![
                Column::Text("Training...".to_owned()),
                Column::Animation,
                Column::Percentage(1),
                Column::Text("•".to_owned()),
                Column::CountTotal,
                Column::Text("•".to_owned()),
                Column::ElapsedTime,
                Column::Text("<".to_owned()),
                Column::RemainingTime,
                Column::Text(" ".to_owned()),
            ],
        )
    }
}

fn stack_views<F: Float>(samples: &[ArrayD<F>], indices: &[usize]) -> ArrayD<F> {
    let views: Vec<_> = indices.iter().map(|&i| samples[i].view()).collect();
    ndarray::stack(Axis(0), &views).unwrap()
}

fn argmax<F: Float>(row: &ArrayView1<F>) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

pub struct NNConfig<F: Float> {
    pub epochs: usize,
    pub learning_rate: F,
    pub batch_size: usize,
    pub evaluate_step: usize,
    pub loss_function: Box<dyn Loss<F>>,
    pub optimizer: OptimizerConfig<F>,
    pub seed: Option<u64>,
}

impl<F: Float> NNConfig<F> {
    pub fn new(
        epochs: usize,
        learning_rate: F,
        batch_size: usize,
        evaluate_step: usize,
        loss_function: Box<dyn Loss<F>>,
        optimizer: OptimizerConfig<F>,
    ) -> Self {
        NNConfig {
            epochs,
            learning_rate,
            batch_size,
            evaluate_step,
            loss_function,
            optimizer,
            seed: None,
        }
    }
}
impl<F: Float> Default for NNConfig<F> {
    fn default() -> Self {
        NNConfig {
            epochs: 1,
            learning_rate: F::from_f32(0.5).unwrap(),
            batch_size: 1,
            evaluate_step: 10,
            loss_function: HalfMSE::new().into(),
            optimizer: OptimizerConfig::SGD,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::layers::activation::*;
    use super::super::layers::*;
    use super::*;
    use crate::box_layers;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Axis};
    use rstest::*;
    use std::env::temp_dir;
    use std::fs;

    type F = f64;

    #[fixture]
    fn simple_nn() -> (NeuralNetwork<'static, f64>, ArrayD<f64>, ArrayD<f64>) {
        let mut layer_1 = Dense::new(2, 2);
        layer_1.weights = array![[0.15, 0.2], [0.25, 0.3]];
        layer_1.bias = array![0.35, 0.35];

        let layer_2 = Sigmoid::new();

        let mut layer_3 = Dense::new(2, 2);
        layer_3.weights = array![[0.4, 0.45], [0.5, 0.55]];
        layer_3.bias = array![0.6, 0.6];

        let layer_4 = Sigmoid::new();

        let inputs = array![[0.05, 0.1]].into_dyn();
        let desired = array![[0.01, 0.99]].into_dyn();
        let nn = NeuralNetwork {
            layers: vec![
                Box::new(layer_1),
                Box::new(layer_2),
                Box::new(layer_3),
                Box::new(layer_4),
            ],
            terminal_output: false,
        };

        (nn, inputs, desired)
    }

    #[rstest]
    fn test_nn_forward(simple_nn: (NeuralNetwork<f64>, ArrayD<f64>, ArrayD<f64>)) {
        let (mut nn, inputs, _) = simple_nn;

        let out = nn.forward(inputs);

        assert_abs_diff_eq!(
            out,
            array![[0.75136507, 0.77292847]].into_dyn(),
            epsilon = 1e-8
        );
    }

    #[rstest]
    fn test_save_load_preserves_predictions_dense_network(
        simple_nn: (NeuralNetwork<'static, f64>, ArrayD<f64>, ArrayD<f64>),
    ) {
        let (mut trained_nn, inputs, _) = simple_nn;
        let mut loaded_nn = NeuralNetwork::new(box_layers![
            Dense::new(2, 2),
            Sigmoid::new(),
            Dense::new(2, 2),
            Sigmoid::new()
        ]);
        let path = temp_dir().join("nn_from_scratch_test_checkpoint.bin");
        let path = path.to_str().unwrap();

        trained_nn.save(path);
        loaded_nn.load(path);

        let expected = trained_nn.forward(inputs.clone());
        let actual = loaded_nn.forward(inputs);
        fs::remove_file(path).unwrap();

        assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_checkpoint_load_skips_non_trainable_layers() {
        let mut source = NeuralNetwork::new(box_layers![
            Dense::new(2, 2),
            Sigmoid::new(),
            Dense::new(2, 1)
        ]);
        let mut target = NeuralNetwork::new(box_layers![
            Dense::new(2, 2),
            Sigmoid::new(),
            Dense::new(2, 1)
        ]);
        let path = temp_dir().join("nn_from_scratch_test_checkpoint_with_activation.bin");
        let path = path.to_str().unwrap();

        source.layers[0].set_weights(array![[0.1, 0.2], [0.3, 0.4]].into_dyn());
        source.layers[0].set_bias(array![0.5, 0.6].into_dyn());
        source.layers[2].set_weights(array![[0.7, 0.8]].into_dyn());
        source.layers[2].set_bias(array![0.9].into_dyn());

        source.save(path);
        target.load(path);
        fs::remove_file(path).unwrap();

        assert_abs_diff_eq!(
            target.layers[0].get_weights().unwrap(),
            array![[0.1, 0.2], [0.3, 0.4]].into_dyn(),
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            target.layers[2].get_weights().unwrap(),
            array![[0.7, 0.8]].into_dyn(),
            epsilon = 1e-8
        );
    }

    #[test]
    #[should_panic]
    fn test_load_rejects_mismatched_layer_count() {
        let source = NeuralNetwork::<f64>::new(box_layers![Dense::new(2, 2), Sigmoid::new()]);
        let mut target = NeuralNetwork::<f64>::new(box_layers![Dense::new(2, 2)]);
        let path = temp_dir().join("nn_from_scratch_test_checkpoint_mismatch.bin");
        let path = path.to_str().unwrap();

        source.save(path);
        target.load(path);
    }

    #[rstest]
    fn test_nn_backward(simple_nn: (NeuralNetwork<f64>, ArrayD<f64>, ArrayD<f64>)) {
        let (mut nn, inputs, desired) = simple_nn;

        let x_train = vec![inputs.index_axis(Axis(0), 0).to_owned().into_dyn()];
        let y_train = vec![desired.index_axis(Axis(0), 0).to_owned().into_dyn()];

        nn.fit(&x_train, &y_train, NNConfig::default());

        assert_abs_diff_eq!(
            nn.layers[0].get_weights().unwrap(),
            array![[0.14978072, 0.19956143], [0.24975114, 0.29950229]].into_dyn(),
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            nn.layers[2].get_weights().unwrap(),
            array![[0.35891648, 0.408666186], [0.511301270, 0.561370121]].into_dyn(),
            epsilon = 1e-9
        );
    }

    #[rstest]
    fn test_nn_backward_minibatch(simple_nn: (NeuralNetwork<f64>, ArrayD<f64>, ArrayD<f64>)) {
        let (mut nn, _, _) = simple_nn;

        let x_train = array![[0.05, 0.1], [0.05, 0.1]];
        let y_train = array![[0.01, 0.99], [0.01, 0.99]];
        let x_train: Vec<ArrayD<_>> = x_train
            .axis_iter(Axis(0))
            .map(|item| item.into_owned().into_dyn())
            .collect();
        let y_train: Vec<ArrayD<_>> = y_train
            .axis_iter(Axis(0))
            .map(|item| item.into_owned().into_dyn())
            .collect();

        let mut config = NNConfig::default();
        config.batch_size = 2;
        nn.fit(&x_train, &y_train, config);

        assert_abs_diff_eq!(
            nn.layers[0].get_weights().unwrap(),
            array![[0.14978072, 0.19956143], [0.24975114, 0.29950229]].into_dyn(),
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            nn.layers[2].get_weights().unwrap(),
            array![[0.35891648, 0.408666186], [0.511301270, 0.561370121]].into_dyn(),
            epsilon = 1e-9
        );
    }
}
