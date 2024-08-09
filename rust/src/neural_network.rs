extern crate blas_src;

use kdam::{term, term::Colorizer, tqdm, BarExt, Column, RichProgress};
use ndarray::{stack, Array1, Array2, Axis};
use rand::seq::SliceRandom;
use std::io::{stderr, IsTerminal};
use std::time::Instant;

use crate::functions::ErrorFunction;
use crate::layer::Layer;
use crate::utils::FloatNN;

pub struct NeuralNetwork<F: FloatNN> {
    pub layers: Vec<Box<dyn Layer<F>>>,
    pub terminal_output: bool,
}

impl<F: FloatNN + for<'a> std::iter::Sum<&'a F>> NeuralNetwork<F> {
    pub fn new(layers: Vec<impl Layer<F> + 'static>) -> Self {
        let layers: Vec<Box<dyn Layer<F>>> = layers
            .into_iter()
            .map(|x| Box::new(x) as Box<dyn Layer<F>>)
            .collect();

        NeuralNetwork {
            layers,
            terminal_output: true,
        }
    }

    pub fn forward(&mut self, x_input: &Array1<F>) -> Array1<F> {
        let mut in_out = x_input;
        for layer in self.layers.iter_mut() {
            layer.forward(in_out);
            in_out = layer.get_output().unwrap();
        }
        in_out.clone()
    }

    pub fn get_layers_gradients(&mut self, error: &Array1<F>) -> Vec<Array2<F>> {
        let mut gradients = Vec::new();
        let mut next_layer = None;

        for layer in self.layers.iter_mut().rev() {
            let gradient = if next_layer.is_some() {
                layer.gradient_descent(None, next_layer)
            } else {
                layer.gradient_descent(Some(error), None)
            };
            gradients.push(gradient);
            next_layer = Some(&**layer);
        }

        gradients
    }

    pub fn update_weights(&mut self, alpha: F, mean_gradients: Vec<Array2<F>>) {
        for (layer, gradient) in self.layers.iter_mut().rev().zip(mean_gradients) {
            layer.update_weights(alpha, &gradient);
        }
    }

    pub fn fit(
        &mut self,
        x_train: Array2<F>,
        y_train: Array2<F>,
        epochs: usize,
        alpha: F,
        batch_size: usize,
        evaluate_step: usize,
    ) {
        let x_train = x_train
            .rows()
            .into_iter()
            .map(|row| row.into_owned())
            .collect::<Vec<Array1<F>>>();
        let y_train = y_train
            .rows()
            .into_iter()
            .map(|row| row.into_owned())
            .collect::<Vec<Array1<F>>>();

        term::init(stderr().is_terminal());
        let mut pb = RichProgress::new(
            tqdm!(total = epochs),
            vec![
                Column::Text("Training...".to_owned()),
                Column::Animation,
                Column::Percentage(1),
                Column::Text("•".to_owned()),
                Column::CountTotal,
                Column::Text("•".to_owned()),
                Column::RemainingTime,
                Column::Text(" ".to_owned()),
            ],
        );
        if self.terminal_output {
            pb.refresh().unwrap();
        }
        let start_time = Instant::now();

        let mut permutation: Vec<usize> = (0..x_train.len()).collect();
        let mut rng = rand::thread_rng();
        for epoch in 1..=epochs {
            permutation.shuffle(&mut rng);

            let mut outputs: Vec<Array1<F>> = Vec::new();
            let mut desired: Vec<&Array1<F>> = Vec::new();

            for window in permutation.chunks(batch_size) {
                let batch_gradients: Vec<Vec<Array2<F>>> = window
                    .iter()
                    .map(|&idx| {
                        let x = unsafe { x_train.get_unchecked(idx) };
                        let y = unsafe { y_train.get_unchecked(idx) };

                        let out = self.forward(x);
                        let error = y - &out;

                        desired.push(y);
                        outputs.push(out);

                        self.get_layers_gradients(&error)
                    })
                    .collect();

                let num_layers = batch_gradients[0].len();
                let mean_gradients: Vec<Array2<F>> = (0..num_layers)
                    .map(|layer_idx| {
                        let gradients = batch_gradients
                            .iter()
                            .map(|gradient| gradient[layer_idx].view())
                            .collect::<Vec<_>>();

                        stack(Axis(0), &gradients)
                            .unwrap()
                            .mean_axis(Axis(0))
                            .unwrap()
                    })
                    .collect();

                self.update_weights(alpha, mean_gradients);
            }

            if self.terminal_output && epoch % evaluate_step == 0 {
                let mse = ErrorFunction::get_mse::<F>(&desired, &outputs);
                let epoch_str = format!("{: >width$}", epoch, width = epochs.to_string().len());
                pb.write(format!(
                    "Epoch: {} | MSE: {:}",
                    epoch_str.to_string().colorize("bold cyan"),
                    mse.to_string().colorize("bold cyan")
                ))
                .unwrap();
            }
            if self.terminal_output {
                pb.update(1).unwrap();
                pb.refresh().unwrap();
            }
        }
        if self.terminal_output {
            pb.refresh().unwrap();
            let end_time = Instant::now();
            let execution_time = end_time.duration_since(start_time);
            eprintln!("{}", pb.pb.fmt_elapsed_time());
            eprintln!("Tempo de execução: {:?}", execution_time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::Sigmoid;
    use crate::layer::Dense;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rstest::*;

    #[fixture]
    fn simple_nn_a() -> (NeuralNetwork<f64>, Array1<f64>, Array1<f64>) {
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

    #[rstest]
    fn test_nn_forward_a(simple_nn_a: (NeuralNetwork<f64>, Array1<f64>, Array1<f64>)) {
        let (mut nn, inputs, _) = simple_nn_a;

        nn.forward(&inputs);

        assert_abs_diff_eq!(
            nn.layers[0].get_net().unwrap(),
            &array![0.3775, 0.3925],
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            nn.layers[0].get_output().unwrap(),
            &array![0.59326999, 0.59688438],
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            nn.layers[1].get_net().unwrap(),
            &array![1.10590597, 1.2249214],
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            nn.layers[1].get_output().unwrap(),
            &array![0.75136507, 0.77292847],
            epsilon = 1e-8
        );
    }

    #[rstest]
    fn test_nn_backward_a(simple_nn_a: (NeuralNetwork<f64>, Array1<f64>, Array1<f64>)) {
        let (mut nn, inputs, desired) = simple_nn_a;

        let x_train = inputs.insert_axis(ndarray::Axis(0));
        let y_train = desired.insert_axis(ndarray::Axis(0));

        nn.fit(x_train, y_train, 1, 0.5, 1, 10);

        assert_abs_diff_eq!(
            nn.layers[0].get_weights(),
            &array![[0.14978072, 0.19956143], [0.24975114, 0.29950229]],
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            nn.layers[1].get_weights(),
            &array![[0.35891648, 0.408666186], [0.511301270, 0.561370121]],
            epsilon = 1e-9
        );
    }
}
