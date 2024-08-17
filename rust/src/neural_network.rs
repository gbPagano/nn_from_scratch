extern crate blas_src;

use kdam::{term, term::Colorizer, tqdm, BarExt, Column, RichProgress};
use ndarray::{Array3, ArrayD, Axis};
use rand::seq::SliceRandom;
use std::io::{stderr, IsTerminal};

use super::layers::Layer;
use super::loss::{HalfMSE, Loss};
use super::Float;

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

    fn backward(&mut self, mut grad: ArrayD<F>, learning_rate: F, batch_size: usize) {
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(grad, learning_rate, batch_size);
        }
    }

    pub fn fit(&mut self, x_train: &Array3<F>, y_train: &Array3<F>, config: NNConfig<F>) {
        let x_train: Vec<ArrayD<_>> = x_train
            .axis_iter(Axis(0))
            .map(|item| item.into_owned().into_dyn())
            .collect();
        let y_train: Vec<ArrayD<_>> = y_train
            .axis_iter(Axis(0))
            .map(|item| item.into_owned().into_dyn())
            .collect();

        term::init(stderr().is_terminal());
        let mut pb = self.get_bar(config.epochs);
        if self.terminal_output {
            pb.update(0).unwrap();
        }

        let mut permutation: Vec<usize> = (0..x_train.len()).collect();
        let mut rng = rand::thread_rng();
        for epoch in 1..=config.epochs {
            let mut loss = F::from_f32(0.0).unwrap();
            permutation.shuffle(&mut rng);
            for &idx in permutation.iter() {
                let x = unsafe { x_train.get_unchecked(idx) };
                let y = unsafe { y_train.get_unchecked(idx) };

                let out = self.forward(x.clone());
                loss += config.loss_function.loss(y, &out);
                let grad = config.loss_function.gradient(y, &out);
                self.backward(grad, config.learning_rate, config.batch_size);
            }
            loss /= F::from_usize(x_train.len()).unwrap();
            if self.terminal_output && epoch % config.evaluate_step == 0 {
                let epoch_str = format!(
                    "{: >width$}",
                    epoch,
                    width = config.epochs.to_string().len()
                );
                pb.write(format!(
                    "Epoch: {} | Loss: {:}",
                    epoch_str.to_string().colorize("bold cyan"),
                    loss.to_string().colorize("bold cyan")
                ))
                .unwrap();
            }
            if self.terminal_output {
                pb.update(1).unwrap();
            }
        }
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

pub struct NNConfig<F: Float> {
    pub epochs: usize,
    pub learning_rate: F,
    pub batch_size: usize,
    pub evaluate_step: usize,
    pub loss_function: Box<dyn Loss<F>>,
}

impl<F: Float> NNConfig<F> {
    pub fn new(
        epochs: usize,
        learning_rate: F,
        batch_size: usize,
        evaluate_step: usize,
        loss_function: Box<dyn Loss<F>>,
    ) -> Self {
        NNConfig {
            epochs,
            learning_rate,
            batch_size,
            evaluate_step,
            loss_function,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::layers::activation::*;
    use super::super::layers::*;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Ix3};
    use rstest::*;

    #[fixture]
    fn simple_nn() -> (NeuralNetwork<'static, f64>, ArrayD<f64>, ArrayD<f64>) {
        let mut layer_1 = Dense::new(2, 2);
        layer_1.weights = array![[0.15, 0.2], [0.25, 0.3]];
        layer_1.bias = array![[0.35], [0.35]];

        let layer_2 = Sigmoid::new();

        let mut layer_3 = Dense::new(2, 2);
        layer_3.weights = array![[0.4, 0.45], [0.5, 0.55]];
        layer_3.bias = array![[0.6], [0.6]];

        let layer_4 = Sigmoid::new();

        let inputs = array![[0.05], [0.1]].into_dyn();
        let desired = array![[0.01], [0.99]].into_dyn();
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
            array![[0.75136507], [0.77292847]].into_dyn(),
            epsilon = 1e-8
        );
    }

    #[rstest]
    fn test_nn_backward(simple_nn: (NeuralNetwork<f64>, ArrayD<f64>, ArrayD<f64>)) {
        let (mut nn, inputs, desired) = simple_nn;

        let x_train = inputs
            .insert_axis(ndarray::Axis(0))
            .into_dimensionality::<Ix3>()
            .unwrap();
        let y_train = desired
            .insert_axis(ndarray::Axis(0))
            .into_dimensionality::<Ix3>()
            .unwrap();

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

        let x_train = array![[[0.05], [0.1]], [[0.05], [0.1]]];
        let y_train = array![[[0.01], [0.99]], [[0.01], [0.99]]];

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
