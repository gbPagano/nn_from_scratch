extern crate blas_src;

use ndarray::{arr2, Array2, ArrayD, Ix2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use super::super::optimizer::{Optimizer, OptimizerConfig, SGD};
use super::super::Float;
use super::Layer;

pub struct Dense<F: Float> {
    pub weights: Array2<F>,
    pub bias: Array2<F>,
    input: Array2<F>,
    curr_batch: usize,
    weights_gradient: Array2<F>,
    bias_gradient: Array2<F>,
    weights_optimizer: Box<dyn Optimizer<F>>,
    bias_optimizer: Box<dyn Optimizer<F>>,
}

impl<F: Float> Dense<F> {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        // He/Kaiming initialization: std = sqrt(2 / fan_in), suited to ReLU/ELU.
        let std = (2.0 / inputs as f32).sqrt();
        let weights = Array2::random((outputs, inputs), Normal::new(0.0, std).unwrap())
            .mapv(|v: f32| F::from_f32(v).unwrap());
        Self::with_weights(weights)
    }

    pub fn new_with_rng<R: Rng>(inputs: usize, outputs: usize, rng: &mut R) -> Self {
        // He/Kaiming initialization: std = sqrt(2 / fan_in), suited to ReLU/ELU.
        let std = (2.0 / inputs as f32).sqrt();
        let weights = Array2::random_using((outputs, inputs), Normal::new(0.0, std).unwrap(), rng)
            .mapv(|v: f32| F::from_f32(v).unwrap());
        Self::with_weights(weights)
    }

    fn with_weights(weights: Array2<F>) -> Self {
        let outputs = weights.shape()[0];
        let bias = Array2::zeros((outputs, 1));
        Dense {
            weights,
            bias,
            input: arr2(&[[]]),
            curr_batch: 0,
            weights_gradient: arr2(&[[]]),
            bias_gradient: arr2(&[[]]),
            weights_optimizer: Box::new(SGD),
            bias_optimizer: Box::new(SGD),
        }
    }
}

impl<F: Float> Layer<F> for Dense<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.input = input.into_dimensionality::<Ix2>().unwrap();
        let output = self.weights.dot(&self.input) + &self.bias;
        output.into_dyn()
    }

    fn backward(
        &mut self,
        output_gradient: ArrayD<F>,
        learning_rate: F,
        batch_size: usize,
    ) -> ArrayD<F> {
        let output_gradient = output_gradient.into_dimensionality::<Ix2>().unwrap();
        let weights_gradient = output_gradient.dot(&self.input.t());
        let input_gradient = self.weights.t().dot(&output_gradient);

        if self.curr_batch == 0 {
            self.weights_gradient = weights_gradient;
            self.bias_gradient = output_gradient;
        } else {
            self.weights_gradient += &weights_gradient;
            self.bias_gradient += &output_gradient;
        }

        self.curr_batch += 1;
        if self.curr_batch == batch_size {
            let inv_batch = F::from_f32(1.0).unwrap() / F::from_usize(batch_size).unwrap();
            self.weights_gradient *= inv_batch;
            self.bias_gradient *= inv_batch;

            self.weights_optimizer.step(
                self.weights.view_mut().into_dyn(),
                self.weights_gradient.view().into_dyn(),
                learning_rate,
            );
            self.bias_optimizer.step(
                self.bias.view_mut().into_dyn(),
                self.bias_gradient.view().into_dyn(),
                learning_rate,
            );

            self.curr_batch = 0;
        }

        input_gradient.into_dyn()
    }

    fn get_weights(&self) -> Option<ArrayD<F>> {
        Some(self.weights.clone().into_dyn())
    }

    fn get_bias(&self) -> Option<ArrayD<F>> {
        Some(self.bias.clone().into_dyn())
    }

    fn set_weights(&mut self, weights: ArrayD<F>) {
        self.weights = weights.into_dimensionality::<Ix2>().unwrap();
    }

    fn set_bias(&mut self, bias: ArrayD<F>) {
        self.bias = bias.into_dimensionality::<Ix2>().unwrap();
    }

    fn set_optimizer(&mut self, config: &OptimizerConfig<F>) {
        self.weights_optimizer = config.build();
        self.bias_optimizer = config.build();
    }
}

impl<F: Float> From<Dense<F>> for Box<dyn Layer<F>> {
    fn from(item: Dense<F>) -> Self {
        Box::new(item)
    }
}
