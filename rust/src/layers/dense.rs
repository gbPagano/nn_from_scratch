extern crate blas_src;

use ndarray::{Array1, Array2, ArrayD, Axis, Ix1, Ix2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use super::super::optimizer::{Optimizer, OptimizerConfig, SGD};
use super::super::Float;
use super::Layer;

pub struct Dense<F: Float> {
    pub weights: Array2<F>,
    pub bias: Array1<F>,
    input: Array2<F>,
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
        let bias = Array1::zeros(outputs);
        Dense {
            weights,
            bias,
            input: Array2::zeros((0, 0)),
            weights_optimizer: Box::new(SGD),
            bias_optimizer: Box::new(SGD),
        }
    }
}

impl<F: Float> Layer<F> for Dense<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.input = input.into_dimensionality::<Ix2>().unwrap();
        let output = self.input.dot(&self.weights.t()) + &self.bias;
        output.into_dyn()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, learning_rate: F) -> ArrayD<F> {
        let output_gradient = output_gradient.into_dimensionality::<Ix2>().unwrap();
        let b = F::from_usize(output_gradient.shape()[0]).unwrap();
        let weights_gradient = output_gradient.t().dot(&self.input) / b;
        let bias_gradient = output_gradient.sum_axis(Axis(0)) / b;
        let input_gradient = output_gradient.dot(&self.weights);

        self.weights_optimizer.step(
            self.weights.view_mut().into_dyn(),
            weights_gradient.view().into_dyn(),
            learning_rate,
        );
        self.bias_optimizer.step(
            self.bias.view_mut().into_dyn(),
            bias_gradient.view().into_dyn(),
            learning_rate,
        );

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
        self.bias = bias.into_dimensionality::<Ix1>().unwrap();
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
