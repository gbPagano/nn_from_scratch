extern crate blas_src;

use ndarray::{arr2, Array2, ArrayD, Ix2, Zip};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use super::super::Float;
use super::Layer;

pub struct Dense<F: Float> {
    pub weights: Array2<F>,
    pub bias: Array2<F>,
    input: Array2<F>,
    curr_batch: usize,
    weights_gradient: Array2<F>,
    bias_gradient: Array2<F>,
}

impl<F: Float> Dense<F> {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        // He/Kaiming initialization: std = sqrt(2 / fan_in), suited to ReLU/ELU.
        let std = (2.0 / inputs as f32).sqrt();
        let weights = Array2::random((outputs, inputs), Normal::new(0.0, std).unwrap())
            .mapv(|v: f32| F::from_f32(v).unwrap());
        let bias = Array2::zeros((outputs, 1));
        Dense {
            weights,
            bias,
            input: arr2(&[[]]),
            curr_batch: 0,
            weights_gradient: arr2(&[[]]),
            bias_gradient: arr2(&[[]]),
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

        // gradient descent as optimizer
        self.curr_batch += 1;
        if self.curr_batch == batch_size {
            Zip::from(&mut self.weights)
                .and(&self.weights_gradient)
                .for_each(|a, &b| *a -= b * learning_rate / F::from_usize(batch_size).unwrap());

            Zip::from(&mut self.bias)
                .and(&self.bias_gradient)
                .for_each(|a, &b| *a -= b * learning_rate / F::from_usize(batch_size).unwrap());

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
}

impl<F: Float> From<Dense<F>> for Box<dyn Layer<F>> {
    fn from(item: Dense<F>) -> Self {
        Box::new(item)
    }
}
