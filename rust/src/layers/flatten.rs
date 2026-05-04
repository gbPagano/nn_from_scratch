extern crate blas_src;

use ndarray::*;

use super::super::Float;
use super::Layer;

pub struct Flatten {
    input_shape: (usize, usize, usize),
    flat_size: usize,
}

impl Flatten {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        let flat_size = input_shape.0 * input_shape.1 * input_shape.2;
        Self {
            input_shape,
            flat_size,
        }
    }
}

impl<F: Float> Layer<F> for Flatten {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        let b = input.shape()[0];
        input.into_shape((b, self.flat_size)).unwrap().into_dyn()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let b = output_gradient.shape()[0];
        let (c, h, w) = self.input_shape;
        output_gradient.into_shape((b, c, h, w)).unwrap().into_dyn()
    }

    fn get_weights(&self) -> Option<ArrayD<F>> {
        None
    }

    fn get_bias(&self) -> Option<ArrayD<F>> {
        None
    }
}

impl<F: Float> From<Flatten> for Box<dyn Layer<F>> {
    fn from(item: Flatten) -> Self {
        Box::new(item)
    }
}
