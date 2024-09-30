extern crate blas_src;

use ndarray::*;

use super::super::Float;
use super::Layer;

pub struct Flatten {
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize),
}

impl Flatten {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        let output_shape = (input_shape.0 * input_shape.1 * input_shape.2, 1);
        Self {
            input_shape,
            output_shape,
        }
    }
}

impl<F: Float> Layer<F> for Flatten {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        input.into_shape(self.output_shape).unwrap().into_dyn()
    }

    fn backward(
        &mut self,
        output_gradient: ArrayD<F>,
        _learning_rate: F,
        _batch_size: usize,
    ) -> ArrayD<F> {
        output_gradient
            .into_shape(self.input_shape)
            .unwrap()
            .into_dyn()
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
