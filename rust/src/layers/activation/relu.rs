use ndarray::{array, ArrayD};

use super::Float;
use super::Layer;

pub struct ReLU<F: Float> {
    input: ArrayD<F>,
}

impl<F: Float> ReLU<F> {
    pub fn new() -> Self {
        Self {
            input: array![[]].into_dyn(),
        }
    }
    pub fn activate(&self, array: &ArrayD<F>) -> ArrayD<F> {
        array.mapv(|x| {
            if x >= F::from_f32(0.0).unwrap() {
                x
            } else {
                F::from_f32(0.0).unwrap()
            }
        })
    }
    pub fn derivative(&self, array: &ArrayD<F>) -> ArrayD<F> {
        array.mapv(|x| {
            if x >= F::from_f32(0.0).unwrap() {
                F::from_f32(1.0).unwrap()
            } else {
                F::from_f32(0.0).unwrap()
            }
        })
    }
}
impl<F: Float> Default for ReLU<F> {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Float> Layer<F> for ReLU<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.input = input;
        self.activate(&self.input)
    }

    fn backward(
        &mut self,
        output_gradient: ArrayD<F>,
        _learning_rate: F,
        _batch_size: usize,
    ) -> ArrayD<F> {
        output_gradient * self.derivative(&self.input)
    }
}

impl<F: Float> From<ReLU<F>> for Box<dyn Layer<F>> {
    fn from(item: ReLU<F>) -> Self {
        Box::new(item)
    }
}
