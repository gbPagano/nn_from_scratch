use ndarray::{array, ArrayD};

use super::Activate;
use super::Float;
use super::Layer;

pub struct Sigmoid<F: Float> {
    input: ArrayD<F>,
}

impl<F: Float> Default for Sigmoid<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Sigmoid<F> {
    pub fn new() -> Sigmoid<F> {
        Sigmoid {
            input: array![[]].into_dyn(),
        }
    }
}

impl<F: Float> Activate<F> for Sigmoid<F> {
    fn activate(&self, input: &ArrayD<F>) -> ArrayD<F> {
        input.mapv(|x| F::from_f32(1.0).unwrap() / (F::from_f32(1.0).unwrap() + F::exp(-x)))
    }

    fn derivative(&self, input: &ArrayD<F>) -> ArrayD<F> {
        let activation = self.activate(input);
        &activation * (&-activation.clone() + F::from_f32(1.0).unwrap())
    }
}

impl<F: Float> Layer<F> for Sigmoid<F> {
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
impl<F: Float> From<Sigmoid<F>> for Box<dyn Layer<F>> {
    fn from(item: Sigmoid<F>) -> Self {
        Box::new(item)
    }
}
