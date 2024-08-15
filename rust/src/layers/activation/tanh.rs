use ndarray::{array, ArrayD};

use super::Activate;
use super::Float;
use super::Layer;

pub struct TanH<F: Float> {
    input: ArrayD<F>,
}

impl<F: Float> Default for TanH<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> TanH<F> {
    pub fn new() -> TanH<F> {
        TanH {
            input: array![[]].into_dyn(),
        }
    }
}

impl<F: Float> Activate<F> for TanH<F> {
    fn activate(&self, input: &ArrayD<F>) -> ArrayD<F> {
        input.mapv(|x| x.tanh())
    }

    fn derivative(&self, input: &ArrayD<F>) -> ArrayD<F> {
        let activation = self.activate(input);
        -activation.mapv(|x| F::powf(x, F::from_f32(2.0).unwrap())) + F::from_f32(1.0).unwrap()
    }
}

impl<F: Float> Layer<F> for TanH<F> {
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

impl<F: Float> From<TanH<F>> for Box<dyn Layer<F>> {
    fn from(item: TanH<F>) -> Self {
        Box::new(item)
    }
}
