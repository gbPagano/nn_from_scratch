use ndarray::{array, ArrayD};

use super::Activate;
use super::Float;
use super::Layer;

pub struct ELU<F: Float> {
    input: ArrayD<F>,
    alpha: F,
}

impl<F: Float> ELU<F> {
    pub fn new(alpha: F) -> ELU<F> {
        ELU {
            input: array![[]].into_dyn(),
            alpha,
        }
    }
}
impl<F: Float> Default for ELU<F> {
    fn default() -> Self {
        Self::new(F::from_f32(1.0).unwrap())
    }
}
impl<F: Float> Activate<F> for ELU<F> {
    fn activate(&self, array: &ArrayD<F>) -> ArrayD<F> {
        array.mapv(|x| {
            if x >= F::from_f32(0.0).unwrap() {
                x
            } else {
                self.alpha * (x.exp() - F::from_f32(1.0).unwrap())
            }
        })
    }

    fn derivative(&self, array: &ArrayD<F>) -> ArrayD<F> {
        array.mapv(|x| {
            if x >= F::from_f32(0.0).unwrap() {
                F::from_f32(1.0).unwrap()
            } else {
                self.alpha * (x.exp() - F::from_f32(1.0).unwrap()) + self.alpha
            }
        })
    }
}

impl<F: Float> Layer<F> for ELU<F> {
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

impl<F: Float> From<ELU<F>> for Box<dyn Layer<F>> {
    fn from(item: ELU<F>) -> Self {
        Box::new(item)
    }
}
