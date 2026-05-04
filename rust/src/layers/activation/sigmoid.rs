use ndarray::{ArrayD, IxDyn};

use super::Float;
use super::Layer;

pub struct Sigmoid<F: Float> {
    output: ArrayD<F>,
}

impl<F: Float> Default for Sigmoid<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Sigmoid<F> {
    pub fn new() -> Sigmoid<F> {
        Sigmoid {
            output: ArrayD::zeros(IxDyn(&[0])),
        }
    }
    pub fn activate(&self, input: &ArrayD<F>) -> ArrayD<F> {
        input.mapv(|x| F::from_f32(1.0).unwrap() / (F::from_f32(1.0).unwrap() + F::exp(-x)))
    }
}

impl<F: Float> Layer<F> for Sigmoid<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.output = self.activate(&input);
        self.output.clone()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let one = F::from_f32(1.0).unwrap();
        let derivative = self.output.mapv(|y| y * (one - y));
        output_gradient * derivative
    }
}
impl<F: Float> From<Sigmoid<F>> for Box<dyn Layer<F>> {
    fn from(item: Sigmoid<F>) -> Self {
        Box::new(item)
    }
}
