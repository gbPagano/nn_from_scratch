use ndarray::{ArrayD, IxDyn};

use super::Float;
use super::Layer;

pub struct TanH<F: Float> {
    output: ArrayD<F>,
}

impl<F: Float> Default for TanH<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> TanH<F> {
    pub fn new() -> TanH<F> {
        TanH {
            output: ArrayD::zeros(IxDyn(&[0])),
        }
    }
    pub fn activate(&self, input: &ArrayD<F>) -> ArrayD<F> {
        input.mapv(|x| x.tanh())
    }
}

impl<F: Float> Layer<F> for TanH<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.output = input.mapv(|x| x.tanh());
        self.output.clone()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let one = F::from_f32(1.0).unwrap();
        let derivative = self.output.mapv(|y| one - y * y);
        output_gradient * derivative
    }
}

impl<F: Float> From<TanH<F>> for Box<dyn Layer<F>> {
    fn from(item: TanH<F>) -> Self {
        Box::new(item)
    }
}
