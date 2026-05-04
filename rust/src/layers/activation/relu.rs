use ndarray::{ArrayD, IxDyn};

use super::Float;
use super::Layer;

pub struct ReLU<F: Float> {
    output: ArrayD<F>,
}

impl<F: Float> ReLU<F> {
    pub fn new() -> Self {
        Self {
            output: ArrayD::zeros(IxDyn(&[0])),
        }
    }
    pub fn activate(&self, array: &ArrayD<F>) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        array.mapv(|x| if x >= zero { x } else { zero })
    }
}
impl<F: Float> Default for ReLU<F> {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Float> Layer<F> for ReLU<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        self.output = input.mapv(|x| if x >= zero { x } else { zero });
        self.output.clone()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        let one = F::from_f32(1.0).unwrap();
        let derivative = self.output.mapv(|y| if y > zero { one } else { zero });
        output_gradient * derivative
    }
}

impl<F: Float> From<ReLU<F>> for Box<dyn Layer<F>> {
    fn from(item: ReLU<F>) -> Self {
        Box::new(item)
    }
}
