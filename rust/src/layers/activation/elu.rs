use ndarray::{ArrayD, IxDyn};

use super::Float;
use super::Layer;

pub struct ELU<F: Float> {
    output: ArrayD<F>,
    alpha: F,
}

impl<F: Float> ELU<F> {
    pub fn new(alpha: F) -> ELU<F> {
        ELU {
            output: ArrayD::zeros(IxDyn(&[0])),
            alpha,
        }
    }
    pub fn activate(&self, array: &ArrayD<F>) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        let one = F::from_f32(1.0).unwrap();
        let alpha = self.alpha;
        array.mapv(|x| {
            if x >= zero {
                x
            } else {
                alpha * (x.exp() - one)
            }
        })
    }
}
impl<F: Float> Default for ELU<F> {
    fn default() -> Self {
        Self::new(F::from_f32(1.0).unwrap())
    }
}

impl<F: Float> Layer<F> for ELU<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        let one = F::from_f32(1.0).unwrap();
        let alpha = self.alpha;
        self.output = input.mapv(|x| {
            if x >= zero {
                x
            } else {
                alpha * (x.exp() - one)
            }
        });
        self.output.clone()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        let one = F::from_f32(1.0).unwrap();
        let alpha = self.alpha;
        let derivative = self
            .output
            .mapv(|y| if y >= zero { one } else { y + alpha });
        output_gradient * derivative
    }
}

impl<F: Float> From<ELU<F>> for Box<dyn Layer<F>> {
    fn from(item: ELU<F>) -> Self {
        Box::new(item)
    }
}
