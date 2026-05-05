use ndarray::{ArrayD, IxDyn, Zip};

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
    fn forward(&mut self, mut input: ArrayD<F>) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        let one = F::from_f32(1.0).unwrap();
        let alpha = self.alpha;
        input.mapv_inplace(|x| {
            if x >= zero {
                x
            } else {
                alpha * (x.exp() - one)
            }
        });
        self.output = input.clone();
        input
    }

    fn backward(&mut self, mut output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        let one = F::from_f32(1.0).unwrap();
        let alpha = self.alpha;
        Zip::from(&mut output_gradient)
            .and(&self.output)
            .for_each(|g, &y| {
                let d = if y >= zero { one } else { y + alpha };
                *g = *g * d;
            });
        output_gradient
    }
}

impl<F: Float> From<ELU<F>> for Box<dyn Layer<F>> {
    fn from(item: ELU<F>) -> Self {
        Box::new(item)
    }
}
