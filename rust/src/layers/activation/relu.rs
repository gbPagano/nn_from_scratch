use ndarray::{ArrayD, IxDyn, Zip};

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
    fn forward(&mut self, mut input: ArrayD<F>) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        input.mapv_inplace(|x| if x >= zero { x } else { zero });
        self.output = input.clone();
        input
    }

    fn backward(&mut self, mut output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let zero = F::from_f32(0.0).unwrap();
        Zip::from(&mut output_gradient)
            .and(&self.output)
            .for_each(|g, &y| {
                if y <= zero {
                    *g = zero;
                }
            });
        output_gradient
    }
}

impl<F: Float> From<ReLU<F>> for Box<dyn Layer<F>> {
    fn from(item: ReLU<F>) -> Self {
        Box::new(item)
    }
}
