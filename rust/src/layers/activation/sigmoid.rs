use ndarray::{ArrayD, IxDyn, Zip};

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
    fn forward(&mut self, mut input: ArrayD<F>) -> ArrayD<F> {
        let one = F::from_f32(1.0).unwrap();
        input.mapv_inplace(|x| one / (one + F::exp(-x)));
        self.output = input.clone();
        input
    }

    fn backward(&mut self, mut output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let one = F::from_f32(1.0).unwrap();
        Zip::from(&mut output_gradient)
            .and(&self.output)
            .for_each(|g, &y| {
                *g = *g * (y * (one - y));
            });
        output_gradient
    }
}
impl<F: Float> From<Sigmoid<F>> for Box<dyn Layer<F>> {
    fn from(item: Sigmoid<F>) -> Self {
        Box::new(item)
    }
}
