use ndarray::ArrayD;

use super::super::Float;
use super::Layer;

pub trait Activate<F: Float>: Layer<F> {
    fn activate(&self, input: &ArrayD<F>) -> ArrayD<F>;
    fn derivative(&self, input: &ArrayD<F>) -> ArrayD<F>;
}

mod sigmoid;
pub use sigmoid::*;
mod tanh;
pub use tanh::*;
mod elu;
pub use elu::*;
mod relu;
pub use relu::*;
