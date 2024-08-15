extern crate blas_src;

use ndarray::ArrayD;

mod mse;
pub use mse::*;

pub trait Loss<F> {
    fn loss(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> F;
    fn gradient(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> ArrayD<F>;
}
