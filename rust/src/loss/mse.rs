extern crate blas_src;

use ndarray::ArrayD;

use super::super::Float;
use super::Loss;

pub struct MSE {}
impl Default for MSE {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Float> From<MSE> for Box<dyn Loss<F>> {
    fn from(item: MSE) -> Self {
        Box::new(item)
    }
}
impl MSE {
    pub fn new() -> Self {
        Self {}
    }
}
impl<F: Float> Loss<F> for MSE {
    fn loss(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> F {
        (y_real - y_pred).mapv(|v| v.powi(2)).mean().unwrap()
    }
    fn gradient(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> ArrayD<F> {
        (y_pred - y_real) * F::from_f32(2.0).unwrap()
    }
}

pub struct HalfMSE {}
impl Default for HalfMSE {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Float> From<HalfMSE> for Box<dyn Loss<F>> {
    fn from(item: HalfMSE) -> Self {
        Box::new(item)
    }
}
impl HalfMSE {
    pub fn new() -> Self {
        Self {}
    }
}
impl<F: Float> Loss<F> for HalfMSE {
    fn loss(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> F {
        (y_real - y_pred).mapv(|v| v.powi(2)).mean().unwrap() / F::from_f32(2.0).unwrap()
    }
    fn gradient(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> ArrayD<F> {
        y_pred - y_real
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_half_mse_loss() {
        let y_real = array![[0.01], [0.99]].into_dyn();
        let y_pred = array![[0.75136507], [0.77292847]].into_dyn();

        let loss = MSE::new().loss(&y_real, &y_pred);
        assert_relative_eq!(loss, 0.298371109, max_relative = 0.00001);
    }
}
