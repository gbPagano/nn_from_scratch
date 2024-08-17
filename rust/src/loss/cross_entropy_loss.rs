extern crate blas_src;

use ndarray::ArrayD;

use super::super::Float;
use super::Loss;

// This loss function considers that we are using softmax as the last layer
pub struct CrossEntropySoftmax {}
impl Default for CrossEntropySoftmax {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Float> From<CrossEntropySoftmax> for Box<dyn Loss<F>> {
    fn from(item: CrossEntropySoftmax) -> Self {
        Box::new(item)
    }
}
impl CrossEntropySoftmax {
    pub fn new() -> Self {
        Self {}
    }
}
impl<F: Float> Loss<F> for CrossEntropySoftmax {
    fn loss(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> F {
        let epsilon = F::from_f32(1e-7).unwrap();
        -(y_real * y_pred.mapv(|v| if v > epsilon { v.ln() } else { epsilon.ln() })).sum()
    }
    fn gradient(&self, y_real: &ArrayD<F>, y_pred: &ArrayD<F>) -> ArrayD<F> {
        // This gradient considers that we are using softmax as the last layer
        y_pred - y_real
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_cross_entropy_loss() {
        let y_real = array![[0.0], [1.0], [0.0], [0.0], [0.0]].into_dyn();
        let y_pred = array![[0.0201], [0.9025,], [0.0496], [0.0110], [0.0165]].into_dyn();

        let loss = CrossEntropySoftmax::new().loss(&y_real, &y_pred);
        assert_relative_eq!(loss, 0.10258, max_relative = 0.0001);
    }
}
