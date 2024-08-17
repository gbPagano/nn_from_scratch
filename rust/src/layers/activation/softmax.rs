use ndarray::ArrayD;

use super::Float;
use super::Layer;

// Softmax Cross Entropy
// This layer considers that we are using softmax as the last layer,
// together with cross entropy loss as the loss function.
pub struct SoftmaxCE {}

impl SoftmaxCE {
    pub fn new() -> Self {
        Self {}
    }
    pub fn activate<F: Float>(&self, array: &ArrayD<F>) -> ArrayD<F> {
        let max = array
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let tmp = array.mapv(|x| (x - *max).exp());
        &tmp / tmp.sum()
    }
}
impl Default for SoftmaxCE {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Float> Layer<F> for SoftmaxCE {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.activate(&input)
    }

    fn backward(
        &mut self,
        output_gradient: ArrayD<F>,
        _learning_rate: F,
        _batch_size: usize,
    ) -> ArrayD<F> {
        // This function considers that we are using softmax as the last layer,
        // together with cross entropy loss as the loss function.
        output_gradient
    }
}

impl<F: Float> From<SoftmaxCE> for Box<dyn Layer<F>> {
    fn from(item: SoftmaxCE) -> Self {
        Box::new(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_activate_softmax() {
        let input = array![[1.3], [5.1], [2.2], [0.7], [1.1]].into_dyn();
        let out = SoftmaxCE::new().forward(input);
        assert_abs_diff_eq!(
            out,
            array![[0.0201], [0.9025,], [0.0496], [0.0110], [0.0165]].into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_activate_big_values_softmax() {
        let input = array![[1000.3], [5.1], [2.2], [0.7], [1.1]].into_dyn();
        let out = SoftmaxCE::new().forward(input);
        assert_abs_diff_eq!(
            out,
            array![[1.0], [0.0], [0.0], [0.0], [0.0]].into_dyn(),
            epsilon = 1e-4
        );
    }
}
