extern crate blas_src;

use ndarray::Array1;
use num_traits::FromPrimitive;
use std::iter::Sum;

use crate::utils::FloatNN;

pub trait ActivateFunction<F>: Sync + Send {
    fn activate(&self, x: &Array1<F>) -> Array1<F>;
    fn derivative(&self, x: &Array1<F>) -> Array1<F>;
}

pub struct ELU<F: FloatNN> {
    alpha: F,
}

impl<F: FloatNN> ELU<F> {
    pub fn new(alpha: F) -> Self {
        Self { alpha }
    }
}

impl<F: FloatNN> ActivateFunction<F> for ELU<F> {
    fn activate(&self, array: &Array1<F>) -> Array1<F> {
        array.mapv(|x| {
            if x >= FromPrimitive::from_f64(0.0).unwrap() {
                x
            } else {
                self.alpha * (x.exp() - FromPrimitive::from_f64(1.0).unwrap())
            }
        })
    }

    fn derivative(&self, array: &Array1<F>) -> Array1<F> {
        array.mapv(|x| {
            if x >= FromPrimitive::from_f64(0.0).unwrap() {
                FromPrimitive::from_f64(1.0).unwrap()
            } else {
                self.alpha * (x.exp() - FromPrimitive::from_f64(1.0).unwrap()) + self.alpha
            }
        })
    }
}
pub struct Sigmoid;
impl ActivateFunction<f64> for Sigmoid {
    fn activate(&self, array: &Array1<f64>) -> Array1<f64> {
        array.mapv(|x| 1.0 / (1.0 + f64::exp(-x)))
    }

    fn derivative(&self, array: &Array1<f64>) -> Array1<f64> {
        let activation = self.activate(array);
        &activation * (1.0 - &activation)
    }
}
impl ActivateFunction<f32> for Sigmoid {
    fn activate(&self, array: &Array1<f32>) -> Array1<f32> {
        array.mapv(|x| 1.0 / (1.0 + f32::exp(-x)))
    }

    fn derivative(&self, array: &Array1<f32>) -> Array1<f32> {
        let activation = self.activate(array);
        &activation * (1.0 - &activation)
    }
}

pub struct TanH;
impl ActivateFunction<f64> for TanH {
    fn activate(&self, array: &Array1<f64>) -> Array1<f64> {
        array.mapv(|x| x.tanh())
    }

    fn derivative(&self, array: &Array1<f64>) -> Array1<f64> {
        let activation = self.activate(array);
        1.0 - activation.mapv(|x| x.powf(2.0))
    }
}
impl ActivateFunction<f32> for TanH {
    fn activate(&self, array: &Array1<f32>) -> Array1<f32> {
        array.mapv(|x| x.tanh())
    }

    fn derivative(&self, array: &Array1<f32>) -> Array1<f32> {
        let activation = self.activate(array);
        1.0 - activation.mapv(|x| x.powf(2.0))
    }
}

pub struct ErrorFunction;
impl ErrorFunction {
    pub fn get_mse<F: FloatNN + for<'a> Sum<&'a F>>(
        desired: &[&Array1<F>],
        outputs: &[Array1<F>],
    ) -> F {
        let mut errors = Vec::new();

        for (y, out) in desired.iter().zip(outputs.iter()) {
            let err: F =
                <F as FromPrimitive>::from_f64(0.5).unwrap() * (*y - out).mapv(|v| v.powi(2)).sum();
            errors.push(err);
        }
        errors.iter().sum::<F>() / FromPrimitive::from_usize(errors.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::Sigmoid;
    use crate::layer::Dense;
    use crate::neural_network::NeuralNetwork;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};
    use rstest::*;

    #[fixture]
    fn simple_nn() -> (NeuralNetwork<f64>, Array2<f64>, Array2<f64>) {
        let mut layer_a = Dense::new(2, 2, Sigmoid);
        layer_a.weights = array![[0.15, 0.2], [0.25, 0.3]];
        layer_a.bias = array![0.35, 0.35];

        let mut layer_b = Dense::new(2, 2, Sigmoid);
        layer_b.weights = array![[0.4, 0.45], [0.5, 0.55]];
        layer_b.bias = array![0.6, 0.6];

        let inputs = array![[0.05, 0.1]];
        let desired = array![[0.01, 0.99]];
        let nn = NeuralNetwork::new(vec![layer_a, layer_b]);

        (nn, inputs, desired)
    }

    #[rstest]
    fn test_error_function(simple_nn: (NeuralNetwork<f64>, Array2<f64>, Array2<f64>)) {
        let (mut nn, inputs, desired) = simple_nn;

        let out = nn.forward(&inputs.row(0).to_owned());

        let error = ErrorFunction::get_mse::<f64>(&vec![&desired.row(0).to_owned()], &vec![out]);

        assert_abs_diff_eq!(error, 0.298371109, epsilon = 1e9);
    }
}
