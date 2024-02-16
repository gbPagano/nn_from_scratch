extern crate blas_src;

use ndarray::{Array1, Array2, Zip};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_traits::FromPrimitive;

use crate::functions::*;
use crate::utils::FloatNN;

pub trait Layer<F>: Sync + Send {
    fn forward(&mut self, input: &Array1<F>) -> Array1<F>;
    fn gradient_descent(
        &mut self,
        error: Option<&Array1<F>>,
        next_layer: Option<&dyn Layer<F>>,
    ) -> Array2<F>;
    fn update_weights(&mut self, alpha: F, gradient_descent: &Array2<F>);

    fn get_delta(&self) -> Option<&Array1<F>>;
    fn get_net(&self) -> Option<&Array1<F>>;
    fn get_output(&self) -> Option<&Array1<F>>;
    fn get_weights(&self) -> &Array2<F>;
}

pub struct Dense<F: FloatNN> {
    pub function: Box<dyn ActivateFunction<F>>,
    pub weights: Array2<F>,
    pub bias: Array1<F>,

    // to be defined later
    pub delta: Option<Array1<F>>,
    pub input: Option<Array1<F>>,
    pub net: Option<Array1<F>>,
    pub output: Option<Array1<F>>,
}
impl<F: FloatNN> Dense<F> {
    pub fn new(
        inputs: usize,
        outputs: usize,
        function: impl ActivateFunction<F> + 'static,
    ) -> Self {
        let weights: Array2<F> = Array2::random(
            (outputs, inputs),
            Uniform::new::<F, F>(
                FromPrimitive::from_f64(-0.5).unwrap(),
                FromPrimitive::from_f64(0.5).unwrap(),
            ),
        );
        let bias: Array1<F> = Array1::random(
            outputs,
            Uniform::new::<F, F>(
                FromPrimitive::from_f64(-0.5).unwrap(),
                FromPrimitive::from_f64(0.5).unwrap(),
            ),
        );
        Dense {
            function: Box::new(function),
            weights,
            bias,
            delta: None,
            net: None,
            input: None,
            output: None,
        }
    }
}
impl<F: FloatNN> Layer<F> for Dense<F> {
    fn forward(&mut self, input: &Array1<F>) -> Array1<F> {
        let net = input.dot(&self.weights.t()) + &self.bias;
        self.output = Some(self.function.activate(&net));
        self.input = Some(input.to_owned());
        self.net = Some(net);

        self.output.as_ref().unwrap().clone()
    }

    fn get_delta(&self) -> Option<&Array1<F>> {
        self.delta.as_ref()
    }
    fn get_net(&self) -> Option<&Array1<F>> {
        self.net.as_ref()
    }
    fn get_output(&self) -> Option<&Array1<F>> {
        self.output.as_ref()
    }
    fn get_weights(&self) -> &Array2<F> {
        &self.weights
    }

    fn gradient_descent(
        &mut self,
        error: Option<&Array1<F>>,
        next_layer: Option<&dyn Layer<F>>,
    ) -> Array2<F> {
        self.delta = if next_layer.is_some() {
            let next_layer_delta = next_layer.unwrap().get_delta().unwrap();
            let next_layer_old_weights = next_layer.unwrap().get_weights();
            Some(
                next_layer_delta.dot(next_layer_old_weights)
                    * self.function.derivative(self.net.as_ref().unwrap()),
            )
        } else {
            // is last layer
            Some(error.unwrap() * self.function.derivative(self.net.as_ref().unwrap()))
        };

        let input = self.input.as_ref().unwrap();
        let delta = self.delta.as_ref().unwrap();
        Array2::from_shape_fn((delta.len(), input.len()), |(i, j)| unsafe {
            *input.uget(j) * *delta.uget(i)
        })
    }

    fn update_weights(&mut self, alpha: F, gradient_descent: &Array2<F>) {
        Zip::from(&mut self.weights)
            .and(&(gradient_descent * alpha))
            .for_each(|a, &b| *a += b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rstest::*;

    #[fixture]
    fn simple_layers_a() -> (Dense<f64>, Dense<f64>, Array1<f64>, Array1<f64>) {
        let mut layer_a = Dense::new(2, 2, Sigmoid);
        layer_a.weights = array![[0.15, 0.2], [0.25, 0.3]];
        layer_a.bias = array![0.35, 0.35];

        let mut layer_b = Dense::new(2, 2, Sigmoid);
        layer_b.weights = array![[0.4, 0.45], [0.5, 0.55]];
        layer_b.bias = array![0.6, 0.6];

        let inputs = array![0.05, 0.1];
        let desired = array![0.01, 0.99];

        (layer_a, layer_b, inputs, desired)
    }

    #[fixture]
    fn simple_layers_b() -> (Dense<f64>, Dense<f64>, Dense<f64>, Array1<f64>) {
        let mut layer_a = Dense::new(2, 3, TanH);
        layer_a.weights = array![[0.4, 0.5], [0.6, 0.7], [0.8, 0.3]];
        layer_a.bias = array![-0.2, -0.3, -0.4];

        let mut layer_b = Dense::new(3, 2, TanH);
        layer_b.weights = array![[0.6, 0.2, 0.7], [0.7, 0.2, 0.8]];
        layer_b.bias = array![0.7, 0.3];

        let mut layer_c = Dense::new(2, 1, TanH);
        layer_c.weights = array![[0.8, 0.5]];
        layer_c.bias = array![-0.1];

        let inputs = array![0.3, 0.7];

        (layer_a, layer_b, layer_c, inputs)
    }

    #[rstest]
    fn test_forward_a(simple_layers_a: (Dense<f64>, Dense<f64>, Array1<f64>, Array1<f64>)) {
        let (mut layer_a, mut layer_b, inputs, _) = simple_layers_a;

        layer_a.forward(&inputs);
        layer_b.forward(&layer_a.output.as_ref().unwrap());

        assert_abs_diff_eq!(layer_a.net.unwrap(), array![0.3775, 0.3925], epsilon = 1e-8);
        assert_abs_diff_eq!(
            layer_a.output.unwrap(),
            array![0.59326999, 0.59688438],
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            layer_b.net.unwrap(),
            array![1.10590597, 1.2249214],
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            layer_b.output.unwrap(),
            array![0.75136507, 0.77292847],
            epsilon = 1e-8
        );
    }

    #[rstest]
    fn test_last_layer_backward(
        simple_layers_a: (Dense<f64>, Dense<f64>, Array1<f64>, Array1<f64>),
    ) {
        let (mut layer_a, mut layer_b, inputs, desired) = simple_layers_a;

        layer_a.forward(&inputs);
        layer_b.forward(&layer_a.output.as_ref().unwrap());

        let output_err = &desired - layer_b.output.as_ref().unwrap();
        let gradient = layer_b.gradient_descent(Some(&output_err), None);
        layer_b.update_weights(0.5, &gradient);

        let expected_weights = array![[0.35891648, 0.408666186], [0.511301270, 0.561370121]];
        assert_abs_diff_eq!(layer_b.weights, expected_weights, epsilon = 1e-9);
    }

    #[rstest]
    fn test_middle_layer_backward(
        simple_layers_a: (Dense<f64>, Dense<f64>, Array1<f64>, Array1<f64>),
    ) {
        let (mut layer_a, mut layer_b, inputs, desired) = simple_layers_a;

        layer_a.forward(&inputs);
        layer_b.forward(&layer_a.output.as_ref().unwrap());

        let output_err = &desired - layer_b.output.as_ref().unwrap();
        let grad_b = layer_b.gradient_descent(Some(&output_err), None);
        let grad_a = layer_a.gradient_descent(None, Some(&layer_b));
        layer_b.update_weights(0.5, &grad_b);
        layer_a.update_weights(0.5, &grad_a);

        let expected_weights = array![[0.14978072, 0.19956143], [0.24975114, 0.29950229]];
        assert_abs_diff_eq!(layer_a.weights, expected_weights, epsilon = 1e-8);
    }

    #[rstest]
    fn test_forward_b(simple_layers_b: (Dense<f64>, Dense<f64>, Dense<f64>, Array1<f64>)) {
        let (mut layer_a, mut layer_b, mut layer_c, inputs) = simple_layers_b;

        layer_a.forward(&inputs);
        layer_b.forward(&layer_a.output.as_ref().unwrap());
        layer_c.forward(&layer_b.output.as_ref().unwrap());

        assert_abs_diff_eq!(
            layer_a.net.unwrap(),
            array![0.27, 0.37, 0.05],
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(
            layer_a.output.unwrap(),
            array![0.26, 0.35, 0.05],
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(layer_b.net.unwrap(), array![0.96, 0.59], epsilon = 1e-2);
        assert_abs_diff_eq!(layer_b.output.unwrap(), array![0.74, 0.53], epsilon = 1e-2);
        assert_abs_diff_eq!(layer_c.net.unwrap(), array![0.76], epsilon = 1e-2);
        assert_abs_diff_eq!(layer_c.output.unwrap(), array![0.64], epsilon = 1e-2);
    }
}
