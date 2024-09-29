extern crate blas_src;

use ndarray::ArrayD;

pub mod activation;
mod dense;
pub use dense::*;
mod conv;
pub use conv::*;

pub trait Layer<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F>;
    fn backward(
        &mut self,
        output_gradient: ArrayD<F>,
        learning_rate: F,
        batch_size: usize,
    ) -> ArrayD<F>;
    fn get_weights(&self) -> Option<ArrayD<F>> {
        None
    }
    fn get_bias(&self) -> Option<ArrayD<F>> {
        None
    }
}

#[macro_export]
macro_rules! box_layers {
    [ $( $layer:expr ),* ] => {
        vec![
            $( Box::new($layer) as Box<dyn Layer<F>> ),*
        ]
    };
}

#[cfg(test)]
mod tests {
    use super::activation::*;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rstest::*;

    #[fixture]
    fn simple_layers_a() -> (
        Dense<f64>,
        Dense<f64>,
        Sigmoid<f64>,
        Sigmoid<f64>,
        ArrayD<f64>,
        ArrayD<f64>,
    ) {
        let mut layer_a = Dense::new(2, 2);
        layer_a.weights = array![[0.15, 0.2], [0.25, 0.3]];
        layer_a.bias = array![[0.35], [0.35]];

        let mut layer_b = Dense::new(2, 2);
        layer_b.weights = array![[0.4, 0.45], [0.5, 0.55]];
        layer_b.bias = array![[0.6], [0.6]];

        let inputs = array![[0.05], [0.1]].into_dyn();
        let desired = array![[0.01], [0.99]].into_dyn();

        let activate_a = Sigmoid::new();
        let activate_b = Sigmoid::new();

        (layer_a, layer_b, activate_a, activate_b, inputs, desired)
    }

    #[fixture]
    fn simple_layers_b() -> (
        Dense<f64>,
        Dense<f64>,
        Dense<f64>,
        TanH<f64>,
        TanH<f64>,
        TanH<f64>,
        ArrayD<f64>,
    ) {
        let mut layer_a = Dense::new(2, 3);
        layer_a.weights = array![[0.4, 0.5], [0.6, 0.7], [0.8, 0.3]];
        layer_a.bias = array![[-0.2], [-0.3], [-0.4]];

        let mut layer_b = Dense::new(3, 2);
        layer_b.weights = array![[0.6, 0.2, 0.7], [0.7, 0.2, 0.8]];
        layer_b.bias = array![[0.7], [0.3]];

        let mut layer_c = Dense::new(2, 1);
        layer_c.weights = array![[0.8, 0.5]];
        layer_c.bias = array![[-0.1]];

        let inputs = array![[0.3], [0.7]].into_dyn();

        let activate_a = TanH::new();
        let activate_b = TanH::new();
        let activate_c = TanH::new();

        (
            layer_a, layer_b, layer_c, activate_a, activate_b, activate_c, inputs,
        )
    }

    #[rstest]
    fn test_forward_a(
        simple_layers_a: (
            Dense<f64>,
            Dense<f64>,
            Sigmoid<f64>,
            Sigmoid<f64>,
            ArrayD<f64>,
            ArrayD<f64>,
        ),
    ) {
        let (mut layer_a, mut layer_b, mut activate, _, inputs, _) = simple_layers_a;

        let out_1 = layer_a.forward(inputs);
        let out_2 = activate.forward(out_1.clone());
        let out_3 = layer_b.forward(out_2.clone());
        let out_4 = activate.forward(out_3.clone());

        assert_abs_diff_eq!(out_1, array![[0.3775], [0.3925]].into_dyn(), epsilon = 1e-8);
        assert_abs_diff_eq!(
            out_2,
            array![[0.59326999], [0.59688438]].into_dyn(),
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            out_3,
            array![[1.10590597], [1.2249214]].into_dyn(),
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            out_4,
            array![[0.75136507], [0.77292847]].into_dyn(),
            epsilon = 1e-8
        );
    }

    #[rstest]
    fn test_forward_b(
        simple_layers_b: (
            Dense<f64>,
            Dense<f64>,
            Dense<f64>,
            TanH<f64>,
            TanH<f64>,
            TanH<f64>,
            ArrayD<f64>,
        ),
    ) {
        let (mut layer_a, mut layer_b, mut layer_c, mut activate, _, _, inputs) = simple_layers_b;

        let out_1 = layer_a.forward(inputs);
        let out_2 = activate.forward(out_1.clone());
        let out_3 = layer_b.forward(out_2.clone());
        let out_4 = activate.forward(out_3.clone());
        let out_5 = layer_c.forward(out_4.clone());
        let out_6 = activate.forward(out_5.clone());

        assert_abs_diff_eq!(
            out_1,
            array![[0.27], [0.37], [0.05]].into_dyn(),
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(
            out_2,
            array![[0.26], [0.35], [0.05]].into_dyn(),
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(out_3, array![[0.96], [0.59]].into_dyn(), epsilon = 1e-2);
        assert_abs_diff_eq!(out_4, array![[0.74], [0.53]].into_dyn(), epsilon = 1e-2);
        assert_abs_diff_eq!(out_5, array![[0.76]].into_dyn(), epsilon = 1e-2);
        assert_abs_diff_eq!(out_6, array![[0.64]].into_dyn(), epsilon = 1e-2);
    }

    #[rstest]
    fn test_last_layer_backward(
        simple_layers_a: (
            Dense<f64>,
            Dense<f64>,
            Sigmoid<f64>,
            Sigmoid<f64>,
            ArrayD<f64>,
            ArrayD<f64>,
        ),
    ) {
        let (mut layer_a, mut layer_b, mut activate_a, mut activate_b, inputs, desired) =
            simple_layers_a;

        let out_1 = layer_a.forward(inputs);
        let out_2 = activate_a.forward(out_1.clone());
        let out_3 = layer_b.forward(out_2.clone());
        let out_4 = activate_b.forward(out_3.clone());

        // TODO: document cost function used here
        let output_err = -(desired - out_4);
        let grad = activate_b.backward(output_err, 0.5, 1);
        layer_b.backward(grad, 0.5, 1);

        let expected_weights = array![[0.35891648, 0.408666186], [0.511301270, 0.561370121]];
        assert_abs_diff_eq!(layer_b.weights, expected_weights, epsilon = 1e-9);
    }

    #[rstest]
    fn test_middle_layer_backward(
        simple_layers_a: (
            Dense<f64>,
            Dense<f64>,
            Sigmoid<f64>,
            Sigmoid<f64>,
            ArrayD<f64>,
            ArrayD<f64>,
        ),
    ) {
        let (mut layer_a, mut layer_b, mut activate_a, mut activate_b, inputs, desired) =
            simple_layers_a;

        let out_1 = layer_a.forward(inputs);
        let out_2 = activate_a.forward(out_1.clone());
        let out_3 = layer_b.forward(out_2.clone());
        let out_4 = activate_b.forward(out_3.clone());

        // TODO: document cost function used here
        let output_err = -(desired - out_4);
        let grad = activate_b.backward(output_err, 0.5, 1);
        let grad = layer_b.backward(grad, 0.5, 1);
        let grad = activate_a.backward(grad, 0.5, 1);
        layer_a.backward(grad, 0.5, 1);

        let expected_weights = array![[0.14978072, 0.19956143], [0.24975114, 0.29950229]];
        assert_abs_diff_eq!(layer_a.weights, expected_weights, epsilon = 1e-8);
    }
}
