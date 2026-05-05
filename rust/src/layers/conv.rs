extern crate blas_src;

use ndarray::*;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use super::super::optimizer::{Optimizer, OptimizerConfig, SGD};
use super::super::Float;
use super::Layer;

pub struct Conv<F: Float> {
    pub weights: Array4<F>,
    pub bias: Array3<F>,
    input: Array4<F>,
    input_cols: Array2<F>,
    weights_optimizer: Box<dyn Optimizer<F>>,
    bias_optimizer: Box<dyn Optimizer<F>>,
}

impl<F: Float> Conv<F> {
    pub fn new(input_shape: (usize, usize, usize), kernels: usize, kernel_size: usize) -> Self {
        let (input_depth, _, _) = input_shape;

        // He/Kaiming initialization: std = sqrt(2 / fan_in), suited to ReLU/ELU.
        let fan_in = (input_depth * kernel_size * kernel_size) as f32;
        let std = (2.0 / fan_in).sqrt();
        let weights = Array4::random(
            (kernels, input_depth, kernel_size, kernel_size),
            Normal::new(0.0, std).unwrap(),
        )
        .mapv(|v: f32| F::from_f32(v).unwrap());
        Self::with_weights(input_shape, weights)
    }

    pub fn new_with_rng<R: Rng>(
        input_shape: (usize, usize, usize),
        kernels: usize,
        kernel_size: usize,
        rng: &mut R,
    ) -> Self {
        let (input_depth, _, _) = input_shape;

        // He/Kaiming initialization: std = sqrt(2 / fan_in), suited to ReLU/ELU.
        let fan_in = (input_depth * kernel_size * kernel_size) as f32;
        let std = (2.0 / fan_in).sqrt();
        let weights = Array4::random_using(
            (kernels, input_depth, kernel_size, kernel_size),
            Normal::new(0.0, std).unwrap(),
            rng,
        )
        .mapv(|v: f32| F::from_f32(v).unwrap());
        Self::with_weights(input_shape, weights)
    }

    fn with_weights(input_shape: (usize, usize, usize), weights: Array4<F>) -> Self {
        let (_input_depth, input_height, input_width) = input_shape;
        let kernels = weights.shape()[0];
        let kernel_size = weights.shape()[2];
        let bias = Array3::zeros((
            kernels,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        ));

        Self {
            weights,
            bias,
            input: Array4::zeros((0, 0, 0, 0)),
            input_cols: Array2::zeros((0, 0)),
            weights_optimizer: Box::new(SGD),
            bias_optimizer: Box::new(SGD),
        }
    }
}

impl<F: Float> Layer<F> for Conv<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.input = input.into_dimensionality::<Ix4>().unwrap();
        let b = self.input.shape()[0];
        let (k, _c, kh, kw) = self.weights.dim();
        let h_out = self.input.shape()[2] - kh + 1;
        let w_out = self.input.shape()[3] - kw + 1;

        self.input_cols = im2col(&self.input, kh, kw);
        let weights_col = self
            .weights
            .view()
            .into_shape((k, self.input_cols.shape()[1]))
            .unwrap();
        let output_col = self.input_cols.dot(&weights_col.t());
        let mut output = output_col
            .into_shape((b, h_out, w_out, k))
            .unwrap()
            .permuted_axes([0, 3, 1, 2]);
        output += &self.bias;
        let output = output.as_standard_layout().to_owned();

        output.into_dyn()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, learning_rate: F) -> ArrayD<F> {
        let output_gradient = output_gradient.into_dimensionality::<Ix4>().unwrap();
        let b = output_gradient.shape()[0];
        let b_f = F::from_usize(b).unwrap();
        let (k, c, kh, kw) = self.weights.dim();

        let h_out = output_gradient.shape()[2];
        let w_out = output_gradient.shape()[3];
        let output_gradient_col = output_gradient
            .view()
            .permuted_axes([0, 2, 3, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape((b * h_out * w_out, k))
            .unwrap();
        let weights_col = self.weights.view().into_shape((k, c * kh * kw)).unwrap();

        let weights_gradient = output_gradient_col
            .t()
            .dot(&self.input_cols)
            .into_shape((k, c, kh, kw))
            .unwrap()
            / b_f;
        let input_gradient_col = output_gradient_col.dot(&weights_col);
        let input_gradient = col2im(&input_gradient_col, self.input.dim(), kh, kw, h_out, w_out);
        let bias_gradient = output_gradient.sum_axis(Axis(0)) / b_f;

        self.weights_optimizer.step(
            self.weights.view_mut().into_dyn(),
            weights_gradient.view().into_dyn(),
            learning_rate,
        );
        self.bias_optimizer.step(
            self.bias.view_mut().into_dyn(),
            bias_gradient.view().into_dyn(),
            learning_rate,
        );

        input_gradient.into_dyn()
    }

    fn get_weights(&self) -> Option<ArrayD<F>> {
        Some(self.weights.clone().into_dyn())
    }

    fn get_bias(&self) -> Option<ArrayD<F>> {
        Some(self.bias.clone().into_dyn())
    }

    fn set_weights(&mut self, weights: ArrayD<F>) {
        self.weights = weights.into_dimensionality::<Ix4>().unwrap();
    }

    fn set_bias(&mut self, bias: ArrayD<F>) {
        self.bias = bias.into_dimensionality::<Ix3>().unwrap();
    }

    fn set_optimizer(&mut self, config: &OptimizerConfig<F>) {
        self.weights_optimizer = config.build();
        self.bias_optimizer = config.build();
    }
}

fn im2col<F: Float>(input: &Array4<F>, kh: usize, kw: usize) -> Array2<F> {
    let (b, c, h, w) = input.dim();
    let h_out = h - kh + 1;
    let w_out = w - kw + 1;
    let mut cols = Array2::zeros((b * h_out * w_out, c * kh * kw));

    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = (bi * h_out + oh) * w_out + ow;
                for ci in 0..c {
                    for r in 0..kh {
                        for s in 0..kw {
                            let col = (ci * kh + r) * kw + s;
                            cols[[row, col]] = input[[bi, ci, oh + r, ow + s]];
                        }
                    }
                }
            }
        }
    }

    cols
}

fn col2im<F: Float>(
    cols: &Array2<F>,
    input_dim: (usize, usize, usize, usize),
    kh: usize,
    kw: usize,
    h_out: usize,
    w_out: usize,
) -> Array4<F> {
    let (b, c, h, w) = input_dim;
    let mut input_gradient = Array4::zeros((b, c, h, w));

    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = (bi * h_out + oh) * w_out + ow;
                for ci in 0..c {
                    for r in 0..kh {
                        for s in 0..kw {
                            let col = (ci * kh + r) * kw + s;
                            input_gradient[[bi, ci, oh + r, ow + s]] += cols[[row, col]];
                        }
                    }
                }
            }
        }
    }

    input_gradient
}

impl<F: Float> From<Conv<F>> for Box<dyn Layer<F>> {
    fn from(item: Conv<F>) -> Self {
        Box::new(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::activation::SoftmaxCE;
    use approx::assert_abs_diff_eq;

    #[test]
    fn conv_forward_a() {
        let input = arr2(&[
            [2., 0., 0., 4., 4., 0.],
            [1., 1., 0., 0., 2., 0.],
            [1., 0., 1., 2., 3., 0.],
            [1., 1., 2., 3., 1., 0.],
        ])
        .insert_axis(Axis(0))
        .insert_axis(Axis(0))
        .into_dyn();
        let kernel = arr3(&[[[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]]]).insert_axis(Axis(0));

        let mut conv_layer: Conv<f32> = Conv::new((1, 4, 6), 1, 3);
        assert_eq!(conv_layer.weights.dim(), (1, 1, 3, 3));
        assert_eq!(conv_layer.bias.dim(), (1, 2, 4));
        conv_layer.weights = kernel;
        conv_layer.bias = arr3(&[[[0., 0., 0., 1.], [0., 0., 0., 1.]]]);

        let out = conv_layer.forward(input);
        assert_eq!(
            out,
            arr3(&[[[5., 6., 8., 9.], [4., 6., 7., 7.]]])
                .insert_axis(Axis(0))
                .into_dyn()
        );
    }

    #[test]
    fn conv_forward_b() {
        let input = arr3(&[
            [
                [2., 0., 0., 4., 4., 0.],
                [1., 1., 0., 0., 2., 0.],
                [1., 0., 1., 2., 3., 0.],
                [1., 1., 2., 3., 1., 0.],
            ],
            [
                [4., 0., 0., 8., 8., 0.],
                [2., 2., 0., 0., 4., 0.],
                [2., 0., 2., 4., 6., 0.],
                [2., 2., 4., 6., 2., 0.],
            ],
        ])
        .insert_axis(Axis(0))
        .into_dyn();
        let kernel = arr3(&[
            [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]],
            [[0.5, 0., 0.5], [0., 0.5, 0.], [0.5, 0., 0.5]],
        ])
        .insert_axis(Axis(0));

        let mut conv_layer: Conv<f32> = Conv::new((2, 4, 6), 1, 3);
        assert_eq!(conv_layer.weights.dim(), (1, 2, 3, 3));
        assert_eq!(conv_layer.bias.dim(), (1, 2, 4));
        conv_layer.weights = kernel;
        conv_layer.bias = arr3(&[[[0., 0., 0., 1.], [0., 0., 0., 1.]]]);

        let out = conv_layer.forward(input);
        assert_eq!(
            out,
            arr3(&[[[10., 12., 16., 17.], [8., 12., 14., 13.]]])
                .insert_axis(Axis(0))
                .into_dyn()
        );
    }

    #[test]
    fn conv_forward_c() {
        let input = arr3(&[
            [
                [2., 0., 0., 4., 4., 0.],
                [1., 1., 0., 0., 2., 0.],
                [1., 0., 1., 2., 3., 0.],
                [1., 1., 2., 3., 1., 0.],
            ],
            [
                [4., 0., 0., 8., 8., 0.],
                [2., 2., 0., 0., 4., 0.],
                [2., 0., 2., 4., 6., 0.],
                [2., 2., 4., 6., 2., 0.],
            ],
        ])
        .insert_axis(Axis(0))
        .into_dyn();

        let kernel_a = arr3(&[
            [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]],
            [[0.5, 0., 0.5], [0., 0.5, 0.], [0.5, 0., 0.5]],
        ]);
        let kernel_b = arr3(&[
            [[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]],
            [[0., 2., 0.], [2., 0., 2.], [0., 2., 0.]],
        ]);
        let kernel = stack![Axis(0), kernel_a, kernel_b];

        let mut conv_layer: Conv<f32> = Conv::new((2, 4, 6), 2, 3);
        assert_eq!(conv_layer.weights.dim(), (2, 2, 3, 3));
        assert_eq!(conv_layer.bias.dim(), (2, 2, 4));
        conv_layer.weights = kernel;
        conv_layer.bias = arr3(&[
            [[0., 0., 0., 1.], [0., 0., 0., 1.]],
            [[1., 0., 0., 0.], [1., 0., 0., 0.]],
        ]);

        let out = conv_layer.forward(input);
        assert_abs_diff_eq!(
            out,
            arr3(&[
                [[10., 12., 16., 17.], [8., 12., 14., 13.]],
                [[6., 10., 40., 35.], [21., 20., 35., 25.]]
            ])
            .insert_axis(Axis(0))
            .into_dyn(),
            epsilon = 1e-2
        );
    }

    #[test]
    fn conv_backward() {
        let input = arr3(&[
            [
                [2., 0., 0., 4., 4., 0.],
                [1., 1., 0., 0., 2., 0.],
                [1., 0., 1., 2., 3., 0.],
                [1., 1., 2., 3., 1., 0.],
            ],
            [
                [4., 0., 0., 8., 8., 0.],
                [2., 2., 0., 0., 4., 0.],
                [2., 0., 2., 4., 6., 0.],
                [2., 2., 4., 6., 2., 0.],
            ],
        ])
        .insert_axis(Axis(0))
        .into_dyn();

        let kernel_a = arr3(&[
            [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]],
            [[0.5, 0., 0.5], [0., 0.5, 0.], [0.5, 0., 0.5]],
        ]);
        let kernel_b = arr3(&[
            [[1., 1., 0.], [1., 0., 1.], [0., 1., 0.]],
            [[1., 2., 0.], [2., 0., 2.], [0., 2., 0.]],
        ]);
        let kernel = stack![Axis(0), kernel_a, kernel_b];

        let mut conv_layer: Conv<f32> = Conv::new((2, 4, 6), 2, 3);
        assert_eq!(conv_layer.weights.dim(), (2, 2, 3, 3));
        assert_eq!(conv_layer.bias.dim(), (2, 2, 4));
        conv_layer.weights = kernel;
        conv_layer.bias = arr3(&[
            [[0., 0., 0., 1.], [0., 0., 0., 1.]],
            [[1., 0., 0., 0.], [1., 0., 0., 0.]],
        ]);

        let conv_out = conv_layer.forward(input);
        let conv_out_shape = conv_out.raw_dim();
        let flat_len = conv_out.len();
        // This test historically treats the whole conv map as one softmax vector.
        let pred = SoftmaxCE::default()
            .forward(conv_out.into_shape((1, flat_len)).unwrap().into_dyn())
            .into_shape(conv_out_shape)
            .unwrap();
        let real = arr3(&[
            [[1., 0., 0., 1.], [0., 1., 1., 0.]],
            [[0., 1., 1., 0.], [1., 0., 0., 1.]],
        ])
        .insert_axis(Axis(0));

        let error = real.into_dyn() - pred;
        conv_layer.backward(error.clone(), 0.005);
        let input_grad = conv_layer.backward(error, 0.005);
        assert_abs_diff_eq!(
            conv_layer.get_weights().unwrap().index_axis(Axis(0), 0),
            arr3(&[
                [
                    [0.929, -0.04, 0.98],
                    [-0.020, 0.940, -0.0499],
                    [0.9400, -0.08, 0.9500]
                ],
                [
                    [0.36, -0.08, 0.460],
                    [-0.04, 0.38, -0.099],
                    [0.38, -0.16, 0.399]
                ]
            ])
            .into_dyn(),
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(
            conv_layer.get_weights().unwrap().index_axis(Axis(0), 1),
            arr3(&[
                [
                    [1.029, 0.969, -0.079],
                    [0.960, -0.010, 0.970],
                    [-0.030, 0.979, -0.069]
                ],
                [
                    [1.059, 1.9399996, -0.15992686],
                    [1.920, -0.020, 1.940],
                    [-0.0600, 1.959, -0.1399]
                ]
            ])
            .into_dyn(),
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            input_grad,
            arr3(&[
                [
                    [0.96, 0.99, 2.98, 0.89, -1.04, 1.02],
                    [1.004, 3.899, 1.854, 1.975, 3.93, -1.049],
                    [1.95, -0.070, 3.895, 3.86, -1.094, 1.994],
                    [-0.0150, 1.959, 0.895, 0.9199, 1.964, -0.034]
                ],
                [
                    [0.429, 0.989, 3.479, 1.28, -2.08, 0.551],
                    [1.009, 4.799, 2.208, 1.451, 4.868, -2.09],
                    [2.4, -0.140, 4.790, 4.728, -2.188, 2.489],
                    [-0.030, 2.419, 0.290, 0.339, 2.429, -0.069]
                ]
            ])
            .insert_axis(Axis(0))
            .into_dyn(),
            epsilon = 1e-2
        );
    }
}
