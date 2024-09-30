extern crate blas_src;

use ndarray::*;
use ndarray_conv::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::super::Float;
use super::Layer;

pub struct Conv<F: Float> {
    pub weights: Array4<F>,
    pub bias: Array3<F>,
    input: Array3<F>,
    curr_batch: usize,
    input_shape: (usize, usize, usize),
    weights_gradient: Array4<F>,
    bias_gradient: Array3<F>,
}

impl<F: Float> Conv<F> {
    pub fn new(input_shape: (usize, usize, usize), kernels: usize, kernel_size: usize) -> Self {
        let (input_depth, input_height, input_width) = input_shape;

        let weights = Array4::random(
            (kernels, input_depth, kernel_size, kernel_size),
            Uniform::new(F::from_f32(-0.5).unwrap(), F::from_f32(0.5).unwrap()),
        );

        let bias = Array3::random(
            (
                kernels,
                input_height - kernel_size + 1,
                input_width - kernel_size + 1,
            ),
            Uniform::new(F::from_f32(-0.5).unwrap(), F::from_f32(0.5).unwrap()),
        );

        Self {
            weights,
            bias,
            input: arr3(&[[[]]]),
            curr_batch: 0,
            input_shape,
            bias_gradient: arr3(&[[[]]]),
            weights_gradient: Array4::zeros((kernels, input_depth, kernel_size, kernel_size)),
        }
    }
}

impl<F: Float> Layer<F> for Conv<F> {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        self.input = if input.ndim() == 3 {
            input.into_dimensionality::<Ix3>().unwrap()
        } else {
            input
                .into_dimensionality::<Ix2>()
                .unwrap()
                .insert_axis(Axis(0))
        };

        let mut output = Vec::new();
        for w in self.weights.outer_iter() {
            output.push(
                self.input
                    .conv_fft(&w, ConvMode::Valid, PaddingMode::Zeros)
                    .unwrap(),
            );
        }
        let mut output = concatenate(
            Axis(0),
            &output.iter().map(|i| i.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        output += &self.bias;
        output.into_dyn()
    }

    fn backward(
        &mut self,
        output_gradient: ArrayD<F>,
        learning_rate: F,
        batch_size: usize,
    ) -> ArrayD<F> {
        let output_gradient = output_gradient.into_dimensionality::<Ix3>().unwrap();
        let mut weights_gradient: Array4<F> = Array4::zeros(self.weights.dim());
        let mut input_gradient: Array3<F> = Array3::zeros(self.input_shape);

        for i in 0..self.weights.dim().0 {
            for j in 0..self.weights.dim().1 {
                weights_gradient.slice_mut(s![i, j, .., ..]).assign(
                    &self
                        .input
                        .index_axis(Axis(0), j)
                        .conv_fft(
                            &output_gradient.index_axis(Axis(0), i),
                            ConvMode::Valid,
                            PaddingMode::Zeros,
                        )
                        .unwrap(),
                );
                input_gradient.slice_mut(s![j, .., ..]).zip_mut_with(
                    &output_gradient
                        .index_axis(Axis(0), i)
                        .conv_fft(
                            &self.weights.slice(s![i, j, ..;-1, ..;-1]),
                            ConvMode::Full,
                            PaddingMode::Zeros,
                        )
                        .unwrap(),
                    |x, &y| *x += y,
                );
            }
        }

        if self.curr_batch == 0 {
            self.weights_gradient = weights_gradient;
            self.bias_gradient = output_gradient;
        } else {
            self.weights_gradient += &weights_gradient;
            self.bias_gradient += &output_gradient;
        }
        // gradient descent as optimizer
        self.curr_batch += 1;
        if self.curr_batch == batch_size {
            Zip::from(&mut self.weights)
                .and(&self.weights_gradient)
                .for_each(|a, &b| *a -= b * learning_rate / F::from_usize(batch_size).unwrap());

            Zip::from(&mut self.bias)
                .and(&self.bias_gradient)
                .for_each(|a, &b| *a -= b * learning_rate / F::from_usize(batch_size).unwrap());

            self.curr_batch = 0;
        }

        input_gradient.into_dyn()
    }

    fn get_weights(&self) -> Option<ArrayD<F>> {
        Some(self.weights.clone().into_dyn())
    }

    fn get_bias(&self) -> Option<ArrayD<F>> {
        Some(self.bias.clone().into_dyn())
    }
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
            arr3(&[[[5., 6., 8., 9.], [4., 6., 7., 7.]]]).into_dyn()
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
            arr3(&[[[10., 12., 16., 17.], [8., 12., 14., 13.]]]).into_dyn()
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
        let pred = SoftmaxCE::default().forward(conv_out);
        let real = arr3(&[
            [[1., 0., 0., 1.], [0., 1., 1., 0.]],
            [[0., 1., 1., 0.], [1., 0., 0., 1.]],
        ]);

        let error = real.into_dyn() - pred;
        conv_layer.backward(error.clone(), 0.005, 1);
        let input_grad = conv_layer.backward(error, 0.005, 1);
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
            .into_dyn(),
            epsilon = 1e-2
        );
    }
}
