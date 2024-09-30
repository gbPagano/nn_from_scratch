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

        let conv_out = conv_layer.forward(input);
        let pred = SoftmaxCE::default().forward(conv_out);
        let real = arr3(&[
            [[1., 0., 0., 1.], [0., 1., 1., 0.]],
            [[0., 1., 1., 0.], [1., 0., 0., 1.]],
        ]);

        let error = real.into_dyn() - pred;
        let input_grad = conv_layer.backward(error, 0.5, 1);
        assert_abs_diff_eq!(
            conv_layer.get_weights().unwrap().index_axis(Axis(0), 0),
            arr3(&[
                [[-2.5, -2.0, 0.0], [-1.0, -2.0, -2.5], [-2.0, -4.0, -1.5]],
                [[-6.5, -4.0, -1.5], [-2.0, -5.5, -5.0], [-5.5, -8.0, -4.5]]
            ])
            .into_dyn(),
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(
            conv_layer.get_weights().unwrap().index_axis(Axis(0), 1),
            arr3(&[
                [
                    [-0.4867033, -0.5132971, -2.0199459],
                    [-0.9966755, -1.4867028, 0.49667543],
                    [-1.9933513, -0.49335182, -2.0166214]
                ],
                [
                    [-0.9734066, -1.0265942, -4.0398917],
                    [-1.993351, -2.9734056, 0.99335086],
                    [-3.9867027, -0.98670363, -4.0332427]
                ]
            ])
            .into_dyn(),
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            input_grad,
            arr3(&[
                [
                    [1.0, -6.83e-13, 2.0, 1.013, -0.006, 1.0],
                    [-1.4186888e-14, 4.0, 1.0132971, 1.9867033, 3.0132968, -0.006],
                    [2.0, -2.03e-9, 3.993, 3.01, -0.013, 1.99],
                    [-1.24e-14, 2.0, 1.0, 0.99, 1.99, -1.85e-12]
                ],
                [
                    [0.5, -3.44e-13, 2.5, 0.52, -0.013, 0.5],
                    [-9.63e-15, 5.0, 0.526, 2.473, 3.02, -0.013296704],
                    [2.5, -4.06e-9, 4.986, 3.026, -0.026, 2.4999995],
                    [-6.24e-15, 2.5, 0.5, 0.4867033, 2.4999995, -9.272684e-13]
                ]
            ])
            .into_dyn(),
            epsilon = 1e-2
        );
    }
}
