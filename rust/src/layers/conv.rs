extern crate blas_src;

use ndarray::{arr3, concatenate,Array3, ArrayD, Axis, Ix2, Ix3};
use ndarray_conv::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::super::Float;
use super::Layer;

pub struct Conv<F: Float> {
    pub weights: Vec<Array3<F>>,
    pub bias: Array3<F>,
    input: Array3<F>,
}

impl<F: Float> Conv<F> {
    pub fn new(input_shape: (usize, usize, usize), kernels: usize, kernel_size: usize) -> Self {
        let (input_depth, input_height, input_width) = input_shape;

        let weights: Vec<_> = (0..kernels)
            .map(|_| {
                Array3::random(
                    (input_depth, kernel_size, kernel_size),
                    Uniform::new(F::from_f32(-0.5).unwrap(), F::from_f32(0.5).unwrap()),
                )
            })
            .collect();
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
        for w in self.weights.iter() {
            output.push(
                self.input
                    .conv(w, ConvMode::Valid, PaddingMode::Zeros)
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
        output_gradient
    }

    fn get_weights(&self) -> Option<ArrayD<F>> {
        None
    }

    fn get_bias(&self) -> Option<ArrayD<F>> {
        None
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
    use ndarray::arr2;

    #[test]
    fn conv_forward_a() {
        let input = arr2(&[
            [2., 0., 0., 4., 4., 0.],
            [1., 1., 0., 0., 2., 0.],
            [1., 0., 1., 2., 3., 0.],
            [1., 1., 2., 3., 1., 0.],
        ])
        .into_dyn();
        let kernel = arr3(&[[[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]]]);

        let mut conv_layer: Conv<f32> = Conv::new((1, 4, 6), 1, 3);
        assert_eq!(conv_layer.weights.len(), 1);
        assert_eq!(conv_layer.weights[0].dim(), (1, 3, 3));
        assert_eq!(conv_layer.bias.dim(), (1, 2, 4));
        conv_layer.weights[0] = kernel;
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
        ]);

        let mut conv_layer: Conv<f32> = Conv::new((2, 4, 6), 1, 3);
        assert_eq!(conv_layer.weights.len(), 1);
        assert_eq!(conv_layer.weights[0].dim(), (2, 3, 3));
        assert_eq!(conv_layer.bias.dim(), (1, 2, 4));
        conv_layer.weights[0] = kernel;
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

        let mut conv_layer: Conv<f32> = Conv::new((2, 4, 6), 2, 3);
        assert_eq!(conv_layer.weights.len(), 2);
        assert_eq!(conv_layer.weights[0].dim(), (2, 3, 3));
        assert_eq!(conv_layer.bias.dim(), (2, 2, 4));
        conv_layer.weights[0] = kernel_a;
        conv_layer.weights[1] = kernel_b;
        conv_layer.bias = arr3(&[
            [[0., 0., 0., 1.], [0., 0., 0., 1.]],
            [[1., 0., 0., 0.], [1., 0., 0., 0.]],
        ]);

        let out = conv_layer.forward(input);
        assert_eq!(
            out,
            arr3(&[
                [[10., 12., 16., 17.], [8., 12., 14., 13.]],
                [[6., 10., 40., 35.], [21., 20., 35., 25.]]
            ])
            .into_dyn()
        );
    }
}
