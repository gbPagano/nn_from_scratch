extern crate blas_src;

use ndarray::*;

use super::super::Float;
use super::Layer;

pub struct MaxPooling {
    max_indexes: Array3<usize>,
    kernel_size: usize,
    stride: usize,
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
}

impl MaxPooling {
    pub fn new(input_shape: (usize, usize, usize), kernel_size: usize, stride: usize) -> Self {
        assert!(stride <= kernel_size);
        let output_shape = (
            input_shape.0,
            ((input_shape.1 - 1) / stride) + 1,
            ((input_shape.2 - 1) / stride) + 1,
        );

        Self {
            max_indexes: arr3(&[[[]]]),
            stride,
            kernel_size,
            output_shape,
            input_shape,
        }
    }
}

impl<F: Float> Layer<F> for MaxPooling {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        let input = if input.ndim() == 3 {
            input.into_dimensionality::<Ix3>().unwrap()
        } else {
            input
                .into_dimensionality::<Ix2>()
                .unwrap()
                .insert_axis(Axis(0))
        };

        let mut output: Array3<F> = Array::zeros(self.output_shape);
        self.max_indexes = Array::zeros(self.output_shape);
        for d in 0..self.output_shape.0 {
            let input_2d = input.slice(s![d, .., ..]);

            for i in 0..self.output_shape.1 {
                for j in 0..self.output_shape.2 {
                    let start_h = i * self.stride;
                    let start_w = j * self.stride;

                    let end_h = (start_h + self.kernel_size).min(self.input_shape.1);
                    let end_w = (start_w + self.kernel_size).min(self.input_shape.2);

                    let pool_region = input_2d.slice(s![start_h..end_h, start_w..end_w]);
                    let (max_value, max_idx) = pool_region.indexed_iter().fold(
                        (F::from_f32(f32::NEG_INFINITY).unwrap(), 0),
                        |(max_val, max_idx), (idx, &val)| {
                            if val > max_val {
                                (
                                    val,
                                    (start_h + idx.0) * self.input_shape.2 + start_w + idx.1,
                                )
                            } else {
                                (max_val, max_idx)
                            }
                        },
                    );

                    output[[d, i, j]] = max_value;
                    self.max_indexes[[d, i, j]] = max_idx
                }
            }
        }

        output.into_dyn()
    }

    fn backward(
        &mut self,
        output_gradient: ArrayD<F>,
        _learning_rate: F,
        _batch_size: usize,
    ) -> ArrayD<F> {
        let mut input_gradient: Array3<F> = Array::zeros(self.input_shape);
        for d in 0..self.input_shape.0 {
            for i in 0..self.output_shape.1 {
                for j in 0..self.output_shape.2 {
                    let max_idx = self.max_indexes[[d, i, j]];
                    let grad_val = output_gradient[[d, i, j]];

                    let row = max_idx / self.input_shape.2;
                    let col = max_idx % self.input_shape.2;

                    input_gradient[[d, row, col]] += grad_val;
                }
            }
        }
        input_gradient.into_dyn()
    }

    fn get_weights(&self) -> Option<ArrayD<F>> {
        None
    }

    fn get_bias(&self) -> Option<ArrayD<F>> {
        None
    }
}

impl<F: Float> From<MaxPooling> for Box<dyn Layer<F>> {
    fn from(item: MaxPooling) -> Self {
        Box::new(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_pooling_forward_a() {
        let input: ArrayD<f32> = array![
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [17.0, 18.0, 19.0, 20.0, 21.0]
            ],
            [
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [29.0, 30.0, 31.0, 32.0, 33.0],
                [33.0, 34.0, 35.0, 36.0, 37.0],
                [37.0, 38.0, 39.0, 40.0, 41.0]
            ]
        ]
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 2, 2);
        let output = layer.forward(input.clone());
        assert_eq!(
            output,
            array![
                [[6.0, 8.0, 9.0], [14.0, 16.0, 17.0], [18.0, 20.0, 21.0]],
                [[26.0, 28.0, 29.0], [34.0, 36.0, 37.0], [38.0, 40.0, 41.0]]
            ]
            .into_dyn()
        );
    }

    #[test]
    fn max_pooling_backward_a() {
        let input: ArrayD<f32> = array![
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [17.0, 18.0, 19.0, 20.0, 21.0]
            ],
            [
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [29.0, 30.0, 31.0, 32.0, 33.0],
                [33.0, 34.0, 35.0, 36.0, 37.0],
                [37.0, 38.0, 39.0, 40.0, 41.0]
            ]
        ]
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 2, 2);
        layer.forward(input.clone());

        let grad = array![
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
        .into_dyn();

        let output = layer.backward(grad, 1.0, 1);
        assert_eq!(
            output,
            array![
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0]
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0]
                ]
            ]
            .into_dyn()
        );
    }

    #[test]
    fn max_pooling_forward_b() {
        let input: ArrayD<f32> = array![
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [17.0, 18.0, 19.0, 20.0, 21.0]
            ],
            [
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [29.0, 30.0, 31.0, 32.0, 33.0],
                [33.0, 34.0, 35.0, 36.0, 37.0],
                [37.0, 38.0, 39.0, 40.0, 41.0]
            ]
        ]
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 3, 2);
        let output = layer.forward(input.clone());
        assert_eq!(
            output,
            array![
                [[11.0, 13.0, 13.0], [19.0, 21.0, 21.0], [19.0, 21.0, 21.0]],
                [[31.0, 33.0, 33.0], [39.0, 41.0, 41.0], [39.0, 41.0, 41.0]]
            ]
            .into_dyn()
        );
    }

    #[test]
    fn max_pooling_backward_b() {
        let input: ArrayD<f32> = array![
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [17.0, 18.0, 19.0, 20.0, 21.0]
            ],
            [
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [29.0, 30.0, 31.0, 32.0, 33.0],
                [33.0, 34.0, 35.0, 36.0, 37.0],
                [37.0, 38.0, 39.0, 40.0, 41.0]
            ]
        ]
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 3, 2);
        layer.forward(input.clone());

        let grad = array![
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
        .into_dyn();

        let output = layer.backward(grad, 1.0, 1);
        assert_eq!(
            output,
            array![
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 4.0]
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 4.0]
                ]
            ]
            .into_dyn()
        );
    }
}
