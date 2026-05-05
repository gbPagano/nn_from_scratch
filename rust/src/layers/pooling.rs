extern crate blas_src;

use ndarray::*;

use super::super::Float;
use super::Layer;

pub struct MaxPooling {
    max_indexes: Array4<usize>,
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
            max_indexes: Array4::zeros((0, 0, 0, 0)),
            stride,
            kernel_size,
            output_shape,
            input_shape,
        }
    }
}

impl<F: Float> Layer<F> for MaxPooling {
    fn forward(&mut self, input: ArrayD<F>) -> ArrayD<F> {
        let input = input.into_dimensionality::<Ix4>().unwrap();
        let b = input.shape()[0];
        let mut output: Array4<F> = Array::zeros((
            b,
            self.output_shape.0,
            self.output_shape.1,
            self.output_shape.2,
        ));
        self.max_indexes = Array4::zeros(output.raw_dim());

        let kernel_size = self.kernel_size;
        let stride = self.stride;
        let in_h = self.input_shape.1;
        let in_w = self.input_shape.2;
        let (out_d, out_h, out_w) = self.output_shape;

        Zip::from(input.axis_iter(Axis(0)))
            .and(output.axis_iter_mut(Axis(0)))
            .and(self.max_indexes.axis_iter_mut(Axis(0)))
            .par_for_each(|input_sample, mut output_sample, mut idx_sample| {
                for d in 0..out_d {
                    let input_2d = input_sample.slice(s![d, .., ..]);
                    for i in 0..out_h {
                        for j in 0..out_w {
                            let start_h = i * stride;
                            let start_w = j * stride;
                            let end_h = (start_h + kernel_size).min(in_h);
                            let end_w = (start_w + kernel_size).min(in_w);
                            let mut max_value = F::from_f32(f32::NEG_INFINITY).unwrap();
                            let mut max_idx = 0;
                            for h in start_h..end_h {
                                for w in start_w..end_w {
                                    let val = input_2d[[h, w]];
                                    if val > max_value {
                                        max_value = val;
                                        max_idx = h * in_w + w;
                                    }
                                }
                            }
                            output_sample[[d, i, j]] = max_value;
                            idx_sample[[d, i, j]] = max_idx;
                        }
                    }
                }
            });

        output.into_dyn()
    }

    fn backward(&mut self, output_gradient: ArrayD<F>, _learning_rate: F) -> ArrayD<F> {
        let output_gradient = output_gradient.into_dimensionality::<Ix4>().unwrap();
        let b = output_gradient.shape()[0];
        let mut input_gradient: Array4<F> = Array::zeros((
            b,
            self.input_shape.0,
            self.input_shape.1,
            self.input_shape.2,
        ));
        let in_w = self.input_shape.2;
        let in_d = self.input_shape.0;
        let (_, out_h, out_w) = self.output_shape;

        Zip::from(input_gradient.axis_iter_mut(Axis(0)))
            .and(output_gradient.axis_iter(Axis(0)))
            .and(self.max_indexes.axis_iter(Axis(0)))
            .par_for_each(|mut input_grad_sample, output_grad_sample, idx_sample| {
                for d in 0..in_d {
                    for i in 0..out_h {
                        for j in 0..out_w {
                            let max_idx = idx_sample[[d, i, j]];
                            let grad_val = output_grad_sample[[d, i, j]];
                            let row = max_idx / in_w;
                            let col = max_idx % in_w;
                            input_grad_sample[[d, row, col]] += grad_val;
                        }
                    }
                }
            });
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
        .insert_axis(Axis(0))
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 2, 2);
        let output = layer.forward(input.clone());
        assert_eq!(
            output,
            array![
                [[6.0, 8.0, 9.0], [14.0, 16.0, 17.0], [18.0, 20.0, 21.0]],
                [[26.0, 28.0, 29.0], [34.0, 36.0, 37.0], [38.0, 40.0, 41.0]]
            ]
            .insert_axis(Axis(0))
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
        .insert_axis(Axis(0))
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 2, 2);
        layer.forward(input.clone());

        let grad = array![
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
        .insert_axis(Axis(0))
        .into_dyn();

        let output = layer.backward(grad, 1.0);
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
            .insert_axis(Axis(0))
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
        .insert_axis(Axis(0))
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 3, 2);
        let output = layer.forward(input.clone());
        assert_eq!(
            output,
            array![
                [[11.0, 13.0, 13.0], [19.0, 21.0, 21.0], [19.0, 21.0, 21.0]],
                [[31.0, 33.0, 33.0], [39.0, 41.0, 41.0], [39.0, 41.0, 41.0]]
            ]
            .insert_axis(Axis(0))
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
        .insert_axis(Axis(0))
        .into_dyn();

        let mut layer = MaxPooling::new((2, 5, 5), 3, 2);
        layer.forward(input.clone());

        let grad = array![
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
        .insert_axis(Axis(0))
        .into_dyn();

        let output = layer.backward(grad, 1.0);
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
            .insert_axis(Axis(0))
            .into_dyn()
        );
    }
}
