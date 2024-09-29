fn main() {
    use ndarray::prelude::*;
    use ndarray::*;
    use ndarray_conv::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    //let x = Array::random((2000, 4000), Uniform::new(0., 1.));
    //let k = Array::random((9, 9), Uniform::new(0., 1.));

    let x = arr3(&[
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
    ]);
    let k = arr3(&[
        [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]],
        [[0.5, 0., 0.5], [0., 0.5, 0.], [0.5, 0., 0.5]],
    ]);
    let k2 = arr3(&[
        [[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]],
        [[0., 2., 0.], [2., 0., 2.], [0., 2., 0.]],
    ]);


    let mut out = Vec::new();

    let res = x.conv(&k, ConvMode::Valid, PaddingMode::Zeros).unwrap();
    println!("{:?}", res);
    out.push(res);

    let res = x.conv(&k2, ConvMode::Valid, PaddingMode::Zeros).unwrap();
    println!("{:?}", res);
    out.push(res);

    let a: Vec<_> = out.iter().map(|i| i.view()).collect::<Vec<_>>();
    let out = concatenate(Axis(0), &out.iter().map(|i| i.view()).collect::<Vec<_>>());
    println!("{:?}", out);
}
