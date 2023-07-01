use pyo3::prelude::*;
use std::f64;
use ndarray::{Array, Array2, concatenate};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use kdam::tqdm;
use itertools::izip;
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn mod_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ActivationFunction>()?;
    m.add_class::<Layer>()?;
    m.add_class::<NeuralNetwork>()?;
    Ok(())
}

#[pyclass]
#[derive(Clone)]
enum ActivationFunction {
    Tanh,
}

#[pymethods]
impl ActivationFunction {
    fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Tanh => self.tanh(x),
        }
    }
    
    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Tanh => self.tanh_derivative(x),
        }
    }
    
    fn tanh(&self, x: f64) -> f64 {
        x.tanh()
    }
    
    fn tanh_derivative(&self, x: f64) -> f64 {
        1.0 - self.tanh(x).powi(2)
    }
}




#[pyclass]
#[derive(FromPyObject)]
struct Layer {
    #[pyo3(get)]
    len_inputs: usize,
    #[pyo3(get)]
    neurons: usize,
    #[pyo3(get)]
    function: ActivationFunction,
    #[pyo3(get)]
    weights: Vec<Vec<f64>>,
    #[pyo3(get)]
    input: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    net: Option<Vec<Vec<f64>>,>,
    #[pyo3(get)]
    output: Option<Vec<Vec<f64>>,>,
    #[pyo3(get)]
    idx: Option<usize>,
}



#[pymethods]
impl Layer {
    #[new]
    fn new(len_inputs: usize, neurons: usize, function: ActivationFunction) -> PyResult<Self> {
        let shape = (neurons, len_inputs + 1);
        let weights = Array::random(shape, Uniform::new(-0.5, 0.5));

        Ok(Layer {
            len_inputs: len_inputs,
            neurons: neurons,
            function: function,
            weights: ndarray_to_vec2d(&weights),
            input: None,
            net: None,
            output: None,
            idx: None,
        })
    }
    
    fn forward(&mut self, layer_input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        
        let weights = vec2d_to_ndarray(&self.weights);

        self.input = Some(layer_input);
        let input = vec2d_to_ndarray(&self.input.as_ref().unwrap());
        self.net = Some(ndarray_to_vec2d(&input.dot(&weights.t())));
        let net = vec2d_to_ndarray(&self.net.as_ref().unwrap());
        self.output = Some(ndarray_to_vec2d(&net.mapv(|x| self.function.activate(x))));
        self.output.clone().unwrap()
    }

    fn backward(
        &mut self, 
        alpha: f64,
        last: bool,
        previous_delta: Option<Vec<Vec<f64>>>,
        previous_weight: Option<Vec<Vec<f64>>>,
        error: Option<Vec<Vec<f64>>>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
       
        let net = vec2d_to_ndarray(&self.net.as_ref().unwrap());

        let delta = if last {
            let error = vec2d_to_ndarray(&error.unwrap());
            error * net.mapv(|x| self.function.derivative(x))
        } else {
            let previous_delta = vec2d_to_ndarray(&previous_delta.unwrap());
            let previous_weight = vec2d_to_ndarray(&previous_weight.unwrap());
            let delta = previous_delta.dot(&previous_weight).slice(s![.., 1..]).to_owned();
            delta * net.mapv(|x| self.function.derivative(x))
        };
        
        let input = vec2d_to_ndarray(&self.input.as_ref().unwrap());
        let weights = vec2d_to_ndarray(&self.weights);

        self.weights = ndarray_to_vec2d(&(delta.t().dot(&input) * alpha + &weights));
        
        (ndarray_to_vec2d(&delta), self.weights.to_owned().clone())
    }



    fn set_idx(&mut self, idx: usize) {
        self.idx = Some(idx);
    }

}

#[pyclass]
struct NeuralNetwork {
    layers: Vec<Layer>,
    #[pyo3(get)]
    all_mse: Vec<f64>,
}

#[pymethods]
impl NeuralNetwork {
    #[new]
    fn new(mut layers: Vec<Layer>) -> Self {
        for (idx, layer) in layers.iter_mut().enumerate() {
            layer.set_idx(idx + 1);
        }

        NeuralNetwork {
            layers,
            all_mse: Vec::new(),
        }
    }

    fn forward(&mut self, x_input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {


        let x_input = vec2d_to_ndarray(&x_input);

        let mut input_layer = concatenate![Axis(1), Array::from_shape_vec((1, 1), vec![1.0]).unwrap(), x_input.clone()];
        let mut output: Array2<f64> = Array::zeros((0, 0));

        

        for layer in &mut self.layers {
            output = vec2d_to_ndarray(&layer.forward(ndarray_to_vec2d(&input_layer)));
            input_layer = concatenate![Axis(1), Array::from_shape_vec((1, 1), vec![1.0]).unwrap(), output];
        }

        ndarray_to_vec2d(&output)
    }

    fn backward(&mut self, alpha: f64, error: Vec<Vec<f64>>) {


        let mut previous_delta = None;
        let mut previous_weight = None;
        let mut last = true;
        
        for layer in self.layers.iter_mut().rev() {
            let (delta, weights) = layer.backward(alpha, last, previous_delta, previous_weight, Some(error.clone()));
            last = false;
            previous_delta = Some(delta);
            previous_weight = Some(weights);
        }
    }

    fn get_weights(&self) -> Vec<Vec<Vec<f64>>> {
        let mut weights = Vec::new();
        for layer in &self.layers {
            weights.push(layer.weights.clone());
        }
        weights
    }

    fn predict(&mut self, x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.forward(x)
    }

    fn fit(
        &mut self, 
        x_train: Vec<Vec<Vec<f64>>>, 
        y_train: Vec<Vec<Vec<f64>>>,
        epochs: i32,
        alpha: f64,
    ) {
        
        for epoch in tqdm!(0..epochs) {
            let mut errors = Vec::new();
            for (x, y) in izip!(&x_train, &y_train) {
                let out = self.forward(x.to_vec());
                let error = subtract_vectors(y.to_vec(), out);
                errors.push(sum_rows(square_matrix(&error)));
                self.backward(alpha, error);
            }
            
            let sum_errors = sum_rows(errors);
            let mse = calculate_mean(sum_errors);

            if epoch % 100 == 0 {
                println!("MSE: {}", mse);
            }

            self.all_mse.push(mse);
        }

    }


}

fn sum_rows(matrix: Vec<Vec<f64>>) -> Vec<f64> {
    let mut sums = Vec::new();

    for row in matrix {
        let row_sum: f64 = row.iter().sum();
        sums.push(row_sum);
    }

    sums
}


fn calculate_mean(numbers: Vec<f64>) -> f64 {
    let sum: f64 = numbers.iter().sum();
    let count = numbers.len() as f64;
    sum / count
}


fn square_matrix(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();

    for row in matrix {
        let mut squared_row = Vec::new();

        for &element in row {
            let squared_element = element * element;
            squared_row.push(squared_element);
        }

        result.push(squared_row);
    }

    result
}



fn subtract_vectors(v1: Vec<Vec<f64>>, v2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {


    let mut result = Vec::new();

    for i in 0..v1.len() {
        let mut row = Vec::new();
        for j in 0..v1[i].len() {
            let element = v1[i][j] - v2[0][0];
            row.push(element);
        }
        result.push(row);
    }

    result
}


fn vec2d_to_ndarray(vec2d: &Vec<Vec<f64>>) -> Array2<f64> {
    let rows = vec2d.len();
    let cols = vec2d[0].len();
    
    let mut arr = Array::zeros((rows, cols));
    
    for (i, row) in vec2d.iter().enumerate() {
        for (j, &element) in row.iter().enumerate() {
            arr[[i, j]] = element;
        }
    }
    
    arr
}


fn ndarray_to_vec2d(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    let mut vec2d: Vec<Vec<f64>> = Vec::new();
    
    for row in arr.outer_iter() {
        let row_vec: Vec<f64> = row.iter().cloned().collect();
        vec2d.push(row_vec);
    }
    
    vec2d
}
