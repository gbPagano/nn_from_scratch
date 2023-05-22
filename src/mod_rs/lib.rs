use pyo3::prelude::*;
use std::f64;
use ndarray::{Array, Array2, stack, concatenate};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;

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

    // fn backward(
    //     &mut self, 
    //     alpha: f64,
    //     last: bool,
    //     previous_delta: Option<Vec<Vec<f64>>>,
    //     previous_weight: Option<Vec<Vec<f64>>>,
    //     error: Option<Vec<Vec<f64>>>,
    // ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    //    
    //         

    //     let delta = if last {
    //         let error = vec2d_to_ndarray(&error.unwrap());
    //         error * self.net.as_ref().unwrap().mapv(|x| self.function.derivative(x))
    //     } else {
    //         let previous_delta = vec2d_to_ndarray(&previous_delta.unwrap());
    //         let previous_weight = vec2d_to_ndarray(&previous_weight.unwrap());
    //         let delta = previous_delta.dot(&previous_weight).slice(s![.., 1..]).to_owned();
    //         delta * self.net.as_ref().unwrap().mapv(|x| self.function.derivative(x))
    //     };
    //     
    //     
    //     self.weights = delta.t().dot(self.input.as_ref().unwrap()) * alpha + &self.weights;
    //     
    //     (ndarray_to_vec2d(&delta), ndarray_to_vec2d(&self.weights.to_owned()))
    // }



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
