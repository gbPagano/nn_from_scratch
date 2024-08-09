use ndarray::ScalarOperand;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive};
use std::fmt::Display;
use std::iter::Sum;
use std::marker::{Send, Sync};
use std::ops::AddAssign;

pub trait FloatNN:
    Float + SampleUniform + FromPrimitive + ScalarOperand + AddAssign + Display + Sum + Send + Sync
{
}

impl FloatNN for f32 {}
impl FloatNN for f64 {}
