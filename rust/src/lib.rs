use ndarray::ScalarOperand;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{float, NumAssign, FromPrimitive};
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::marker::{Send, Sync};
use std::ops::{AddAssign, DivAssign, SubAssign};
use ndarray::DataMut;

pub mod layers;
pub mod loss;
mod neural_network;
pub use neural_network::*;

pub trait Float:
    float::Float
    + SampleUniform
    + FromPrimitive
    + ScalarOperand
    + AddAssign
    + SubAssign
    + DivAssign
    + Display
    + Debug
    + Copy
    + NumAssign
    + Sum
    + Send
    + Sync
{
}
impl Float for f32 {}
impl Float for f64 {}
