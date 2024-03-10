use std::os;
use std::os::raw::c_int;
use crate::bindings::{CreateUniformRandomMatrix, location_CPU, location_GPU};

pub unsafe fn matrix_factor_interface(shape: Vec<u64>) {
    let num_dim :os::raw::c_int = shape.len() as c_int;
    let mut c_shape = [shape[0] as c_int, shape[1] as c_int];
    
    
    let low: c_int = 0;
    let high: c_int = 10;
    let location = location_GPU;

    let mat = CreateUniformRandomMatrix(c_shape.as_ptr(), num_dim, location, low, high);
}
