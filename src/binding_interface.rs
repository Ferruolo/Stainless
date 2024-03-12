use std::os;
use std::os::raw::c_int;
use std::sync::{Arc, Mutex};
use crate::bindings::{CreateUniformRandomMatrix, location_CPU, location_GPU};
use crate::object::Object;

pub unsafe fn create_uniform_random_mat_interface(shape: Vec<u64>) {
    let num_dim :os::raw::c_int = shape.len() as c_int;
    let mut c_shape = [shape[0] as c_int, shape[1] as c_int];
    
    
    let low: c_int = 0;
    let high: c_int = 10;
    let location = location_GPU;

    let mat = CreateUniformRandomMatrix(c_shape.as_ptr(), num_dim, location, low, high);
}

pub unsafe fn matrix_mul_interface(left: Arc<Mutex<Object>>, right: Arc<Mutex<Object>>) {
    // let l_mat = left.lock().unwrap().get_executable();
    // let r_mat = right.lock().unwrap().get_executable();
}
