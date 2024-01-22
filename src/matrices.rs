extern crate libc;
use crate::bindings::{CpuMatrix, CreateOnesMatrix};
use libc::c_int;


pub enum Matrix {
    CpuMatrix(*const CpuMatrix),
    GpuMatrix(())
}


pub fn create_ones_matrix_rust(num_dim: i32, shape: Vec<i32>) -> *const CpuMatrix {
    let mut v:Vec<c_int> = vec![];
    for i in shape {
        let i_int = i as c_int;
        v.push(i_int);
    }

    return unsafe {CreateOnesMatrix(num_dim as c_int, v.as_ptr())};
}

