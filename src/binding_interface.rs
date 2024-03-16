use std::os;
use std::os::raw::c_int;
use std::sync::{Arc, Mutex};
use crate::binding_interface::RustMatType::{ComputationPending, GpuMatrix};
use crate::bindings::{CreateUniformRandomMatrix, location_GPU, MatMul, Matrix};
use crate::object::Object;

pub enum RustMatType {
    GpuMatrix,
    CpuMatrix,
    DiskItem,
    ComputationPending,
    None,
}



pub struct RustMatrix {
    mode: RustMatType,
    gpu_matrix: Option<*mut Matrix>


}
impl RustMatrix {
    pub fn init() -> Self {
        return Self {
            mode: ComputationPending,
            gpu_matrix: None
        }
    }

    pub fn set_gpu_mat(&mut self, mat: *mut Matrix) {
        self.mode = GpuMatrix;
        self.gpu_matrix = Some(mat);
    }


    pub fn get_gpu_mat(&self) -> *mut Matrix {
        if let Some(mat) = self.gpu_matrix {
            return mat;
        } else {
            panic!("No GPU Matrix")
        }
    }
}


unsafe impl Send for RustMatrix {}



pub unsafe fn create_uniform_random_mat_interface(shape: &Vec<u64>) -> *mut Matrix {
    let num_dim :os::raw::c_int = shape.len() as c_int;
    let c_shape = [shape[0] as c_int, shape[1] as c_int];
    
    
    let low: c_int = 0;
    let high: c_int = 10;
    let location = location_GPU;

    let mat = CreateUniformRandomMatrix(c_shape.as_ptr(), num_dim, location, low, high);
    return mat;
}

pub unsafe fn matrix_mul_interface(left: Arc<Mutex<Object>>, right: Arc<Mutex<Object>>) -> *mut Matrix {
    let l_mat = left.lock().unwrap().get_executable();
    let r_mat = right.lock().unwrap().get_executable();
    let new_mat = MatMul(l_mat, r_mat);
    return new_mat;
}
