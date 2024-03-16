use std::sync::{Arc, Mutex};
// use crate::binding_interface::RustMatrix;
use crate::bindings::Matrix;
use crate::classes::{ItemLoc, Operation};
use crate::binding_interface::RustMatrix;

pub(crate) struct Object {
    loc: ItemLoc,
    name: u64,
    shape: Vec<u64>,
    left: Option<Arc<Mutex<Object>>>,
    right: Option<Arc<Mutex<Object>>>,
    track_deps: bool,
    executable: RustMatrix,
    forge_op: Operation,
}

impl Object {
    pub(crate) fn init(
        name: u64,
        shape: &Vec<u64>,
        track_deps: bool,
        forge_op: Operation,
        left: Option<Arc<Mutex<Object>>>,
        right: Option<Arc<Mutex<Object>>>,
    ) -> Self {
        Self {
            loc: ItemLoc::CPU,
            name,
            shape: shape.clone(),
            left,
            right,
            track_deps,
            forge_op,
            executable: RustMatrix::init(),
        }
    }

    pub fn get_shape(&self) -> &Vec<u64> {
        &self.shape
    }

    pub fn get_op(&self) -> Operation {
        self.forge_op
    }

    pub fn get_left(&self) -> Option<Arc<Mutex<Object>>> {
        match &self.left {
            None => {None}
            Some(l) => {Some(Arc::clone(l))}
        }
    }

    pub fn get_right(&self) -> Option<Arc<Mutex<Object>>> {
        match &self.right {
            None => {None}
            Some(r) => {Some(Arc::clone(r))}
        }
    }

    pub fn get_name(&self) -> u64 {
        return self.name;
    }

    pub fn get_loc(&self) -> ItemLoc {
        return self.loc;
    }

    pub fn set_matrix(&mut self, mat: *mut Matrix) {
        self.executable.set_gpu_mat(mat);
    }

    pub fn get_executable(&self) -> *mut Matrix {
        return self.executable.get_gpu_mat();
    }

}

impl PartialEq for Object {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }

    fn ne(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

