use std::sync::{Arc, Mutex};
use crate::bindings::{FreeMatrix, Matrix};
use crate::classes::{ItemLoc, Operation};
use crate::binding_interface::RustMatrix;

pub(crate) struct ObjectContents {
    loc: ItemLoc,
    name: u64,
    shape: Vec<u64>,
    left: Option<Object>,
    right: Option<Object>,
    track_deps: bool,
    executable: RustMatrix,
    forge_op: Operation,
}

pub trait ObjectInterface {
    fn init(
        name: u64,
        shape: &Vec<u64>,
        track_deps: bool,
        forge_op: Operation,
        left: Option<Object>,
        right: Option<Object>,
    ) -> Self;
    fn get_shape(&self) -> &Vec<u64>;
    fn get_op(&self) -> Operation;
    fn get_left(&self) -> Option<Object>;
    fn get_right(&self) -> Option<Object>;
    fn get_name(&self) -> u64;
    fn get_loc(&self) -> ItemLoc;
    fn set_matrix(&mut self, mat: *mut Matrix);
    fn get_executable(&self) -> *mut Matrix;
}



impl ObjectInterface for ObjectContents {
    fn init(
        name: u64,
        shape: &Vec<u64>,
        track_deps: bool,
        forge_op: Operation,
        left: Option<Object>,
        right: Option<Object>,
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

    fn get_shape(&self) -> &Vec<u64> {
        &self.shape
    }

    fn get_op(&self) -> Operation {
        self.forge_op
    }

    fn get_left(&self) -> Option<Object> {
        match &self.left {
            None => {None}
            Some(l) => {Some(l.clone())}
        }
    }


    fn get_right(&self) -> Option<Object> {
        match &self.right {
            None => {None}
            Some(r) => {Some(r.clone())}
        }
    }

    fn get_name(&self) -> u64 {
        return self.name;
    }

    fn get_loc(&self) -> ItemLoc {
        return self.loc;
    }

    fn set_matrix(&mut self, mat: *mut Matrix) {
        self.executable.set_gpu_mat(mat);
    }

    fn get_executable(&self) -> *mut Matrix {
        return self.executable.get_gpu_mat();
    }

}

impl PartialEq for ObjectContents {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }

    fn ne(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Drop for ObjectContents {
    fn drop(&mut self) {
        let mat = self.executable.get_gpu_mat();
        unsafe { FreeMatrix(mat); }
    }
}

#[derive(Clone)]
pub struct Object {
    contains: Arc<Mutex<ObjectContents>>
}


impl ObjectInterface for Object {
    fn init(name: u64, shape: &Vec<u64>, track_deps: bool, forge_op: Operation, left: Option<Object>, right: Option<Object>) -> Self {
        return Self {
            contains: Arc::new(Mutex::new(ObjectContents::init(
                name, shape, track_deps, forge_op, left, right,
            ))),
        }
    }

    fn get_shape(&self) -> &Vec<u64> {
        return self.contains.lock().unwrap().get_shape();
    }

    fn get_op(&self) -> Operation {
        return self.contains.lock().unwrap().get_op();
    }

    fn get_left(&self) -> Option<Object> {
        self.contains.lock().unwrap().get_left()
    }

    fn get_right(&self) -> Option<Object> {
        self.contains.lock().unwrap().get_right()
    }

    fn get_name(&self) -> u64 {
        return self.contains.lock().unwrap().get_name();
    }

    fn get_loc(&self) -> ItemLoc {
        return self.contains.lock().unwrap().get_loc();
    }

    fn set_matrix(&mut self, mat: *mut Matrix) {
        self.contains.lock().unwrap().set_matrix(mat)
    }

    fn get_executable(&self) -> *mut Matrix {
        self.contains.lock().unwrap().get_executable()
    }
}
