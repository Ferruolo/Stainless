use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::classes::{ItemLoc, Operation};
use crate::dep_tree::DepTree;

pub(crate) struct Object {
    loc: ItemLoc,
    name: u64,
    shape: Vec<u64>,
    left: Option<Arc<Mutex<Object>>>,
    right: Option<Arc<Mutex<Object>>>,
    track_deps: bool,
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
        }
    }

    pub fn get_shape(&self) -> &Vec<u64> {
        &self.shape
    }

    pub fn get_op(&self) -> Operation {
        self.forge_op
    }

    pub fn get_left(&self) -> &Option<Arc<Mutex<Object>>> {
        return &self.left;
    }
    pub fn get_right(&self) -> &Option<Arc<Mutex<Object>>> {
        return &self.right;
    }


    pub fn get_name(&self) -> u64 {
        return self.name;
    }

    pub fn get_loc(&self) -> ItemLoc {
        return self.loc;
    }
}

